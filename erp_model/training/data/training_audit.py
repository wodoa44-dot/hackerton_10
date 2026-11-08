"""Audit training-label eligibility before fitting; never reads inference targets.

Run: python -m erp_model.training.data.training_audit --artifact-dir artifacts/generation_v6
All issues are review flags. Source records and model weights are preserved.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from erp_model.encoders.candidates import candidates
from erp_model.prediction.code_metrics import numeric_checks
from erp_model.encoders.rules.company_schema import CompanySchemas, actual_prefix, field_slot
from erp_model.training.data.manual_audit import capacity_modes
from erp_model.encoders.rules.material_schema import silver_family
from erp_model.encoders.preprocess import preprocess
from erp_model.shared.schema import SLOTS
from erp_model.training.io import read_csv


from erp_model.encoders.rules.label_quality import conflicting_indices, explicit_resistance


def audit_training(train_path, artifact_dir, output_dir=None):
    base = Path(artifact_dir)
    out = Path(output_dir) if output_dir else base / "training_audit"
    raw = read_csv(train_path)
    folds = {}
    with (base / "fold_assignments.csv").open(encoding="utf-8-sig") as stream:
        assignments = list(csv.DictReader(stream))
    cache = {}
    prepared = []
    for row in raw:
        key = (row["품목명"], row["규격명"])
        if key not in cache:
            cache[key] = preprocess(*key)
        prepared.append(cache[key])
    keys = [p.key for p in prepared]
    for assignment in assignments:
        i = int(assignment["source_row"]) - 2
        if i in folds or not 0 <= i < len(raw):
            raise ValueError("Invalid or duplicate training source row in fold assignments")
        if tuple(assignment[k] for k in ["normalized_name", "normalized_spec"]) != keys[i]:
            raise ValueError("Fold assignments do not match current training inputs")
        if assignment["classification"] != actual_prefix(raw[i]["품목코드"]):
            raise ValueError("Fold assignments do not match current training codes")
        folds[i] = int(assignment["fold"])
    keep = sorted(folds)
    book = CompanySchemas(base / "company_rules.json")
    inspections = [book.inspect(row["품목코드"]) for row in raw]
    parsed = [book.parse(row["품목코드"]) for row in raw]
    labels = {slot: [actual_prefix(r["품목코드"]) if slot == "classification" else
                     (p["values"][slot] if p else "") for r, p in zip(raw, parsed)] for slot in SLOTS}
    retained = set(keep)
    row_details = []
    issues = []
    def issue(i, kind, slot="", detail=None):
        issues.append({"source_row": i + 2, **{k: raw[i][k] for k in ["품목코드", "품목명", "규격명"]},
                       "issue_type": kind, "slot": slot, "details": json.dumps(detail, ensure_ascii=False)})
    slots = {}
    included_by_slot = {}
    reasons_by_slot = {}
    for slot in SLOTS:
        eligible = []
        reasons = {}
        for i in range(len(raw)):
            if i not in retained:
                reasons[i] = "outside_retained_training"
            elif slot != "classification" and not parsed[i]:
                reasons[i] = inspections[i]["status"]
            elif not labels[slot][i]:
                reasons[i] = "not_applicable"
            elif slot == "part_number" and not any(s["key"] in {"pn", "sub", "model", "name"}
                 for s in book.layout(parsed[i]["category_id"])[0][slot]):
                reasons[i] = "deterministic_numeric_encoding"
            else:
                eligible.append(i)
        conflicts, conflict_keys = conflicting_indices(eligible, keys, labels[slot])
        for i in conflicts:
            reasons[i] = "conflicting_input_labels"
            issue(i, "conflicting_input_labels", slot, {"label": labels[slot][i]})
        included = [i for i in eligible if i not in conflicts]
        included_by_slot[slot] = included
        reasons_by_slot[slot] = reasons
        counts = Counter(labels[slot][i] for i in included)
        groups = defaultdict(set)
        for i in included:
            groups[labels[slot][i]].add(keys[i])
        absent_oof = []
        for fold in sorted(set(folds.values())):
            train_labels = {labels[slot][i] for i in included if folds[i] != fold}
            absent_oof.extend(i for i in included if folds[i] == fold and labels[slot][i] not in train_labels)
        slots[slot] = {
            "eligible_before_conflict_filter": len(eligible), "model_label_rows": len(included),
            "distinct_labels": len(counts), "unique_input_groups": len({keys[i] for i in included}),
            "excluded_conflicting_rows": len(conflicts), "conflicting_input_groups": conflict_keys,
            "placeholder_rows": sum(v and not v.strip("-") for v in (labels[slot][i] for i in included)),
            "labels_in_only_one_input_group": sum(len(v) == 1 for v in groups.values()),
            "oof_truth_absent_from_training_rows": len(absent_oof),
            "oof_closed_class_limit_applies": slot != "part_number",
            "exclusion_counts_retained": dict(Counter(reasons[i] for i in keep if i in reasons)),
        }
    whole_bad, whole_groups = conflicting_indices(keep, keys, [r["품목코드"] for r in raw])
    numeric_rows = set()
    numeric_total = 0
    numeric_fields = Counter()
    family_rows = set()
    attrs = defaultdict(list)
    route_values = {}
    for i in keep:
        if not raw[i]["품목명"].strip() or not raw[i]["규격명"].strip():
            issue(i, "missing_input", detail={"name_empty": not raw[i]["품목명"].strip(), "spec_empty": not raw[i]["규격명"].strip()})
        record = parsed[i]
        if not record:
            continue
        category = record["category_id"]
        route_values[i] = category
        family, _ = silver_family(prepared[i])
        allowed = book.allowed_categories(family)
        if allowed is not None and category not in allowed:
            family_rows.add(i)
            issue(i, "input_family_schema_disagreement", detail={"name_family": family, "schema": category, "allowed": allowed})
        checks = numeric_checks(book, raw[i]["품목코드"], prepared[i], capacity_mode="pf", resistance_reader=explicit_resistance)
        if category in {"CE", "CM"}:
            # Both documented/observed unit conventions are accepted for data review.
            modes = capacity_modes(raw[i], record)
            if modes:
                checks = [c for c in checks if c["field"] != "cap"]
        numeric_total += len(checks)
        for check in checks:
            if not check["consistent"]:
                numeric_rows.add(i)
                numeric_fields[check["field"]] += 1
                affected = field_slot(book.categories[category], check["field"])
                issue(i, "numeric_input_code_disagreement", affected, check)
        for field in book.categories[category]["fields"]:
            if field["t"] != "sel" and not (field["k"] == "ver" and field["n"] == 1):
                continue
            slot = field_slot(book.categories[category], field["k"])
            if slot == "part_number":
                continue
            part = next(p for p in book.parts(category, slot, record["values"][slot]) if p["key"] == field["k"])
            attrs[field["k"]].append((i, category, part["value"]))
    attr_report = {}
    for field, items in attrs.items():
        grouped = defaultdict(set)
        for i, category, value in items:
            grouped[(keys[i], category)].add(value)
        bad = {key for key, values in grouped.items() if len(values) > 1}
        affected = [i for i, category, _ in items if (keys[i], category) in bad]
        attr_report[field] = {"current_rows": len(items), "conflicting_rows": len(affected), "conflicting_groups": len(bad)}
        for i in affected:
            issue(i, "attribute_conflicting_input_labels", detail={"field": field, "schema": parsed[i]["category_id"]})
    router_bad, router_groups = conflicting_indices(list(route_values), keys, route_values)
    for i in router_bad:
        issue(i, "router_conflicting_input_labels", "classification", {"schema": route_values[i]})
    candidate_covered = set()
    candidate_mixed = set()
    for i in included_by_slot["part_number"]:
        pool = candidates(prepared[i].normalized)
        values = {c["value"] for c in pool}
        if labels["part_number"][i] in values:
            candidate_covered.add(i)
            if len(pool) > 1:
                candidate_mixed.add(i)
        else:
            issue(i, "identifier_truth_missing_from_input_candidates", "part_number", {"label": labels["part_number"][i]})
    slots["part_number"].update({
        "candidate_covered_rows": len(candidate_covered), "actual_rank_training_groups": len(candidate_mixed),
        "candidate_missing_rows": len(included_by_slot["part_number"]) - len(candidate_covered)})
    by_key_folds = defaultdict(set)
    for i in keep:
        by_key_folds[keys[i]].add(folds[i])
    cross_fold = sum(len(values) > 1 for values in by_key_folds.values())
    if cross_fold:
        raise ValueError("Normalized input groups cross validation folds")
    trained_counts = json.loads((base / "training_counts.json").read_text())
    count_matches = {slot: slots[slot]["model_label_rows"] == trained_counts[slot]["rows"] for slot in SLOTS}
    for i, row in enumerate(raw):
        d = {"source_row": i + 2, **row, "retained_training": i in retained,
             "fold": folds.get(i, ""), "manual_status": inspections[i]["status"],
             "manual_schema": inspections[i]["category_id"] or "",
             "manual_pages": ",".join(map(str, (parsed[i] or {}).get("manual_pages", []))),
             "numeric_review": i in numeric_rows, "family_review": i in family_rows,
             "whole_code_conflicting_input": i in whole_bad}
        for slot in SLOTS:
            d[slot + "_label"] = labels[slot][i]
            d[slot + "_eligible"] = i in included_by_slot[slot]
            d[slot + "_reason"] = reasons_by_slot[slot].get(i, "included")
        row_details.append(d)
    report = {
        "training_only": True, "inference_targets_read": False, "weights_modified": False,
        "training_csv": str(train_path), "training_sha256": hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256((base / "company_rules.json").read_bytes()).hexdigest(),
        "source_rows": len(raw), "retained_training_rows": len(keep),
        "excluded_before_fit": len(raw) - len(keep),
        "retained_manual_status": dict(Counter(inspections[i]["status"] for i in keep)),
        "cross_fold_normalized_input_groups": cross_fold,
        "whole_code_conflicting_rows": len(whole_bad), "whole_code_conflicting_input_groups": whole_groups,
        "numeric_review_rows": len(numeric_rows), "numeric_checks": numeric_total, "numeric_conflicts_by_field": dict(numeric_fields),
        "family_schema_review_rows": len(family_rows), "slots": slots,
        "attribute_models": attr_report,
        "router": {"current_rows": len(route_values), "conflicting_rows": len(router_bad), "conflicting_groups": router_groups},
        "reconciles_with_saved_training_counts": count_matches,
        "review_issue_counts": dict(Counter(item["issue_type"] for item in issues)),
        "interpretation": [
            "Code-table validity does not prove agreement between input description and code.",
            "Numeric/family disagreements are review flags, not confirmed source errors.",
            "Unknown manual layouts are unavailable semantic labels, not invalid source codes.",
            "PN labels absent from the input-only pool cannot train the current ranker.",
            "Single-group PN strings are expected for an open candidate ranker, not closed classes.",
            "Attribute conflict checks use (normalized input, schema, field); router checks use normalized input."
        ]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, records in [("training_rows.csv", row_details), ("training_review.csv", issues)]:
        if not records:
            continue
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset/ERP_원재료_training.csv")
    parser.add_argument("--artifact-dir", default="artifacts/generation_v6")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report = audit_training(args.train, args.artifact_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False))
