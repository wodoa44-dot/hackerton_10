"""Compare identical code-generation pipelines with/without Char-CNN features.

Both variants produce ten unique full-code hypotheses before any target is
inspected. Every row stays in its fixed denominator, including missing output.
"""
import argparse
import csv
import gc
import json
import time
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .accuracy_candidates import candidates
from .company_schema import actual_prefix
from .evaluation_audit import audit_slots, displayed_slot
from .exploratory import coverage_curve, effective_codes
from .preprocess import preprocess
from .registry import Registry
from .schema import SLOTS
from .spectrum import SpectrumPredictor, describe_code
from .training_io import read_csv, write_json
from .train_encoder_augmented import COMMON_FILES, sha256


KS = tuple(range(1, 11))
DISPLAY_KS = (1, 3, 5, 10)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_values(truth, predicted):
    if not truth:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0))}


def fixed_at_k(hits, n):
    return {str(k): {"correct": int(hits[k]), "n": n, "rate": hits[k] / n if n else None} for k in KS}


def validate_pair(base, augmented):
    """Check the shared pipeline, model selections and data were actually frozen."""
    a, b = [json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8")) for d in (base, augmented)]
    if a.get("encoder_augmentation") or not b.get("encoder_augmentation"):
        raise ValueError("Expected A without encoder and B with encoder feature augmentation")
    hashes = {}
    for name in COMMON_FILES:
        left, right = Path(base) / name, Path(augmented) / name
        if not left.is_file() or not right.is_file() or sha256(left) != sha256(right):
            raise ValueError(f"Common comparison artifact differs: {name}")
        hashes[name] = sha256(left)
    for slot in SLOTS:
        for key in ("model_status", "classes", "model", "weights", "unsafe_labels"):
            if a["slots"][slot].get(key) != b["slots"][slot].get(key):
                raise ValueError(f"Frozen tree selection differs: {slot}/{key}")
    if a.get("refinement_policy", {}) != b.get("refinement_policy", {}):
        raise ValueError("Rule/router/attribute policies differ")
    if b.get("inference_rows_used_for_training_or_selection") != 0:
        raise ValueError("Augmented training must explicitly exclude final evaluation rows")
    return {"common_artifact_sha256": hashes, "baseline_manifest_sha256": sha256(Path(base) / "manifest.json"),
            "encoder_manifest_sha256": sha256(Path(augmented) / "manifest.json")}


def raw_values(prepared, config, slot):
    if config.get("model_status") != "trained":
        return []
    if slot == "part_number":
        ranks = getattr(prepared, "_generation_ranking", [])
        return [prepared.candidates[i]["value"] for i in np.argsort(ranks, kind="stable")]
    probabilities = getattr(prepared, "_generation_probabilities", {}).get(slot)
    if probabilities is None:
        return []
    return [config["classes"][i] for i in np.argsort(-probabilities, kind="stable")]


def precompute(predictor, prepared, batch_size=64, rank_batch_size=16):
    predictor.state.ensure_embeddings(prepared, batch_size=batch_size)
    for slot in SLOTS:
        config = predictor.state.manifest["slots"][slot]
        if config.get("model_status") != "trained":
            continue
        model = predictor.state.model(slot)
        size = rank_batch_size if slot == "part_number" else batch_size
        for start in range(0, len(prepared), size):
            batch = prepared[start:start + size]
            if slot == "part_number":
                for row, ranking in zip(batch, model.rankings(batch)):
                    row._generation_ranking = ranking["ensemble"]
            else:
                values = model.probabilities(batch, config["classes"])[config["model"]]
                for row, probability in zip(batch, values):
                    if not hasattr(row, "_generation_probabilities"):
                        row._generation_probabilities = {}
                    row._generation_probabilities[slot] = probability
        print("COMPARISON_PRECOMPUTED", slot, flush=True)


def evaluate_variant(directory, rows, out, label, batch_size=64, rank_batch_size=16):
    started = time.time()
    registry = Registry.load(Path(directory) / "evaluation_registry.json")
    predictor = SpectrumPredictor(directory, registry=registry, candidate_limit=10)
    book = predictor.state.book
    prepared = [preprocess(row["품목명"], row["규격명"]) for row in rows]
    if {row.key for row in prepared} & {tuple(entry["normalized_input"]) for entry in registry.entries}:
        raise ValueError("Evaluation input keys leaked into the retained-training registry")
    for row in prepared:
        row.candidates = candidates(row.normalized)
        row._identifier_pool_ready = True
    precompute(predictor, prepared, batch_size, rank_batch_size)
    truth_parts, predicted_parts, raw_parts = ({slot: [] for slot in SLOTS} for _ in range(3))
    joint = {slot: Counter() for slot in SLOTS}
    marginal = {slot: Counter() for slot in SLOTS}
    raw_hits = {slot: Counter() for slot in SLOTS}
    errors, source_counts = Counter(), Counter()
    output, part_rows, candidate_rows, curve_records = [], [], [], []
    manual_count = auto_count = 0
    for index, (original, prepared_row) in enumerate(zip(rows, prepared)):
        # Pass only input fields into the complete generation pipeline.
        events = {event["slot"]: event for event in predictor.events(
            original["품목명"], original["규격명"], prepared=prepared_row)}
        assembly = events["assembly"]
        full_codes = effective_codes(assembly)[:10]
        if len(full_codes) != len(set(full_codes)):
            raise ValueError("Full-code candidates must be unique")
        decoded = [describe_code(book, code) for code in full_codes]
        # Targets enter only after the ranked candidate list is final.
        truth = original["품목코드"].strip().upper()
        parsed = book.parse(truth)
        manual_count += bool(parsed)
        curve_records.append((truth, full_codes))
        auto_count += bool(assembly.get("auto_confirmed"))
        source_counts[assembly.get("code_source") or "no_code"] += 1
        display = {slot: displayed_slot(assembly, events[slot], slot) for slot in SLOTS}
        targets = {slot: actual_prefix(truth) if slot == "classification" else
                   parsed["values"].get(slot) if parsed else None for slot in SLOTS}
        for slot in SLOTS:
            target = targets[slot]
            eligible = bool(target)
            raw = raw_values(prepared_row, predictor.state.manifest["slots"][slot], slot)
            alternatives = [candidate["value"] for candidate in assembly.get("part_spectra", {}).get(slot, [])]
            errors[slot] += events[slot].get("status") == "error"
            if eligible:
                truth_parts[slot].append(target)
                predicted_parts[slot].append(display[slot])
                raw_parts[slot].append(raw[0] if raw else "")
                for k in KS:
                    joint[slot][k] += target in [candidate["values"].get(slot) for candidate in decoded[:k]]
                    marginal[slot][k] += target in alternatives[:k]
                    raw_hits[slot][k] += target in raw[:k]
            part_rows.append({"source_row": index + 2, "품목명": original["품목명"], "규격명": original["규격명"],
                              "slot": slot, "truth": target or "", "eligible": eligible,
                              "placeholder": bool(target and not target.strip("-")), "displayed_value": display[slot],
                              "raw_model_value": raw[0] if raw else "", "correct": eligible and target == display[slot],
                              "status": events[slot].get("status"),
                              **{f"joint_hit_at_{k}": bool(eligible and target in [d["values"].get(slot) for d in decoded[:k]]) for k in DISPLAY_KS},
                              "raw_model_candidates_top10": json.dumps(raw[:10], ensure_ascii=False)})
        for rank, (code, detail) in enumerate(zip(full_codes, decoded), 1):
            candidate_rows.append({"source_row": index + 2, "rank": rank, "code": code, "truth": truth,
                                   "correct": code == truth, "semantic_layout_verified": detail["semantic_layout_verified"],
                                   **{slot: detail["values"].get(slot) for slot in SLOTS}})
        output.append({"source_row": index + 2, "품목명": original["품목명"], "규격명": original["규격명"],
                       "truth": truth, "manual_parseable": bool(parsed), "displayed_code": assembly.get("display_code") or "",
                       "effective_top1_code": full_codes[0] if full_codes else "", "candidate_count": len(full_codes),
                       "candidate_codes": full_codes, "displayed_values": display,
                       "auto_confirmed": bool(assembly.get("auto_confirmed")),
                       **{f"hit_at_{k}": truth in full_codes[:k] for k in KS}})
        if (index + 1) % 250 == 0:
            print("COMPARISON_EVALUATED", label, index + 1, "of", len(rows), flush=True)
    audits = audit_slots(book, registry, truth_parts, predicted_parts, len(rows))
    groups = coverage_curve(curve_records, book, max_k=10)
    corpus_codes = {entry["code"].upper() for entry in registry.entries}
    metrics = {
        "model_name": label, "artifact_directory": str(directory), "evaluation_rows": len(rows),
        "manual_parseable_rows": manual_count, "unmapped_rows": len(rows) - manual_count,
        "candidate_k_metrics": {"max_k": 10, "groups": groups},
        "slots": {slot: {**score_values(truth_parts[slot], predicted_parts[slot]),
                         "joint_at_k": fixed_at_k(joint[slot], len(truth_parts[slot])),
                         "marginal_at_k": fixed_at_k(marginal[slot], len(truth_parts[slot])),
                         "audit": audits[slot]} for slot in SLOTS},
        "raw_heads": {slot: {**score_values(truth_parts[slot], raw_parts[slot]),
                             "at_k": fixed_at_k(raw_hits[slot], len(truth_parts[slot]))} for slot in SLOTS},
        "auto_confirmed_rows": auto_count, "slot_errors": dict(errors), "code_sources": dict(source_counts),
        "reference_scope": registry.provenance, "operational_master_used": False,
        "normalized_input_overlap_in_registry": 0,
        "evaluation_code_overlap_in_registry": sum(row["truth"] in corpus_codes for row in output),
        "elapsed_seconds": time.time() - started,
    }
    variant_out = Path(out) / label
    variant_out.mkdir(parents=True, exist_ok=True)
    write_json(variant_out / "evaluation_metrics.json", metrics)
    write_json(variant_out / "candidate_k_metrics.json", metrics["candidate_k_metrics"])
    write_csv(variant_out / "prediction_parts.csv", [
        {**{key: value for key, value in row.items() if key not in {"candidate_codes", "displayed_values"}},
         "candidate_codes": json.dumps(row["candidate_codes"], ensure_ascii=False), **row["displayed_values"]} for row in output])
    write_csv(variant_out / "six_slot_predictions.csv", part_rows)
    for slot in SLOTS:
        write_csv(variant_out / (slot + "_predictions.csv"), [row for row in part_rows if row["slot"] == slot])
    write_csv(variant_out / "full_code_candidates_top10.csv", candidate_rows)
    # Release large forests before constructing the second variant.
    predictor.state.models.clear()
    predictor.state.bridge.close()
    predictor.state._encoder_bundle = None
    del predictor, prepared, part_rows, candidate_rows
    from .accuracy_runtime import refinement
    from .exploratory import training_statistics
    from .generation import load_state
    from .spectrum import operational_lookup
    for cached in (refinement, training_statistics, load_state, operational_lookup):
        cached.cache_clear()
    gc.collect()
    return metrics, output


def paired_deltas(left, right):
    if [row["source_row"] for row in left] != [row["source_row"] for row in right]:
        raise ValueError("Paired evaluation rows differ")
    result = {}
    for group in ("all", "manual", "unmapped"):
        pairs = [(a, b) for a, b in zip(left, right) if group == "all" or a["manual_parseable"] == (group == "manual")]
        points = []
        for k in KS:
            a_only = sum(a[f"hit_at_{k}"] and not b[f"hit_at_{k}"] for a, b in pairs)
            b_only = sum(b[f"hit_at_{k}"] and not a[f"hit_at_{k}"] for a, b in pairs)
            points.append({"k": k, "n": len(pairs), "no_encoder_only_correct": a_only,
                           "encoder_only_correct": b_only,
                           "both_correct": sum(a[f"hit_at_{k}"] and b[f"hit_at_{k}"] for a, b in pairs),
                           "delta_percentage_points": 100 * (b_only - a_only) / len(pairs) if pairs else None})
        result[group] = points
    return result


def write_curve_svg(path, variants):
    """Standalone export without installing a plotting dependency on the server."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="370" viewBox="0 0 1080 370">',
           '<rect width="1080" height="370" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#334155}</style>',
           '<text x="35" y="25" style="font-size:17px;font-weight:bold">Full-code Hit@K | fixed denominators, unique candidates</text>']
    colors = {"no_encoder": "#2563eb", "encoder": "#ea580c"}
    for panel, group in enumerate(("all", "manual", "unmapped")):
        x, y, width, height = 45 + 355 * panel, 70, 285, 235
        n = variants["no_encoder"]["candidate_k_metrics"]["groups"][group]["n"]
        svg.append(f'<text x="{x}" y="53" style="font-weight:bold">{escape(group)} (n={n})</text>')
        for tick in range(0, 101, 25):
            yy = y + height * (1 - tick / 100)
            svg.append(f'<path d="M{x},{yy}H{x+width}" stroke="#e2e8f0"/>')
            svg.append(f'<text x="{x-8}" y="{yy+4}" text-anchor="end">{tick}%</text>')
        for k in DISPLAY_KS:
            xx = x + (k - 1) * width / 9
            svg.append(f'<text x="{xx}" y="{y+height+20}" text-anchor="middle">{k}</text>')
        svg.append(f'<text x="{x+width/2}" y="{y+height+39}" text-anchor="middle">K</text>')
        for variant, color in colors.items():
            points = variants[variant]["candidate_k_metrics"]["groups"][group]["points"]
            coords = [(x + (point["k"] - 1) * width / 9, y + height * (1 - (point["rate"] or 0))) for point in points]
            svg.append(f'<polyline points="{" ".join(f"{xx:.1f},{yy:.1f}" for xx,yy in coords)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for xx, yy in coords:
                svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.5" fill="{color}"/>')
    svg.extend(['<text x="410" y="365" fill="#2563eb">Blue: no encoder</text>',
                '<text x="600" y="365" fill="#ea580c">Orange: trained Char-CNN + same trees</text>', '</svg>'])
    Path(path).write_text("\n".join(svg), encoding="utf-8")


def compare(no_encoder="artifacts/generation_v6", encoder="artifacts/generation_v7",
            evaluation="dataset/ERP_원재료_inference.csv", out="artifacts/encoder_comparison",
            batch_size=64, rank_batch_size=16, expected_total=2850, expected_manual=473):
    started = time.time()
    contract = validate_pair(no_encoder, encoder)
    rows = read_csv(evaluation)
    if len(rows) != expected_total:
        raise ValueError(f"Fixed evaluation denominator changed: {len(rows)} != {expected_total}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    variants, predictions = {}, {}
    for key, directory in (("no_encoder", no_encoder), ("encoder", encoder)):
        metrics, records = evaluate_variant(directory, rows, out, key, batch_size, rank_batch_size)
        if metrics["manual_parseable_rows"] != expected_manual:
            raise ValueError(f"Fixed manual denominator changed: {metrics['manual_parseable_rows']} != {expected_manual}")
        variants[key], predictions[key] = metrics, records
    deltas = paired_deltas(predictions["no_encoder"], predictions["encoder"])
    result = {
        "version": "char-cnn-feature-ablation-1", "selected_k": list(DISPLAY_KS), "max_k": 10,
        "variants": variants, "paired_deltas": deltas,
        "protocol": {**contract, "evaluation_csv_sha256": sha256(evaluation),
                     "fixed_denominators": {"all": expected_total, "manual": expected_manual,
                                            "unmapped": expected_total - expected_manual},
                     "candidate_definition": "First K unique complete codes from effective_codes; candidate_limit=10 for both variants",
                     "effective_candidate_policy": "Unverified expanded spectrum only for unsupported predicted layouts; otherwise validated spectrum",
                     "feature_difference": "B appends a trained input-only Char-CNN embedding; identical TF-IDF, structured features, tree types/parameters/blends, labels, retrieval, rules, attributes and router",
                     "tree_selection": "Frozen A training-only grouped OOF; no B OOF claims and no transferred confidence calibration",
                     "training_or_selection_uses_evaluation": False,
                     "displayed_slot_definition": "First displayed strict spectrum hypothesis; unknown decoded fields remain unknown",
                     "joint_slot_at_k_definition": "Truth slot value occurs among the first K effective complete-code candidates",
                     "marginal_at_k_definition": "Independent slot alternatives; not joint full-code coverage",
                     "unmapped_details": "Detailed labels unavailable; these rows remain in full-code all/unmapped denominators",
                     "evaluation_status": "Previously inspected final diagnostic set; not an untouched blind test",
                     "candidate_scores_are_calibrated_probabilities": False},
        "elapsed_seconds": time.time() - started,
    }
    curve_rows = [{"variant": name, "scope": group, **point}
                  for name, metric in variants.items() for group, values in metric["candidate_k_metrics"]["groups"].items()
                  for point in values["points"]]
    write_csv(out / "candidate_k_curve.csv", curve_rows)
    write_csv(out / "top_k_comparison.csv", [row for row in curve_rows if row["k"] in DISPLAY_KS])
    part_metrics = []
    for name, metric in variants.items():
        for slot, values in metric["slots"].items():
            part_metrics.append({"variant": name, "slot": slot, "n": values["n"], "accuracy": values["accuracy"],
                                 "macro_f1": values["macro_f1"], **values["audit"],
                                 **{f"joint_hit_at_{k}": values["joint_at_k"][str(k)]["rate"] for k in DISPLAY_KS}})
    write_csv(out / "six_slot_comparison.csv", part_metrics)
    paired_rows = []
    for left, right in zip(predictions["no_encoder"], predictions["encoder"]):
        paired_rows.append({"source_row": left["source_row"], "품목명": left["품목명"], "규격명": left["규격명"],
                            "truth": left["truth"], "manual_parseable": left["manual_parseable"],
                            **{f"{name}_code": row["effective_top1_code"] for name, row in (("no_encoder", left), ("encoder", right))},
                            **{f"{name}_{slot}": row["displayed_values"][slot] for name, row in (("no_encoder", left), ("encoder", right)) for slot in SLOTS},
                            **{f"{name}_hit_at_{k}": row[f"hit_at_{k}"] for name, row in (("no_encoder", left), ("encoder", right)) for k in DISPLAY_KS}})
    write_csv(out / "paired_predictions.csv", paired_rows)
    write_curve_svg(out / "candidate_k_curve.svg", variants)
    write_json(out / "comparison.json", result)
    print("COMPARISON_DONE", json.dumps({"out": str(out), "selected_k": list(DISPLAY_KS),
          "groups": {name: metric["candidate_k_metrics"]["groups"] for name, metric in variants.items()},
          "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-encoder", default="artifacts/generation_v6")
    parser.add_argument("--encoder", default="artifacts/generation_v7")
    parser.add_argument("--evaluation", default="dataset/ERP_원재료_inference.csv")
    parser.add_argument("--out", default="artifacts/encoder_comparison")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rank-batch-size", type=int, default=16)
    parser.add_argument("--expected-total", type=int, default=2850)
    parser.add_argument("--expected-manual", type=int, default=473)
    args = parser.parse_args()
    if min(args.batch_size, args.rank_batch_size) < 1:
        parser.error("Batch sizes must be positive")
    compare(**vars(args))


if __name__ == "__main__":
    main()
