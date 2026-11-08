"""Read audited training rows only; validation is an existing grouped training fold."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from erp_model.encoders.candidates import candidates
from erp_model.shared.schema import SLOTS
from erp_model.encoders.preprocess import normalize

CLASSIFIER_SLOTS=tuple(slot for slot in SLOTS if slot!='part_number')


def read_records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_training_data(base, train_path):
    base = Path(base)
    report = json.loads((base / "training_audit/training_audit.json").read_text())
    digest = hashlib.sha256(Path(train_path).read_bytes()).hexdigest()
    if digest != report["training_sha256"]:
        raise ValueError("Run the training audit again: CSV content changed")
    raw = read_records(train_path)
    audited = read_records(base / "training_audit/training_rows.csv")
    review = read_records(base / "training_audit/training_review.csv")
    masks = {}
    for row in review:
        source = int(row["source_row"])
        if row["issue_type"] == "numeric_input_code_disagreement":
            masks.setdefault(source, set()).add(row["slot"])
        if row["issue_type"] == "input_family_schema_disagreement":
            masks.setdefault(source, set()).update(["trait", "part_number", "case", "packing", "vendor"])
    records = []
    for a in audited:
        if a["retained_training"] != "True":
            continue
        source = int(a["source_row"])
        original = raw[source - 2]
        if any(original[key] != a[key] for key in ("품목명", "규격명", "품목코드")):
            raise ValueError("Training audit row no longer matches the source CSV")
        normalized = {key: normalize(original[key]) for key in ("품목명", "규격명")}
        blocked = masks.get(source, set())
        labels = {slot: a[slot + "_label"] if a[slot + "_eligible"] == "True" and slot not in blocked else None
                  for slot in CLASSIFIER_SLOTS}
        rank_truth = a["part_number_label"] if a["part_number_eligible"] == "True" and "part_number" not in blocked else None
        if not normalized["품목명"]:
            labels = {slot: None for slot in CLASSIFIER_SLOTS}
            rank_truth = None
        pool = candidates(normalized) if rank_truth else []
        records.append({"source_row": source, "key": tuple(normalized.values()), "fold": int(a["fold"]),
                        "text": "query: 품목명: " + normalized["품목명"] + " 규격명: " + normalized["규격명"],
                        "byte_text": normalized["품목명"] + " | " + normalized["규격명"],
                        "labels": labels, "rank_truth": rank_truth, "pool": pool,
                        "rank_usable": bool(rank_truth and len(pool) > 1 and any(c["value"] == rank_truth for c in pool))})
    groups = {}
    for row in records:
        groups.setdefault(row["key"], set()).add(row["fold"])
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Normalized input group crosses validation folds")
    classes = {slot: sorted({r["labels"][slot] for r in records if r["labels"][slot] is not None})
               for slot in CLASSIFIER_SLOTS}
    counts = {slot: {"rows": sum(r["labels"][slot] is not None for r in records), "classes": len(classes[slot]),
                     "placeholder_rows": sum(bool(r["labels"][slot]) and not r["labels"][slot].strip("-") for r in records)}
              for slot in CLASSIFIER_SLOTS}
    counts["part_number"] = {"label_rows": sum(bool(r["rank_truth"]) for r in records),
                             "actual_rank_training_groups": sum(r["rank_usable"] for r in records)}
    metadata = {"training_csv_sha256": digest, "training_only": True, "inference_rows_used": 0,
                "retained_rows": len(records), "validation_fold": 0, "normalized_input_group_overlap": 0,
                "numeric_and_family_review_policy": "Mask affected task labels, preserve original source records",
                "label_counts": counts, "fold_counts": dict(Counter(r["fold"] for r in records)),
                "source_rows": [r["source_row"] for r in records]}
    return records, classes, metadata
