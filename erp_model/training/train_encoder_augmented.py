"""Refit the frozen baseline tree heads with a shared trained Char-CNN vector.

No OOF score is claimed here: the final encoder has seen all retained training
rows. Model choice, blend weights, tree parameters and the shared generation
pipeline come exclusively from the audited baseline's training-only validation.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from erp_model.training.data.encoder_data import load_training_data
from erp_model.models.tabular import ClassifierBundle, RankBundle
from erp_model.encoders.preprocess import preprocess
from erp_model.shared.schema import SLOTS
from erp_model.training.train_generation import cpu_inference
from erp_model.training.io import read_csv, write_json


COMMON_FILES = (
    "preprocessor.joblib", "reference.joblib", "evaluation_registry.json",
    "operational_master.json", "company_rules.json", "manual_source.json",
    "scheme_values.json", "attributes.joblib", "attribute_validation.json",
)
OPTIONAL_COMMON_FILES = (
    "fold_assignments.csv", "training_counts.json", "data_audit.json",
    "attribute_label_quality.json", "manual_comparison.json",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimator_copy(estimator):
    """Keep all fitted-baseline hyperparameters; only execution device may vary."""
    result = clone(estimator)
    if estimator.__class__.__module__.startswith("xgboost") and os.environ.get("ERP_XGB_DEVICE"):
        result.set_params(device=os.environ["ERP_XGB_DEVICE"])
    return result


def fit_frozen_bundle(slot, template, rows, truth, shared_text):
    """Fit an existing bundle interface using the baseline's exact model kinds."""
    structured = template.feature.structured
    bundle = RankBundle(structured) if slot == "part_number" else ClassifierBundle(slot, structured)
    bundle.feature.fit(rows, shared_text)
    if slot != "part_number":
        bundle.classes = sorted(set(truth))
        if bundle.classes != list(template.classes):
            raise ValueError(f"Audited training class set changed for {slot}")
        if len(bundle.classes) < 2:
            bundle.issue = "single training class; model fitting impossible"
            return bundle
        matrix = bundle.feature.transform(rows)
        label_index = {value: i for i, value in enumerate(bundle.classes)}
        targets = np.array([label_index[value] for value in truth])
        for kind, estimator in template.models.items():
            bundle.models[kind] = estimator_copy(estimator).fit(matrix, targets)
    else:
        eligible = [(row, value) for row, value in zip(rows, truth)
                    if len(row.candidates) > 1 and value in {c["value"] for c in row.candidates}]
        bundle.train_groups = len(eligible)
        if not eligible:
            bundle.issue = "no mixed positive/negative candidate groups"
            return bundle
        matrix, counts = bundle.feature.candidate_matrix([row for row, _ in eligible])
        targets = np.array([int(candidate["value"] == value)
                            for row, value in eligible for candidate in row.candidates])
        qid = np.repeat(np.arange(len(counts)), counts)
        for kind, estimator in template.models.items():
            fitted = estimator_copy(estimator)
            options = {"qid": qid} if kind == "xgb" else {"group": counts}
            bundle.models[kind] = fitted.fit(matrix, targets, **options)
    return bundle


def validate_common_training(base, records, provenance, train_path):
    """Fail instead of silently comparing a differently filtered cohort."""
    base = Path(base)
    audit_path = base / "data_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("retained_training") != len(records):
            raise ValueError("Baseline and encoder retained cohorts have different sizes")
        hashes = audit.get("source_hashes", {})
        if str(train_path) in hashes and hashes[str(train_path)] != provenance["training_csv_sha256"]:
            raise ValueError("Baseline used a different training CSV")
    fold_path = base / "fold_assignments.csv"
    if not fold_path.exists():
        raise ValueError("Baseline fold assignments are required for a fair comparison")
    with fold_path.open(encoding="utf-8-sig", newline="") as stream:
        folds = {int(row["source_row"]): int(row["fold"]) for row in csv.DictReader(stream)}
    if folds != {row["source_row"]: row["fold"] for row in records}:
        raise ValueError("Baseline and encoder normalized-input folds differ")


def train(base="artifacts/generation_v6", encoder="artifacts/encoder_v7",
          audited_data="artifacts/encoder_training_data", out="artifacts/generation_v7",
          train_path="dataset/ERP_원재료_training.csv", device="cpu", batch_size=64):
    started = time.time()
    base, encoder, out = Path(base), Path(encoder), Path(out)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("encoder_augmentation"):
        raise ValueError("The reference baseline must have no trained encoder features")
    for name in COMMON_FILES:
        if not (base / name).is_file():
            raise FileNotFoundError(base / name)
    if out.resolve() == base.resolve() or out.resolve() == encoder.resolve():
        raise ValueError("Use a separate output directory")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing artifact versions")
    records, _, provenance = load_training_data(audited_data, train_path)
    validate_common_training(base, records, provenance, train_path)
    from erp_model.encoders.cnn.runtime import EncoderBundle
    encoded = EncoderBundle(encoder, device=device)
    if encoded.config.get("model_name") != "char_cnn":
        raise ValueError("This comparison requires the compact character CNN encoder")
    if not encoded.config.get("training_only") or encoded.config.get("inference_rows_used") != 0:
        raise ValueError("Encoder must explicitly declare training-only provenance")
    encoder_digest = encoded.config.get("training_csv_sha256")
    if encoder_digest and encoder_digest != provenance["training_csv_sha256"]:
        raise ValueError("Encoder and tree heads used different training CSVs")
    raw = read_csv(train_path)
    prepared = [preprocess(raw[row["source_row"] - 2]["품목명"], raw[row["source_row"] - 2]["규격명"])
                for row in records]
    for row, record in zip(prepared, records):
        row.candidates = record["pool"]
        row._identifier_pool_ready = True
    dimensions = int(encoded.config["embedding_dimension"])
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        vectors = np.asarray(encoded.encode_vectors(batch, batch_size=batch_size), dtype=np.float32)
        if vectors.shape != (len(batch), dimensions) or not np.isfinite(vectors).all():
            raise ValueError("Invalid encoder output")
        for row, vector in zip(batch, vectors):
            row._shared_char_embedding = vector
    del encoded
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training_status.json", {"status": "running", "training_only": True})
    common_hashes = {}
    for name in (*COMMON_FILES, *OPTIONAL_COMMON_FILES):
        if (base / name).is_file():
            shutil.copy2(base / name, out / name)
            common_hashes[name] = sha256(out / name)
    shared = joblib.load(base / "preprocessor.joblib")["text"]
    baseline_counts = json.loads((base / "training_counts.json").read_text(encoding="utf-8"))
    counts, contracts = {}, {}
    for slot in SLOTS:
        config = manifest["slots"][slot]
        indices = [i for i, record in enumerate(records)
                   if (record["rank_truth"] if slot == "part_number" else record["labels"][slot]) is not None]
        values = [(records[i]["rank_truth"] if slot == "part_number" else records[i]["labels"][slot]) for i in indices]
        counts[slot] = {"rows": len(indices), "classes": len(set(values)), "labels": dict(Counter(values))}
        for key in ("rows", "classes", "labels"):
            if counts[slot][key] != baseline_counts[slot][key]:
                raise ValueError(f"Baseline and augmented labels differ: {slot}/{key}")
        if config.get("model_status") != "trained":
            if indices:
                raise ValueError(f"Baseline {slot} is untrained despite audited labels")
            continue
        template = joblib.load(base / (slot + ".joblib"))
        if getattr(template.feature, "encoder_dimensions", 0):
            raise ValueError("Baseline tree bundle already includes encoder vectors")
        print("AUGMENTED_TRAIN", slot, "rows", len(indices), "classes", len(set(values)), flush=True)
        bundle = fit_frozen_bundle(slot, template, [prepared[i] for i in indices], values, shared)
        gpu_used = cpu_inference(bundle)
        joblib.dump(bundle, out / (slot + ".joblib"), compress=3)
        contracts[slot] = {"selection": config.get("model"), "weights": config.get("weights"),
                           "model_kinds": list(template.models), "gpu_training_verified": gpu_used,
                           "embedding_dimension": dimensions,
                           "baseline_parameters": {key: value.get_params(deep=False) for key, value in template.models.items()}}
        if slot == "part_number":
            contracts[slot]["rank_training_groups"] = bundle.train_groups
            if bundle.train_groups != template.train_groups:
                raise ValueError("Baseline and augmented part-number rank cohorts differ")
        print("AUGMENTED_SAVED", slot, config.get("model"), flush=True)
        del bundle, template
        gc.collect()
    encoder_out = out / "encoder"
    encoder_out.mkdir()
    for name in ("weights.pt", "encoder_config.json", "grouped_validation.json", "training_data_audit.json"):
        if (encoder / name).is_file():
            shutil.copy2(encoder / name, encoder_out / name)
    if (encoder / "tokenizer").is_dir():
        shutil.copytree(encoder / "tokenizer", encoder_out / "tokenizer")
    augmented = copy.deepcopy(manifest)
    for slot, config in augmented["slots"].items():
        # Baseline OOF confidence cuts are not calibrated for changed features.
        config.pop("thresholds", None)
        config["confirmation_policy"] = "registry_exact_only; augmented confidence uncalibrated"
        config["rank_confirmation_enabled"] = False
        config["selection_metric"] = "Frozen baseline training-only OOF selection; no augmented OOF score"
        if slot in contracts:
            config["gpu_training_verified"] = contracts[slot]["gpu_training_verified"]
    augmented.update({
        "version": "7.0-char-cnn-augmented", "baseline_version": manifest.get("version"),
        "baseline_manifest_sha256": sha256(base / "manifest.json"),
        "encoder_augmentation": {"directory": "encoder", "embedding_dimension": dimensions,
                                 "runtime_device": "cpu", "input_columns": ["품목명", "규격명"],
                                 "features": "append shared input embedding to unchanged TF-IDF and structured features",
                                 "weights_sha256": sha256(encoder_out / "weights.pt")},
        "validation": "Final refit only. Baseline grouped OOF model choices are frozen. No augmented OOF estimates.",
        "inference_rows_used_for_training_or_selection": 0,
        "common_artifact_sha256": common_hashes,
        "elapsed_seconds": time.time() - started,
    })
    write_json(out / "augmented_training_counts.json", counts)
    write_json(out / "comparison_training_protocol.json", {
        "feature_ablation": "A=TF-IDF+structured; B=A+trained shared Char-CNN embedding",
        "tree_architecture_and_blends_frozen": True, "training_cohort_and_label_masks_identical": True,
        "training_provenance": provenance, "slot_contracts": contracts,
        "common_artifacts_copied_byte_identically": common_hashes,
        "augmented_oof_metrics_available": False, "augmented_confidence_calibrated": False,
        "selection_uses_final_evaluation": False,
    })
    write_json(out / "manifest.json", augmented)
    write_json(out / "training_status.json", {"status": "complete", "elapsed_seconds": augmented["elapsed_seconds"]})
    print("AUGMENTED_DONE", out, "seconds", augmented["elapsed_seconds"], flush=True)
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="artifacts/generation_v6")
    parser.add_argument("--encoder", default="artifacts/encoder_v7")
    parser.add_argument("--audited-data", default="artifacts/encoder_training_data")
    parser.add_argument("--out", default="artifacts/generation_v7")
    parser.add_argument("--train-path", default="dataset/ERP_원재료_training.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    train(**vars(args))


if __name__ == "__main__":
    main()
