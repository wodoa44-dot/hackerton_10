"""Train six code-field predictors with actual ERP prefixes and company layouts."""
import csv
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize as normalize_matrix

from .company_schema import CompanySchemas, actual_prefix
from .features import Features
from .models import ClassifierBundle, RankBundle, WEIGHTS
from .preprocess import preprocess, identifier_candidates
from .manual_audit import capacity_policy, audit
from .registry import Registry
from .schema import SLOTS
from .train import metrics, read_csv, thresholds, write_json
from .accuracy_candidates import candidates as atomic_candidates
from .encoder_data import load_training_data


def cpu_inference(bundle):
    gpu_used = False
    for estimator in bundle.models.values():
        if estimator.__class__.__module__.startswith("xgboost"):
            config = json.loads(estimator.get_booster().save_config())
            gpu_used = gpu_used or config["learner"]["generic_param"]["device"].startswith("cuda")
            estimator.set_params(device="cpu", n_jobs=2)
    return gpu_used


def train(train_path="dataset/ERP_원재료_training.csv", evaluation_path="dataset/ERP_원재료_inference.csv", out_dir="artifacts/generation_v5", full_path=None, audited_data=None):
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    book = CompanySchemas()
    raw = read_csv(train_path)
    evaluation = read_csv(evaluation_path)
    if full_path:
        full = read_csv(full_path)
        records = lambda rows: Counter(tuple(sorted(row.items())) for row in rows)
        if records(full) != records(raw + evaluation):
            raise ValueError("Full CSV does not match the training and evaluation records")
    prepared = [preprocess(row["품목명"], row["규격명"]) for row in raw]
    evaluation_keys = {preprocess(row["품목명"], row["규격명"]).key for row in evaluation}
    keep = [i for i, row in enumerate(prepared) if row.key not in evaluation_keys
            and re.match(r"[A-Z0-9]", raw[i]["품목코드"].upper())]
    prefix_labels = [actual_prefix(row["품목코드"]) for row in raw]
    parsed = [book.parse(row["품목코드"]) for row in raw]
    for i,item in enumerate(prepared):
        if parsed[i] and any(p['key'] in {'pn','sub','model','name'} for p in book.layout(parsed[i]['category_id'])[0]['part_number']):
            item.candidates=atomic_candidates(item.normalized)
    clean=None
    if audited_data:
        records,_,provenance=load_training_data(audited_data,train_path)
        clean={r['source_row']-2:r for r in records}
        assert set(clean)==set(keep),'Audited cohort differs'
    groups = [json.dumps(prepared[i].key, ensure_ascii=False) for i in keep]
    folds = {}
    split = StratifiedGroupKFold(3, shuffle=True, random_state=42)
    for fold, (_, indices) in enumerate(split.split(np.zeros(len(keep)), [prefix_labels[i] for i in keep], groups)):
        for index in indices:
            folds[keep[index]] = fold
    for fold in range(3):
        assert not {prepared[i].key for i in keep if folds[i] == fold} & {prepared[i].key for i in keep if folds[i] != fold}
    if clean:assert all(folds[i]==clean[i]['fold'] for i in keep),'Audited folds differ'
    shared = {fold: Features("classification", False).fit([prepared[i] for i in keep if folds[i] != fold]).text for fold in range(3)}
    final_text = Features("classification", False).fit([prepared[i] for i in keep]).text
    manifest = {
        "version": "5.0", "seed": 42, "feature_inputs": ["품목명", "규격명"],
        "classification_target": "Actual leading ERP code fragment: two leading digits, otherwise one leading character",
        "classification_is_semantic_family": False, "other_class_used": False,
        "company_rule_source": book.source, "rule_count": len(book.categories),
        "node_executable": shutil.which("node"), "slots": {},
        "manual_source": book.manual,
        "rules_sha256": hashlib.sha256(Path("config/company_rules.json").read_bytes()).hexdigest(),
        "detailed_label_policy": "Only code-table validated, reversible manual layouts; no invented segment labels for customer codes",
        "identifier_candidate_widths": [4, 5, 6, 7, 8, 10],
        "capacity_policy": capacity_policy([raw[i] for i in keep], book),
        "training_device": os.environ.get("ERP_XGB_DEVICE", "cpu"),
        "validation": "Three shared grouped folds; evaluation input keys excluded",
        "audited_learning_data":str(audited_data) if audited_data else None,
        "candidate_version":"atomic_v6",
    }
    comparisons = {}
    counts = {}
    scheme_values = {key: {slot: set() for slot in SLOTS} for key in book.categories}
    for i in keep:
        if parsed[i]:
            for slot, value in parsed[i]["values"].items():
                if value:
                    scheme_values[parsed[i]["category_id"]][slot].add(value)
    for slot in SLOTS:
        indices = keep if slot == "classification" else [
            i for i in keep if parsed[i] and parsed[i]["values"][slot]
            and (slot != "part_number" or any(part["key"] in {"pn", "sub", "model", "name"} for part in book.layout(parsed[i]["category_id"])[0][slot]))
        ]
        if clean:indices=[i for i in indices if (clean[i]['rank_truth'] if slot=='part_number' else clean[i]['labels'][slot]) is not None]
        ambiguous = defaultdict(set)
        for i in indices:
            ambiguous[prepared[i].key].add(prefix_labels[i] if slot == "classification" else parsed[i]["values"][slot])
        conflicting_keys = {key for key, values in ambiguous.items() if len(values) > 1}
        excluded_conflicts = sum(prepared[i].key in conflicting_keys for i in indices)
        indices = [i for i in indices if prepared[i].key not in conflicting_keys]
        rows = [prepared[i] for i in indices]
        truth = [prefix_labels[i] if slot == "classification" else parsed[i]["values"][slot] for i in indices]
        classes = sorted(set(truth))
        print("TRAIN", slot, "rows", len(rows), "classes", len(classes), "device", manifest["training_device"], flush=True)
        counts[slot] = {"rows": len(rows), "classes": len(classes), "labels": dict(Counter(truth)), "excluded_conflicting_input_rows": excluded_conflicts}
        if not rows:
            manifest["slots"][slot] = {"model_status": "no_labels", "classes": []}
            continue
        if slot == "part_number":
            predictions = [""] * len(rows)
            for fold in range(3):
                ti = [j for j, i in enumerate(indices) if folds[i] != fold]
                vi = [j for j, i in enumerate(indices) if folds[i] == fold]
                if not vi: continue
                model = RankBundle(True).fit([rows[j] for j in ti], [truth[j] for j in ti], shared[fold])
                ranking = model.rankings([rows[j] for j in vi])
                for j, result in zip(vi, ranking):
                    if result["ensemble"]:
                        predictions[j] = rows[j].candidates[int(np.argmin(result["ensemble"]))]["value"]
                print(" ", slot, "fold", fold, "rank groups", model.train_groups, flush=True)
            comparisons[slot] = {"oof": metrics(truth, predictions, slot),
                "candidate_coverage": sum(y in {c["value"] for c in p.candidates} for p, y in zip(rows, truth)) / len(rows)}
            model = RankBundle(True).fit(rows, truth, final_text)
            gpu_used = cpu_inference(model)
            cfg = {"model": "mean_rank_ensemble", "weights": {"xgb_rank": .5, "lgb_rank": .5},
                "model_status": "trained", "rank_confirmation_enabled": False, "gpu_training_verified": gpu_used}
        else:
            kinds = ["xgb", "lgb", "ensemble"] + (["forest", "ensemble_forest"] if slot in {"classification", "packing"} else [])
            probabilities = {kind: np.zeros((len(rows), len(classes))) for kind in kinds}
            unsafe = set()
            for fold in range(3):
                ti = [j for j, i in enumerate(indices) if folds[i] != fold]
                vi = [j for j, i in enumerate(indices) if folds[i] == fold]
                if not vi: continue
                model = ClassifierBundle(slot, True).fit([rows[j] for j in ti], [truth[j] for j in ti], shared[fold])
                output = model.probabilities([rows[j] for j in vi], classes)
                for kind in kinds:
                    probabilities[kind][vi] = output[kind]
                unsafe.update(set(classes) - set(model.classes))
                print(" ", slot, "fold", fold, "done", flush=True)
            scores = {kind: metrics(truth, np.array(classes)[values.argmax(axis=1)].tolist(), slot) for kind, values in probabilities.items()}
            selected = max(kinds, key=lambda kind: (scores[kind]["macro_f1"], scores[kind]["accuracy"], kind == "ensemble"))
            comparisons[slot] = {"models": scores, "selected": selected}
            cfg = {"classes": classes, "model": selected, "weights": WEIGHTS[selected],
                "unsafe_labels": sorted(unsafe), "thresholds": thresholds(truth, probabilities[selected], classes, slot),
                "model_status": "trained", "selection_metric": "OOF macro F1, then accuracy"}
            model = ClassifierBundle(slot, True).fit(rows, truth, final_text)
            cfg["gpu_training_verified"] = cpu_inference(model)
        joblib.dump(model, out / (slot + ".joblib"), compress=3)
        manifest["slots"][slot] = cfg
        print("SAVED", slot, cfg["model"], flush=True)
    joblib.dump({"text": final_text}, out / "preprocessor.joblib", compress=3)
    matrix = normalize_matrix(hstack([vector.transform([prepared[i].normalized[column] for i in keep]) for vector, column in zip(final_text, ["품목명", "규격명"])], format="csr"))
    reference_rows = [dict(raw[i], _source_file=train_path, _source_row=i + 2) for i in keep]
    joblib.dump({"matrix": matrix, "rows": reference_rows}, out / "reference.joblib", compress=3)
    Registry.build(reference_rows, {"scope": "retained_training_only", "rows": len(reference_rows)}).save(out / "evaluation_registry.json")
    # The deployed lookup uses the same retained training corpus as evaluation.
    Registry.build(reference_rows, {"scope": "retained_training_only", "rows": len(reference_rows)}).save(out / "operational_master.json")
    for filename in ["company_rules.json", "manual_source.json"]:
        shutil.copy2(Path("config") / filename, out / filename)
    comparison = audit(train_path, evaluation_path, out)
    manifest["elapsed_seconds"] = time.time() - started
    write_json(out / "manifest.json", manifest)
    write_json(out / "training_counts.json", counts)
    write_json(out / "scheme_values.json", {key: {slot: sorted(values) for slot, values in value.items()} for key, value in scheme_values.items()})
    write_json(out / "cv_comparison.json", comparisons)
    write_json(out / "data_audit.json", {
        "train_original": len(raw), "evaluation_rows": len(evaluation), "excluded_overlap": sum(item.key in evaluation_keys for item in prepared),
        "retained_training": len(keep), "normalized_key_overlap": 0,
        "excluded_non_code_records": sum(not re.match(r"[A-Z0-9]", row["품목코드"].upper()) for row in raw),
        "manual_comparison": {k: comparison[k] for k in ["total_rows", "status_counts", "capacity_encoding"]},
        "actual_prefix_classes": counts["classification"]["labels"],
        "company_layout_training_rows": sum(parsed[i] is not None for i in keep),
        "company_layout_evaluation_rows": sum(book.parse(row["품목코드"]) is not None for row in evaluation),
        "source_hashes": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [train_path, evaluation_path]},
        "lookup_evaluation": "Training-only reference; operational master is excluded from evaluation",
    })
    with (out / "fold_assignments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_row", "normalized_name", "normalized_spec", "fold", "classification"])
        for i in keep:
            writer.writerow([i + 2, *prepared[i].key, folds[i], prefix_labels[i]])
    print("DONE", json.dumps({"seconds": manifest["elapsed_seconds"], "slots": {key: {k:v for k,v in value.items() if k != "labels"} for key,value in counts.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--audited-data')
    args=parser.parse_args()
    train(audited_data=args.audited_data)
