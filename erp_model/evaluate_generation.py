"""Diagnostic evaluation using training-only references; no operational-master lookup."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .company_schema import actual_prefix
from .generation import GenerationPredictor
from .preprocess import preprocess, identifier_candidates
from .registry import Registry
from .schema import SLOTS
from .train import metrics, read_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="artifacts/generation_v5")
    parser.add_argument("--evaluation", default="dataset/ERP_원재료_inference.csv")
    args = parser.parse_args()
    directory = Path(args.models)
    registry = Registry.load(directory / "evaluation_registry.json")
    predictor = GenerationPredictor(directory, registry=registry)
    rows = read_csv(args.evaluation)
    prepared = [preprocess(row["품목명"], row["규격명"]) for row in rows]
    for p in prepared:
        p.candidates = identifier_candidates(p.normalized)
        p._identifier_pool_ready = True
    for slot in SLOTS:
        cfg = predictor.state.manifest["slots"][slot]
        if cfg.get("model_status") != "trained":
            continue
        model = predictor.state.model(slot)
        if slot == "part_number":
            for start in range(0, len(prepared), 64):
                batch = prepared[start:start + 64]
                for p, values in zip(batch, model.rankings(batch)):
                    p._generation_ranking = values["ensemble"]
        else:
            probabilities = model.probabilities(prepared, cfg["classes"])[cfg["model"]]
            for p, values in zip(prepared, probabilities):
                if not hasattr(p, "_generation_probabilities"):
                    p._generation_probabilities = {}
                p._generation_probabilities[slot] = values
        print("BATCH SCORED", slot, flush=True)
    truths, predictions = {s: [] for s in SLOTS}, {s: [] for s in SLOTS}
    code_available = code_correct = supported = generated = six_output_rows = manual_code_correct = 0
    sources, errors = Counter(), Counter()
    by_prefix, csv_rows = {}, []
    with (directory / "evaluation_predictions.jsonl").open("w", encoding="utf-8") as output:
        for index, (row, p) in enumerate(zip(rows, prepared)):
            events = list(predictor.events(row["품목명"], row["규격명"], prepared=p))
            by = {event["slot"]: event for event in events}
            six_output_rows += all(s in by for s in SLOTS)
            for slot in SLOTS:
                errors[slot] += by[slot]["status"] == "error"
            expected = row["품목코드"].strip().upper()
            truth_prefix = actual_prefix(expected)
            truths["classification"].append(truth_prefix)
            predictions["classification"].append(by["classification"].get("predicted_value") or "")
            parsed = predictor.state.book.parse(expected)
            if parsed:
                supported += 1
                for slot in SLOTS[1:]:
                    value = parsed["values"][slot]
                    if value:
                        truths[slot].append(value)
                        predictions[slot].append(by[slot].get("predicted_value") or "")
            code = by["assembly"]["display_code"]
            available, correct = bool(code), bool(code and code.upper() == expected)
            code_available += available
            code_correct += correct
            manual_code_correct += bool(parsed and correct)
            source = by["assembly"]["code_source"]
            sources[source or "no_code"] += 1
            generated += source == "model_assembled"
            key = by_prefix.setdefault(truth_prefix, {"n": 0, "code_available": 0, "code_correct": 0})
            key["n"] += 1
            key["code_available"] += available
            key["code_correct"] += correct
            output.write(json.dumps({"source_row": index + 2, "truth": expected, "events": events}, ensure_ascii=False) + "\n")
            csv_rows.append({"품목명": row["품목명"], "규격명": row["규격명"], "정답코드": expected,
                **{s: by[s].get("predicted_value") for s in SLOTS},
                "예측코드": code, "코드출처": source, "코드일치": correct})
            if index % 250 == 0:
                print("EVALUATED", index, flush=True)
    result = {"version": predictor.state.manifest["version"], "evaluation_rows": len(rows),
        "company_layout_rows": supported, "company_layout_coverage": supported / len(rows),
        "manual_full_code_correct": manual_code_correct,
        "manual_full_code_exact_rate": manual_code_correct / supported if supported else None,
        "classification_classes": len(predictor.state.manifest["slots"]["classification"]["classes"]),
        "slots": {s: metrics(truths[s], predictions[s], s) if truths[s] else {"n": 0, "accuracy": None, "macro_f1": None} for s in SLOTS},
        "six_output_rows": six_output_rows, "slot_errors": dict(errors),
        "code_available": code_available, "code_available_rate": code_available / len(rows),
        "model_generated_codes": generated, "full_code_correct": code_correct, "full_code_exact_rate": code_correct / len(rows),
        "code_sources": dict(sources), "by_actual_prefix": by_prefix,
        "evaluation_reference_scope": registry.provenance, "operational_master_used": False,
        "note": "The final set has been inspected. Results are diagnostic; model selection uses training grouped OOF only."}
    write_json(directory / "evaluation_metrics.json", result)
    with (directory / "evaluation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in {"slots", "by_actual_prefix"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
