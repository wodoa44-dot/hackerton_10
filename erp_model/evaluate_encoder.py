"""Evaluate a completed encoder on the existing diagnostic inference set.

This is a separate command: its targets never enter train_encoder.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from .company_schema import actual_prefix
from .encoder_runtime import EncoderSpectrumPredictor
from .evaluation_audit import displayed_slot
from .exploratory import coverage_curve, effective_codes
from .preprocess import preprocess
from .registry import Registry
from .schema import SLOTS
from .train import metrics, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/generation_v6")
    parser.add_argument("--encoder", default="artifacts/encoder_v7")
    parser.add_argument("--input", default="dataset/ERP_원재료_inference.csv")
    args = parser.parse_args()
    base, out = Path(args.base), Path(args.encoder)
    if not (out / "completion.json").exists():
        raise ValueError("Complete encoder training before evaluating inference targets")
    registry = Registry.load(base / "evaluation_registry.json")
    predictor = EncoderSpectrumPredictor(base, registry=registry, candidate_limit=10, encoder_dir=out)
    rows = read_csv(args.input)
    prepared = [preprocess(r["품목명"], r["규격명"]) for r in rows]
    reference_keys = {tuple(r["normalized_input"]) for r in registry.entries}
    if any(p.key in reference_keys for p in prepared):
        raise ValueError("Inference inputs overlap the training lookup index")
    started = time.time()
    for start in range(0, len(prepared), 64):
        predictor.encoder.attach(prepared[start:start + 64], predictor.state.manifest)
    raw_truth, raw_pred = {s: [] for s in SLOTS}, {s: [] for s in SLOTS}
    final_truth, final_pred = {s: [] for s in SLOTS}, {s: [] for s in SLOTS}
    results = []
    curve_records = []
    for i, (row, p) in enumerate(zip(rows, prepared)):
        outputs = {e["slot"]: e for e in predictor.events(row["품목명"], row["규격명"], prepared=p)}
        assembly = outputs["assembly"]
        code = row["품목코드"].strip().upper()
        parsed = predictor.state.book.parse(code)
        item = {"원본행": i + 2, "품목명": row["품목명"], "규격명": row["규격명"], "원본정답": code,
                "예측코드": assembly.get("display_code"), "후보코드": json.dumps(effective_codes(assembly), ensure_ascii=False)}
        for slot in SLOTS:
            target = actual_prefix(code) if slot == "classification" else parsed["values"][slot] if parsed else None
            raw = p._encoder_prediction[slot]
            guess = raw[0]["value"] if raw else ""
            value = displayed_slot(assembly, outputs[slot], slot)
            item[slot + "_encoder"] = guess
            item[slot + "_final"] = value
            item[slot + "_truth"] = target
            item[slot + "_eligible"] = bool(target)
            item[slot + "_top5"] = json.dumps([c["value"] for c in raw[:5]], ensure_ascii=False)
            if target:
                raw_truth[slot].append(target)
                raw_pred[slot].append(guess)
                final_truth[slot].append(target)
                final_pred[slot].append(value)
        results.append(item)
        curve_records.append((code, effective_codes(assembly)))
        if i % 250 == 0:
            print("ENCODER_EVALUATED", i, flush=True)
    curve = coverage_curve(curve_records, predictor.state.book, 10)
    report = {"training_complete": True, "inference_rows": len(rows), "inference_used_for_fit": False,
              "encoder_heads": {s: metrics(raw_truth[s], raw_pred[s], s) for s in SLOTS},
              "displayed_final_slots": {s: metrics(final_truth[s], final_pred[s], s) for s in SLOTS},
              "candidate_k_metrics": curve,
              "note": "Previously inspected diagnostic evaluation. Epoch selection used grouped training fold 0 only.",
              "seconds": time.time() - started}
    baseline_path = base / "evaluation_metrics.json"
    if baseline_path.exists():
        report["baseline_displayed_slots"] = json.loads(baseline_path.read_text())["slots"]
    (out / "inference_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "inference_parts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("ENCODER_EVALUATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
