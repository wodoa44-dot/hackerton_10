"""Reconstructed legacy ranking helpers required by train_verified.

The original material-family training program was not captured in full.
Current training entrypoints are train_generation and train_encoder.
"""
import numpy as np

def top_predictions(model, rows):
    rankings=model.rankings(rows)
    return [row.candidates[int(np.argmin(result["ensemble"]))]["value"]
            if row.candidates and result["ensemble"] else ""
            for row,result in zip(rows,rankings)]

def rank_metrics(rows, truth, predictions):
    n=len(truth)
    correct=sum(a==b for a,b in zip(truth,predictions))
    covered=sum(target in {c["value"] for c in row.candidates} for row,target in zip(rows,truth))
    return {"n":n,"top1_correct":correct,"top1_accuracy":correct/n if n else None,
            "candidate_covered":covered,"candidate_coverage":covered/n if n else None}
