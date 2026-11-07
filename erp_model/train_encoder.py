"""GPU fine tuning of a shared encoder and six audited ERP task heads.

No inference CSV is opened here. Fold 0 of the existing grouped training split
selects the epoch. A fresh model then fits all retained training rows.
"""
import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from transformers import AutoTokenizer

from .encoder_data import load_training_data
from .encoder_network import CLASSIFIER_SLOTS, ERPEncoder, byte_ids, feature_vector


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_metadata(records, indices, keys=None):
    pools = [c for i in indices for c in records[i]["pool"]]
    if keys is None:
        keys = sorted({key for c in pools for key, v in c["features"].items() if isinstance(v, (int, float, bool))})
    scales = [max(1., max((abs(float(c["features"].get(key, 0))) for c in pools), default=1.)) for key in keys]
    return keys, scales


def prepare_inputs(records, tokenizer, classes, max_length, device):
    tokens = tokenizer([r["text"] for r in records], max_length=max_length,
                       padding="max_length", truncation=True, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    text_bytes = torch.tensor([byte_ids(r["byte_text"], 256) for r in records], device=device)
    maps = {slot: {value: i for i, value in enumerate(classes[slot])} for slot in CLASSIFIER_SLOTS}
    labels = {slot: torch.tensor([maps[slot].get(r["labels"][slot], -1) for r in records], device=device)
              for slot in CLASSIFIER_SLOTS}
    return tokens, text_bytes, labels


def rank_batch(model, vectors, pools, keys, scales, device):
    indices = []
    values = []
    features = []
    counts = []
    for query, pool in enumerate(pools):
        counts.append(len(pool))
        for c in pool:
            indices.append(query)
            values.append(byte_ids(c["value"], 12))
            features.append(feature_vector(c, keys, scales))
    if not indices:
        return []
    scores = []
    for start in range(0, len(indices), 4096):
        end = start + 4096
        scores.append(model.rank(vectors, torch.tensor(indices[start:end], device=device),
                                 torch.tensor(values[start:end], device=device),
                                 torch.tensor(features[start:end], device=device, dtype=torch.float32)))
    scores = torch.cat(scores)
    return list(torch.split(scores, counts))


@torch.inference_mode()
def evaluate(model, records, indices, tensors, classes, keys, scales, batch_size=256):
    model.eval()
    tokens, text_bytes, labels = tensors
    device = text_bytes.device
    guesses = {slot: [] for slot in CLASSIFIER_SLOTS}
    truths = {slot: [] for slot in CLASSIFIER_SLOTS}
    rank_hits = {k: 0 for k in (1, 3, 5, 10)}
    rank_n = rank_covered = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        ids = torch.tensor(batch, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vectors = model.encode({key: value[ids] for key, value in tokens.items()}, text_bytes[ids])
            logits = model.classify(vectors)
            rank_scores = rank_batch(model, vectors, [records[i]["pool"] for i in batch], keys, scales, device)
        for slot in CLASSIFIER_SLOTS:
            ys = labels[slot][ids].cpu().tolist()
            predictions = logits[slot].argmax(-1).cpu().tolist()
            for y, prediction in zip(ys, predictions):
                if y >= 0:
                    truths[slot].append(y)
                    guesses[slot].append(prediction)
        for n, i in enumerate(batch):
            target = records[i]["rank_truth"]
            if not target:
                continue
            rank_n += 1
            pool = records[i]["pool"]
            if pool and rank_scores:
                order = torch.argsort(rank_scores[n], descending=True, stable=True).cpu().tolist()
                ranked = [pool[j]["value"] for j in order]
                rank_covered += target in ranked
                for k in rank_hits:
                    rank_hits[k] += target in ranked[:k]
    result = {slot: {"n": len(truths[slot]),
                     "accuracy": accuracy_score(truths[slot], guesses[slot]) if truths[slot] else None,
                     "macro_f1": f1_score(truths[slot], guesses[slot], average="macro", zero_division=0) if truths[slot] else None}
              for slot in CLASSIFIER_SLOTS}
    result["part_number"] = {"n": rank_n, "candidate_covered": rank_covered,
                              "hit_at_k": {str(k): {"correct": v, "rate": v / rank_n if rank_n else None} for k, v in rank_hits.items()}}
    scores = [v["macro_f1"] for slot, v in result.items() if slot != "part_number" and v["macro_f1"] is not None]
    if rank_n:
        scores.append(rank_hits[1] / rank_n)
    result["selection_score"] = float(np.mean(scores))
    return result


def fit_epochs(model, records, indices, tensors, classes, keys, scales, epochs, batch_size,
               validation=None, output=None, early_stopping=False):
    tokens, text_bytes, labels = tensors
    device = text_bytes.device
    indices = [i for i in indices if any(records[i]["labels"][slot] is not None for slot in CLASSIFIER_SLOTS)
               or records[i]["rank_usable"]]
    weights = np.array([1 + 3 * any(records[i]["labels"][s] is not None for s in CLASSIFIER_SLOTS[1:])
                        + 5 * records[i]["rank_usable"] for i in indices], dtype=float)
    weights /= weights.sum()
    ce_weights = {}
    for slot in CLASSIFIER_SLOTS:
        counts = Counter(records[i]["labels"][slot] for i in indices if records[i]["labels"][slot] is not None)
        values = [min(8., math.sqrt(max(1, sum(counts.values())) / (len(classes[slot]) * max(1, counts.get(c, 0)))))
                  for c in classes[slot]]
        ce_weights[slot] = torch.tensor(values, device=device, dtype=torch.float32)
    encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": 2e-5}, {"params": head_params, "lr": 1e-3}], weight_decay=.01)
    steps = math.ceil(len(indices) / batch_size)
    total_steps = max(1, steps * epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, total_steps * .08),
                                                                                 max(0., 1 - step / total_steps)))
    rng = np.random.default_rng(42)
    history = []
    best = -1.
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        started = time.time()
        for step in range(steps):
            batch = rng.choice(indices, size=batch_size, replace=True, p=weights).tolist()
            ids = torch.tensor(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vectors = model.encode({key: value[ids] for key, value in tokens.items()}, text_bytes[ids])
                logits = model.classify(vectors)
                losses = []
                for slot in CLASSIFIER_SLOTS:
                    ys = labels[slot][ids]
                    usable = ys >= 0
                    if usable.any():
                        losses.append(F.cross_entropy(logits[slot][usable].float(), ys[usable], weight=ce_weights[slot]))
                pools = []
                positive_queries = []
                for query, i in enumerate(batch):
                    record = records[i]
                    if not record["rank_usable"]:
                        pools.append([])
                        continue
                    pool = record["pool"]
                    positive = next(c for c in pool if c["value"] == record["rank_truth"])
                    negatives = [c for c in pool if c["value"] != record["rank_truth"]]
                    chosen = rng.choice(len(negatives), min(15, len(negatives)), replace=False).tolist()
                    pools.append([positive] + [negatives[j] for j in chosen])
                    positive_queries.append(query)
                ranks = rank_batch(model, vectors, pools, keys, scales, device)
                if positive_queries:
                    losses.append(torch.stack([-F.log_softmax(ranks[j].float(), dim=0)[0] for j in positive_queries]).mean())
                loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
            if step % 25 == 0:
                print("ENCODER_STEP", epoch, step, steps, "loss", round(float(loss.detach()), 4), flush=True)
        record = {"epoch": epoch, "loss": total_loss / steps, "seconds": time.time() - started}
        if validation is not None:
            record["validation"] = evaluate(model, records, validation, tensors, classes, keys, scales)
            score = record["validation"]["selection_score"]
            if score > best:
                best = score
                best_epoch = epoch
                torch.save(model.state_dict(), Path(output) / "validation_checkpoint.pt")
        history.append(record)
        save_json(Path(output) / ("validation_history.json" if validation is not None else "full_training_history.json"), history)
        print("ENCODER_EPOCH", json.dumps(record), flush=True)
        if early_stopping and epoch >= 4 and epoch - best_epoch >= 3:
            break
    return best_epoch, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="artifacts/generation_v6")
    p.add_argument("--train", default="dataset/ERP_원재료_training.csv")
    p.add_argument("--out", default="artifacts/encoder_v7")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training was requested; CUDA is unavailable")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "heads.pt").exists():
        raise ValueError("Output already contains a trained model; choose a fresh output directory")
    started = time.time()
    records, classes, provenance = load_training_data(args.base, args.train)
    train_indices = [i for i, r in enumerate(records) if r["fold"] != 0]
    validation_indices = [i for i, r in enumerate(records) if r["fold"] == 0]
    keys, validation_scales = candidate_metadata(records, train_indices)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tensors = prepare_inputs(records, tokenizer, classes, 128, "cuda")
    config = {"architecture": "shared_multilingual_encoder_five_classifiers_and_input_candidate_ranker",
              "model_name": args.model, "public_slots": list(CLASSIFIER_SLOTS) + ["part_number"],
              "classes": classes, "feature_keys": keys, "max_length": 128, "freeze_layers": 6,
              "training_only": True, "inference_rows_used": 0, "batch_size": args.batch_size,
              "seed": 42, "selection": "Mean of five task Macro-F1 scores and end-to-end PN Hit@1 on grouped training fold 0"}
    save_json(out / "training_data_audit.json", provenance)
    save_json(out / "encoder_config.json", config)
    seed_all(42)
    model = ERPEncoder(args.model, classes, len(keys)).cuda()
    print("ENCODER_DEVICE", torch.cuda.get_device_name(0), "trainable_parameters",
          sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    selected, history = fit_epochs(model, records, train_indices, tensors, classes, keys, validation_scales,
                                  args.epochs, args.batch_size, validation_indices, out, True)
    model.load_state_dict(torch.load(out / "validation_checkpoint.pt", map_location="cuda", weights_only=True))
    validation = evaluate(model, records, validation_indices, tensors, classes, keys, validation_scales)
    save_json(out / "grouped_validation.json", {"selected_epoch": selected, "metrics": validation,
                                               "feature_scales": validation_scales, "validation_fold": 0,
                                               "training_rows": len(train_indices), "validation_rows": len(validation_indices)})
    del model
    torch.cuda.empty_cache()
    seed_all(43)
    all_indices = list(range(len(records)))
    _, scales = candidate_metadata(records, all_indices, keys=keys)
    model = ERPEncoder(args.model, classes, len(keys)).cuda()
    fit_epochs(model, records, all_indices, tensors, classes, keys, scales, selected, args.batch_size, output=out)
    model.backbone.save_pretrained(out / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}, out / "heads.pt")
    config.update(feature_scales=scales, selected_epochs=selected,
                  pretrained_revision=getattr(model.backbone.config, "_commit_hash", None),
                  trained=True, device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, elapsed_seconds=time.time() - started,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(out / "encoder_config.json", config)
    save_json(out / "completion.json", {"training_complete": True, "inference_used_for_fit": False,
                                       "selected_epochs": selected, "validation": validation,
                                       "seconds": time.time() - started})
    print("ENCODER_TRAINING_COMPLETE", json.dumps(config, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
