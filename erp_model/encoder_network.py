"""Shared multilingual text encoder with six public task heads.

Part numbers are ranked from input-derived candidates, not closed classes.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoModel

CLASSIFIER_SLOTS = ("classification", "trait", "case", "packing", "vendor")


def byte_ids(text, width):
    values = [value + 1 for value in text.encode("utf-8")[:width]]
    return values + [0] * (width - len(values))


class ByteEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 24, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(24, 64, k, padding=k // 2) for k in (3, 5)])

    def forward(self, ids):
        values = self.embedding(ids).transpose(1, 2)
        return torch.cat([torch.relu(conv(values)).amax(dim=-1) for conv in self.convs], dim=-1)


class ERPEncoder(nn.Module):
    def __init__(self, model_name, classes, candidate_feature_count, freeze_layers=6):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=False,
                                                use_safetensors=True, attn_implementation="sdpa")
        self.bytes = ByteEncoder()
        hidden = self.backbone.config.hidden_size + 128
        self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleDict({
            slot: nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(.15),
                                nn.Linear(256, len(classes[slot])))
            for slot in CLASSIFIER_SLOTS
        })
        self.part_number = nn.Sequential(nn.Linear(hidden + 128 + candidate_feature_count, 256),
                                        nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
        for parameter in self.backbone.embeddings.parameters():
            parameter.requires_grad_(False)
        for layer in self.backbone.encoder.layer[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    def encode(self, tokens, text_bytes):
        output = self.backbone(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.norm(torch.cat([pooled, self.bytes(text_bytes)], dim=-1))

    def classify(self, vectors):
        return {slot: head(vectors) for slot, head in self.heads.items()}

    def rank(self, vectors, query_indices, candidate_bytes, candidate_features):
        inputs = torch.cat([vectors[query_indices], self.bytes(candidate_bytes), candidate_features], dim=-1)
        return self.part_number(inputs).squeeze(-1)


def feature_vector(candidate, keys, scales):
    features = candidate.get("features", {})
    return [float(np.clip(float(features.get(key, 0)) / scale, -10, 10))
            for key, scale in zip(keys, scales)]
