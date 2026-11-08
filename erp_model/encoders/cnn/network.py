"""Small shared UTF-8 character CNN and six ERP task heads; no pretrained model."""
import json
from pathlib import Path

import torch
from torch import nn

CLASSIFIER_SLOTS = ("classification", "trait", "case", "packing", "vendor")
MODEL_NAME = "char_cnn"
EMBEDDING_DIMENSION = 256


def byte_ids(text, width):
    """Keep exact UTF-8 bytes (including leading zeros); reserve zero for padding."""
    values = [value + 1 for value in str(text).encode("utf-8")[:width]]
    return values + [0] * (width - len(values))


class CharacterTokenizer:
    """A portable fixed vocabulary with separate name/specification boundaries."""
    def __init__(self, name_width=128, spec_width=384):
        if min(name_width, spec_width) < 5:
            raise ValueError("Character field widths must be at least five bytes")
        self.name_width = int(name_width)
        self.spec_width = int(spec_width)

    def encode_fields(self, names, specs):
        if len(names) != len(specs):
            raise ValueError("Names and specifications must have matching row counts")
        return {"name_ids": torch.tensor([byte_ids(s, self.name_width) for s in names], dtype=torch.long).reshape(-1, self.name_width),
                "spec_ids": torch.tensor([byte_ids(s, self.spec_width) for s in specs], dtype=torch.long).reshape(-1, self.spec_width)}

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        value = {"kind": "fixed_utf8_bytes", "vocabulary_size": 257, "padding_id": 0,
                 "name_width": self.name_width, "spec_width": self.spec_width}
        (directory / "character_tokenizer.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, directory):
        value = json.loads((Path(directory) / "character_tokenizer.json").read_text(encoding="utf-8"))
        if value.get("kind") != "fixed_utf8_bytes" or value.get("vocabulary_size") != 257 or value.get("padding_id") != 0:
            raise ValueError("Unsupported character tokenizer configuration")
        return cls(value["name_width"], value["spec_width"])


class ByteEncoder(nn.Module):
    output_size = 192

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, 24, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(24, 48, k, padding=k // 2) for k in (3, 5)])

    def forward(self, ids):
        mask = ids.ne(0).unsqueeze(1)
        values = self.embedding(ids).transpose(1, 2)
        pieces = []
        for conv in self.convs:
            out = torch.relu(conv(values))
            mean = (out * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
            maximum = out.masked_fill(~mask, torch.finfo(out.dtype).min).amax(dim=-1)
            maximum = torch.where(mask.any(dim=-1), maximum, torch.zeros_like(maximum))
            pieces.extend((maximum, mean))
        return torch.cat(pieces, dim=-1)


class CharacterBackbone(nn.Module):
    output_size = EMBEDDING_DIMENSION

    def __init__(self):
        super().__init__()
        self.bytes = ByteEncoder()
        self.project = nn.Sequential(nn.Linear(2 * self.bytes.output_size, self.output_size),
                                     nn.GELU(), nn.LayerNorm(self.output_size), nn.Dropout(.1))

    def forward(self, tokens):
        # Shared byte weights; field order and length budgets stay separate.
        name = self.bytes(tokens["name_ids"])
        specification = self.bytes(tokens["spec_ids"])
        return self.project(torch.cat([name, specification], dim=-1))


class ERPEncoder(nn.Module):
    def __init__(self, model_name, classes, candidate_feature_count, freeze_layers=0, revision=None):
        super().__init__()
        if model_name != MODEL_NAME or freeze_layers or revision is not None:
            raise ValueError("Only the locally trained char_cnn architecture is supported")
        self.backbone = CharacterBackbone()
        hidden = self.backbone.output_size
        self.heads = nn.ModuleDict({
            slot: nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(.15),
                                nn.Linear(128, len(classes[slot])))
            for slot in CLASSIFIER_SLOTS
        })
        self.part_number = nn.Sequential(nn.Linear(hidden + ByteEncoder.output_size + candidate_feature_count, 128),
                                         nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))

    def encode(self, tokens, text_bytes=None):
        return self.backbone(tokens)

    def classify(self, vectors):
        return {slot: head(vectors) for slot, head in self.heads.items()}

    def rank(self, vectors, query_indices, candidate_bytes, candidate_features):
        inputs = torch.cat([vectors[query_indices], self.backbone.bytes(candidate_bytes), candidate_features], dim=-1)
        return self.part_number(inputs).squeeze(-1)


def feature_vector(candidate, keys, scales):
    features = candidate.get("features", {})
    return [max(-10., min(10., float(features.get(key, 0)) / scale))
            for key, scale in zip(keys, scales)]
