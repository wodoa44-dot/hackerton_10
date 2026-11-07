import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from erp_model.encoder_network import CLASSIFIER_SLOTS, ERPEncoder, byte_ids
from erp_model.train_encoder import rank_batch


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
        self.embeddings = nn.Embedding(100, 8)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])

    def forward(self, input_ids, attention_mask):
        values = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            values = layer(values)
        return types.SimpleNamespace(last_hidden_state=values)


class EncoderTests(unittest.TestCase):
    def test_masks_and_open_candidate_ranker_backpropagate(self):
        classes = {s: ["A", "B"] for s in CLASSIFIER_SLOTS}
        with patch("erp_model.encoder_network.AutoModel.from_pretrained", return_value=FakeBackbone()):
            model = ERPEncoder("fake", classes, 1)
        tokens = {"input_ids": torch.ones(2, 6, dtype=torch.long), "attention_mask": torch.ones(2, 6, dtype=torch.long)}
        vectors = model.encode(tokens, torch.tensor([byte_ids("품목 A", 32), byte_ids("품목 B", 32)]))
        self.assertEqual(set(model.classify(vectors)), set(CLASSIFIER_SLOTS))
        # Empty candidate groups keep their original query positions.
        pools = [[], [{"value": "ABC123-", "features": {"length": 6}},
                      {"value": "XYZ456-", "features": {"length": 6}}]]
        ranks = rank_batch(model, vectors, pools, ["length"], [10], torch.device("cpu"))
        self.assertEqual([len(x) for x in ranks], [0, 2])
        loss = -torch.log_softmax(ranks[1], 0)[0]
        loss.backward()
        self.assertIsNotNone(model.part_number[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.embeddings.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.encoder.layer[0].parameters()))

    def test_byte_encoding_preserves_digits_and_korean(self):
        self.assertNotEqual(byte_ids("00123", 8), byte_ids("123", 8))
        values = byte_ids("저항", 12)
        self.assertEqual(len(values), 12)
        self.assertTrue(all(0 <= v <= 256 for v in values))


if __name__ == "__main__":
    unittest.main()
