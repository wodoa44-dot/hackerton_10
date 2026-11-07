import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from erp_model.encoder_network import (CLASSIFIER_SLOTS, MODEL_NAME, ByteEncoder,
                                       CharacterTokenizer, ERPEncoder, byte_ids)
from erp_model.encoder_runtime import EncoderBundle
from erp_model.train_encoder import evaluate, fit_epochs, prepare_inputs, rank_batch


def candidate(value):
    return {"value": value, "features": {"length": len(value)}, "source_column": "규격명",
            "matched_text": value, "transformation": "identity"}


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masks_and_open_candidate_ranker_backpropagate(self):
        classes = {s: ["A", "B"] for s in CLASSIFIER_SLOTS}
        model = ERPEncoder(MODEL_NAME, classes, 1)
        tokens = CharacterTokenizer(32, 48).encode_fields(["다이오드", "저항"], ["ABC123", "XYZ456"])
        vectors = model.encode(tokens)
        logits = model.classify(vectors)
        self.assertEqual(set(logits), set(CLASSIFIER_SLOTS))
        self.assertTrue(all(x.shape == (2, 2) for x in logits.values()))
        # Empty candidate groups keep their original query positions.
        pools = [[], [candidate("ABC123-"), candidate("XYZ456-")]]
        ranks = rank_batch(model, vectors, pools, ["length"], [10], torch.device("cpu"))
        self.assertEqual([len(x) for x in ranks], [0, 2])
        # Only the eligible classification label contributes to classification
        # loss; the other four heads receive no artificial label target.
        loss = F.cross_entropy(logits["classification"][[0]], torch.tensor([0])) - torch.log_softmax(ranks[1], 0)[0]
        loss.backward()
        self.assertIsNotNone(model.part_number[0].weight.grad)
        self.assertGreater(float(model.backbone.bytes.embedding.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.backbone.project[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.heads["vendor"][0].weight.grad)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_000_000)

    def test_pooling_ignores_padding_and_field_boundaries_are_distinct(self):
        torch.manual_seed(4)
        encoder = ByteEncoder().eval()
        short = encoder(torch.tensor([byte_ids("00123", 8)]))
        padded = encoder(torch.tensor([byte_ids("00123", 32)]))
        torch.testing.assert_close(short, padded)
        torch.testing.assert_close(encoder(torch.zeros(1, 16, dtype=torch.long)), torch.zeros(1, 192))
        model = ERPEncoder(MODEL_NAME, {s: ["A", "B"] for s in CLASSIFIER_SLOTS}, 1).eval()
        tokens = CharacterTokenizer(32, 32).encode_fields(["A", "B"], ["B", "A"])
        vectors = model.encode(tokens)
        self.assertFalse(torch.allclose(vectors[0], vectors[1]))

    def test_cpu_training_keeps_label_masks_and_reports_all_top_k(self):
        classes = {s: ["A", "B"] for s in CLASSIFIER_SLOTS}
        records = []
        for i in range(4):
            records.append({"key": ("저항" if i % 2 else "다이오드", f"ABC12{i}"),
                            "labels": {s: ("A" if i % 2 else "B") if s == "classification" or i % 2 else None
                                       for s in CLASSIFIER_SLOTS},
                            "rank_truth": "ABC123-", "rank_usable": True,
                            "pool": [candidate("ABC123-"), candidate("XYZ456-")]})
        tensors = prepare_inputs(records, CharacterTokenizer(32, 48), classes, 48, "cpu")
        self.assertEqual(tensors[2]["trait"].tolist(), [-1, 0, -1, 0])
        model = ERPEncoder(MODEL_NAME, classes, 1)
        with tempfile.TemporaryDirectory() as directory:
            best, history = fit_epochs(model, records, [0, 1], tensors, classes, ["length"], [10.],
                                       1, 2, [2, 3], directory)
            self.assertEqual(best, 1)
            self.assertTrue(np.isfinite(history[0]["loss"]))
            state = torch.load(Path(directory) / "validation_checkpoint.pt", weights_only=True)
            model.load_state_dict(state, strict=True)
        metrics = evaluate(model, records, [2, 3], tensors, classes, ["length"], [10.])
        for slot in (*CLASSIFIER_SLOTS, "part_number"):
            self.assertEqual(set(metrics[slot]["hit_at_k"]), {"1", "3", "5", "10"})
        self.assertEqual(metrics["trait"]["n"], 1)

    def test_small_checkpoint_loads_and_vectors_use_input_only(self):
        classes = {s: ["A", "B"] for s in CLASSIFIER_SLOTS}
        tokenizer = CharacterTokenizer(32, 48)
        model = ERPEncoder(MODEL_NAME, classes, 1).eval()
        prepared = [SimpleNamespace(normalized={"품목명": "다이오드", "규격명": "ABC123"}, candidates=[]),
                    SimpleNamespace(normalized={"품목명": "저항", "규격명": "XYZ456"},
                                    candidates=[candidate("XYZ456-"), candidate("ABC123-")])]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = {"trained": True, "model_name": MODEL_NAME, "classes": classes,
                      "feature_keys": ["length"], "feature_scales": [10.], "embedding_dimension": 256}
            (directory / "encoder_config.json").write_text(json.dumps(config), encoding="utf-8")
            tokenizer.save_pretrained(directory / "tokenizer")
            torch.save(model.state_dict(), directory / "weights.pt")
            self.assertLess((directory / "weights.pt").stat().st_size, 4_000_000)
            loaded = EncoderBundle(directory, device="cpu")
            expected = model.encode(tokenizer.encode_fields([p.normalized["품목명"] for p in prepared],
                                                             [p.normalized["규격명"] for p in prepared])).detach().numpy()
            vectors = loaded.encode_vectors(prepared, batch_size=1)
            # CPU convolution reductions can differ slightly with batch shape.
            np.testing.assert_allclose(vectors, expected, rtol=1e-4, atol=2e-6)
            self.assertEqual(vectors.shape, (2, 256))
            self.assertEqual(vectors.dtype, np.float32)
            prepared[0].candidates = [candidate("CHANGED")]
            np.testing.assert_array_equal(vectors, loaded.encode_vectors(prepared, batch_size=1))
            self.assertEqual(loaded.encode_vectors([]).shape, (0, 256))
            self.assertEqual(loaded.predict([]), [])
            prepared[0].candidates = []
            result = loaded.predict(prepared)
            self.assertEqual(set(result[0]), set(CLASSIFIER_SLOTS) | {"part_number"})
            self.assertEqual(result[0]["part_number"], [])
            self.assertEqual(len(result[1]["part_number"]), 2)
            manifest = {"slots": {s: {"classes": ["B", "A"]} for s in CLASSIFIER_SLOTS}}
            loaded.attach(prepared, manifest)
            self.assertEqual(prepared[0]._encoder_signature, id(loaded))
            self.assertEqual(prepared[0]._generation_probabilities["trait"].shape, (2,))
            self.assertEqual(len(prepared[1]._generation_ranking), len(prepared[1].candidates))

    def test_byte_encoding_preserves_digits_and_korean(self):
        self.assertNotEqual(byte_ids("00123", 8), byte_ids("123", 8))
        values = byte_ids("저항", 12)
        self.assertEqual(len(values), 12)
        self.assertTrue(all(0 <= v <= 256 for v in values))


if __name__ == "__main__":
    unittest.main()
