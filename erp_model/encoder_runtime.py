"""Input-only encoder inference and an adapter to the existing code spectrum."""
import json
from pathlib import Path

import numpy as np
import torch

from .accuracy_candidates import candidates
from .encoder_network import CLASSIFIER_SLOTS, EMBEDDING_DIMENSION, CharacterTokenizer, ERPEncoder
from .preprocess import preprocess
from .spectrum import SpectrumPredictor
from .train_encoder import rank_batch


class EncoderBundle:
    def __init__(self, directory, device="cuda"):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "encoder_config.json").read_text(encoding="utf-8"))
        if not self.config.get("trained"):
            raise ValueError("Encoder training is incomplete")
        self.device = torch.device(device)
        self.tokenizer = CharacterTokenizer.from_pretrained(self.directory / "tokenizer")
        self.model = ERPEncoder(self.config['model_name'], self.config["classes"],
                                len(self.config["feature_keys"])).to(self.device)
        self.model.load_state_dict(torch.load(self.directory / "weights.pt", map_location=self.device,
                                              weights_only=True), strict=True)
        self.model.eval()

    def _tokens(self, prepared):
        tokens = self.tokenizer.encode_fields([p.normalized["품목명"] for p in prepared],
                                               [p.normalized["규격명"] for p in prepared])
        return {key: value.to(self.device) for key, value in tokens.items()}

    @torch.inference_mode()
    def encode_vectors(self, prepared, batch_size=64):
        """Return input-only float32 features for the identical tree-head A/B fit."""
        if not 1 <= batch_size <= 64:
            raise ValueError("Encoder batch size must be between 1 and 64")
        vectors = [self.model.encode(self._tokens(prepared[start:start + batch_size])).float().cpu().numpy()
                   for start in range(0, len(prepared), batch_size)]
        return np.concatenate(vectors) if vectors else np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    @torch.inference_mode()
    def predict(self, prepared, batch_size=64):
        if not 1 <= batch_size <= 64:
            raise ValueError("Encoder batch size must be between 1 and 64")
        output = []
        for start in range(0, len(prepared), batch_size):
            output.extend(self._predict_batch(prepared[start:start + batch_size]))
        return output

    def _predict_batch(self, prepared):
        vectors = self.model.encode(self._tokens(prepared))
        probabilities = {slot: torch.softmax(logits.float(), -1).cpu().numpy()
                         for slot, logits in self.model.classify(vectors).items()}
        pools = [p.candidates for p in prepared]
        rankings = rank_batch(self.model, vectors, pools, self.config["feature_keys"],
                              self.config["feature_scales"], self.device)
        output = []
        for i, p in enumerate(prepared):
            result = {}
            for slot in CLASSIFIER_SLOTS:
                order = np.argsort(-probabilities[slot][i], kind="stable")
                result[slot] = [{"value": self.config["classes"][slot][j], "score": float(probabilities[slot][i, j]),
                                 "score_type": "encoder_uncalibrated_probability"} for j in order]
            rank_scores = rankings[i].float().cpu().numpy() if rankings else np.empty(0)
            order = np.argsort(-rank_scores, kind="stable")
            result["part_number"] = [{"value": pools[i][j]["value"], "score": float(rank_scores[j]),
                                      "score_type": "encoder_rank_score", "input_origin": {
                                          k: pools[i][j][k] for k in ("source_column", "matched_text", "transformation")}}
                                     for j in order]
            output.append(result)
        return output

    def attach(self, prepared, manifest):
        for p in prepared:
            p.candidates = candidates(p.normalized)
            p._identifier_pool_ready = True
        output = self.predict(prepared)
        for p, result in zip(prepared, output):
            p._generation_probabilities = {}
            for slot in CLASSIFIER_SLOTS:
                values = {c["value"]: c["score"] for c in result[slot]}
                p._generation_probabilities[slot] = np.array([values.get(c, 0.) for c in manifest["slots"][slot]["classes"]])
            ranks = {candidate["value"]: i + 1 for i, candidate in enumerate(result["part_number"])}
            p._generation_ranking = [float(ranks[c["value"]]) for c in p.candidates]
            p._encoder_prediction = result
            p._encoder_signature = id(self)
        return output


class EncoderSpectrumPredictor(SpectrumPredictor):
    def __init__(self, directory="artifacts/generation_v6", slot=None, registry=None, candidate_limit=10,
                 encoder_dir="artifacts/encoder_v7", device="cuda"):
        super().__init__(directory, slot=slot, registry=registry, candidate_limit=candidate_limit)
        self.encoder = EncoderBundle(encoder_dir, device)

    def prepare(self, name, spec, scheme=None, prepared=None):
        p = prepared if prepared is not None else preprocess(name, spec)
        if getattr(p, "_encoder_signature", None) != id(self.encoder):
            self.encoder.attach([p], self.state.manifest)
        p, context = super().prepare(name, spec, scheme, p)
        context["learned_heads"] = "shared_encoder_v7"
        return p, context
