"""Contract tests for a fair, frozen tree/encoder feature ablation."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from erp_model.compare_encoders import DISPLAY_KS, KS, paired_deltas, validate_pair
from erp_model.features import Features
from erp_model.generation import GenerationState
from erp_model.models import ClassifierBundle
from erp_model.preprocess import preprocess
from erp_model.schema import SLOTS
from erp_model.train_encoder_augmented import COMMON_FILES, fit_frozen_bundle


class FeatureAugmentationTests(unittest.TestCase):
    def rows(self):
        return [preprocess("DIODE", "1N4148"), preprocess("DIODE", "1N4007"),
                preprocess("DIODE", "BAV99"), preprocess("DIODE", "BAV70")]

    def test_append_vectors_preserves_baseline_and_text_cache_width(self):
        rows = self.rows()
        baseline = Features("trait").fit(rows)
        original = baseline.transform(rows)
        for i, row in enumerate(rows):
            row._shared_char_embedding = np.array([i / 4, 1.0], dtype=np.float32)
        augmented = Features("trait").fit(rows, baseline.text)
        values = augmented.transform(rows)
        self.assertEqual(values.shape[1], original.shape[1] + 2)
        np.testing.assert_allclose(values[:, :-2].toarray(), original.toarray())
        np.testing.assert_allclose(values[:, -2:].toarray(), [row._shared_char_embedding for row in rows])
        self.assertLess(rows[0]._text_cache[1].shape[1], values.shape[1])
        # A legacy/baseline bundle never silently starts using attached vectors.
        del baseline.encoder_dimensions
        np.testing.assert_allclose(baseline.transform(rows).toarray(), original.toarray())

    def test_missing_mixed_and_wrong_width_vectors_fail(self):
        rows = self.rows()
        rows[0]._shared_char_embedding = np.ones(2, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "every fitting row"):
            Features("trait").fit(rows)
        for row in rows:
            row._shared_char_embedding = np.ones(2, dtype=np.float32)
        features = Features("trait").fit(rows)
        fresh = self.rows()
        with self.assertRaisesRegex(ValueError, "requires a shared"):
            features.transform(fresh)
        for row in fresh:
            row._shared_char_embedding = np.ones(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "width differs"):
            features.transform(fresh)
        fresh[0]._shared_char_embedding = np.array([np.nan, 1, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            features.transform(fresh)

    def test_raw_bundle_call_loads_one_shared_encoder_without_prepare(self):
        rows = self.rows()
        for row in rows:
            row._shared_char_embedding = np.ones(2, dtype=np.float32)
        feature = Features("trait").fit(rows)
        calls = []

        def encode(batch, batch_size=64):
            calls.append(len(batch))
            return np.full((len(batch), 2), .25, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            joblib.dump(SimpleNamespace(feature=feature), Path(directory) / "trait.joblib")
            state = GenerationState.__new__(GenerationState)
            state.directory = Path(directory)
            state.shared = feature.text
            state.manifest = {"encoder_augmentation": {"directory": "encoder", "embedding_dimension": 2}}
            state.models, state.model_lock = {}, threading.Lock()
            state._encoder_bundle = SimpleNamespace(encode_vectors=encode)
            state._encoder_lock = threading.Lock()
            raw_model = state.model("trait")
            fresh = self.rows()
            first = raw_model.feature.transform(fresh)
            second = raw_model.feature.transform(fresh)
            self.assertEqual(calls, [4])
            np.testing.assert_allclose(first.toarray(), second.toarray())
            self.assertEqual(first.shape[1], feature.transform(rows).shape[1])
            self.assertEqual(state.text_features(fresh[0]).shape[1], fresh[0]._text_cache[1].shape[1])

    def test_final_refit_clones_actual_tree_parameters_and_kinds(self):
        rows = self.rows()
        truth = ["A", "A", "B", "B"]
        template = ClassifierBundle("trait")
        template.classes = ["A", "B"]
        template.feature.fit(rows)
        matrix = template.feature.transform(rows)
        template.models = {name: DecisionTreeClassifier(max_depth=1, random_state=42).fit(matrix, [0, 0, 1, 1])
                           for name in ("xgb", "lgb")}
        for index, row in enumerate(rows):
            row._shared_char_embedding = np.array([index, 1], dtype=np.float32)
        fitted = fit_frozen_bundle("trait", template, rows, truth, template.feature.text)
        self.assertEqual(fitted.classes, template.classes)
        self.assertEqual(list(fitted.models), list(template.models))
        for name, model in fitted.models.items():
            self.assertIsNot(model, template.models[name])
            self.assertEqual(model.get_params(), template.models[name].get_params())
            self.assertEqual(model.n_features_in_, template.models[name].n_features_in_ + 2)


class ComparisonProtocolTests(unittest.TestCase):
    def test_k_and_paired_counts_keep_fixed_denominators(self):
        self.assertEqual(KS, tuple(range(1, 11)))
        self.assertEqual(DISPLAY_KS, (1, 3, 5, 10))

        def row(index, manual, after):
            return {"source_row": index, "manual_parseable": manual,
                    **{f"hit_at_{k}": k >= after for k in KS}}

        left = [row(2, True, 1), row(3, False, 99)]
        right = [row(2, True, 3), row(3, False, 1)]
        delta = paired_deltas(left, right)
        self.assertTrue(all(point["n"] == 2 for point in delta["all"]))
        self.assertEqual(delta["all"][0]["delta_percentage_points"], 0)
        self.assertEqual(delta["all"][2]["delta_percentage_points"], 50)
        self.assertEqual(delta["manual"][0]["delta_percentage_points"], -100)
        self.assertEqual(delta["unmapped"][0]["delta_percentage_points"], 100)

    def test_comparison_refuses_changed_common_policy_or_model_blend(self):
        with tempfile.TemporaryDirectory() as directory:
            base, augmented = Path(directory) / "a", Path(directory) / "b"
            base.mkdir(); augmented.mkdir()
            for name in COMMON_FILES:
                (base / name).write_text(name, encoding="utf-8")
                (augmented / name).write_text(name, encoding="utf-8")
            manifest = {"slots": {slot: {"model_status": "trained", "classes": ["A", "B"],
                                        "model": "ensemble", "weights": {"xgb": .5, "lgb": .5}}
                                  for slot in SLOTS}}
            changed = copy.deepcopy(manifest)
            changed.update(encoder_augmentation={"embedding_dimension": 2}, inference_rows_used_for_training_or_selection=0)
            (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (augmented / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            self.assertEqual(len(validate_pair(base, augmented)["common_artifact_sha256"]), len(COMMON_FILES))
            changed["slots"]["trait"]["weights"] = {"xgb": 1.0}
            (augmented / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tree selection"):
                validate_pair(base, augmented)
            changed["slots"]["trait"]["weights"] = {"xgb": .5, "lgb": .5}
            (augmented / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
            (augmented / "company_rules.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Common comparison artifact differs"):
                validate_pair(base, augmented)


if __name__ == "__main__":
    unittest.main()
