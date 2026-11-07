"""Regressions for whole-material prefixes, company layouts, and code creation."""
import csv
import unittest
from pathlib import Path
from unittest.mock import patch

from erp_model.company_schema import CompanySchemas, actual_prefix
from erp_model.generation import GenerationPredictor, candidates_for
from erp_model.schema import SLOTS

ROOT = Path(__file__).resolve().parents[1]


class CompanyLayoutTests(unittest.TestCase):
    def test_capacitor_and_resistor_round_trip(self):
        book = CompanySchemas()
        for code in ["RCP10F1232TPY", "CC10B474K1HR-", "DW1N4148-D3TV"]:
            parsed = book.parse(code)
            self.assertIsNotNone(parsed)
            self.assertEqual(book.assemble(parsed["category_id"], parsed["values"]), code)
        self.assertEqual(actual_prefix("60WH0083541A-R"), "60")

    def test_all_recognized_input_rows_round_trip(self):
        book = CompanySchemas()
        count = 0
        for filename in ["ERP_원재료_training.csv", "ERP_원재료_inference.csv"]:
            with (ROOT / "dataset" / filename).open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    code = row["품목코드"].strip().upper()
                    parsed = book.parse(code)
                    if parsed:
                        self.assertEqual(book.assemble(parsed["category_id"], parsed["values"]), code)
                        count += 1
        self.assertGreater(count, 1500)

    def test_incompatible_fixed_separator_is_not_rewritten(self):
        self.assertIsNone(CompanySchemas().parse("9920HD19-01B-"))


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = GenerationPredictor(ROOT / "artifacts/generation_v5")

    def run_prediction(self, name, spec):
        events = list(self.predictor.events(name, spec))
        self.assertEqual(len(events), 8)
        self.assertEqual(events[-1]["slot"], "assembly")
        by = {event["slot"]: event for event in events}
        self.assertTrue(set(SLOTS) <= by.keys())
        self.assertNotEqual(by["classification"]["predicted_value"], "OTHER")
        return by

    def test_actual_multiclass_model_has_all_material_labels(self):
        labels = self.predictor.state.manifest["slots"]["classification"]["classes"]
        self.assertGreater(len(labels), 50)
        self.assertTrue({"D", "R", "C", "60", "40", "19"} <= set(labels))
        self.assertNotIn("OTHER", labels)
        self.assertTrue(self.predictor.state.manifest["slots"]["classification"]["gpu_training_verified"])

    def test_thermistor_has_actual_prefix_and_full_registered_code(self):
        by = self.run_prediction("THERMISTOR", "IDU,TWI/TWO/THO")
        self.assertEqual(by["classification"]["predicted_value"], "60")
        self.assertEqual(by["assembly"]["display_code"], "60WH0083541A-R")
        self.assertEqual(by["assembly"]["code_length"], 14)
        self.assertEqual(by["assembly"]["code_source"], "registry_candidate")
        self.assertEqual(by["trait"]["status"], "unsupported_schema")

    def test_unregistered_resistor_creates_code_from_numeric_value(self):
        by = self.run_prediction("RES CHIP", "1608 F 12.3K, REEL, YAGEO")
        self.assertEqual(by["classification"]["predicted_value"], "R")
        self.assertEqual(by["part_number"]["predicted_value"], "1232")
        self.assertEqual(by["assembly"]["display_code"], "RCP10F1232TPY")
        self.assertEqual(by["assembly"]["code_source"], "model_assembled")
        self.assertEqual(by["assembly"]["status"], "review")
        self.assertTrue(any(e["matched_text"] == "YAGEO" for e in by["vendor"]["evidence"]))

    def test_unregistered_capacitor_creates_code(self):
        by = self.run_prediction("CAP CHIP", "1608 X7R 0.47uF K 50V, REEL")
        self.assertEqual(by["classification"]["predicted_value"], "C")
        self.assertEqual(by["assembly"]["display_code"], "CC10B474K1HR-")

    def test_diode_trait_evidence_overrides_a_neighbor(self):
        by = self.run_prediction("DIODE", "BZT52-C3V3, 3.3V, 0.5W, SOD-123, SAMYANG, PANJIT")
        self.assertEqual(by["trait"]["predicted_value"], "Z")
        self.assertEqual(by["vendor"]["status"], "review")
        self.assertIn("conflicting_manufacturer_evidence", by["vendor"]["reason"])
        self.assertTrue(by["assembly"]["display_code"].startswith("DZ"))

    def test_one_model_error_does_not_stop_other_slots(self):
        def controlled(slot, prepared, state):
            if slot == "case":
                raise RuntimeError("controlled field error")
            return candidates_for(slot, prepared, state)
        with patch("erp_model.generation.candidates_for", side_effect=controlled):
            by = self.run_prediction("DIODE", "1N4148, DO-35, TAPING, VISHAY")
        self.assertEqual(by["case"]["status"], "error")
        self.assertTrue(all(by[slot]["status"] != "error" for slot in SLOTS if slot != "case"))
        self.assertIsNone(by["assembly"]["display_code"])

    def test_single_slot_and_common_preprocessing(self):
        from erp_model.preprocess import preprocess
        predictor = GenerationPredictor(ROOT / "artifacts/generation_v5", "part_number")
        with patch("erp_model.generation.preprocess", wraps=preprocess) as prepare:
            events = list(predictor.events("RES CHIP", "1608 F 12.3K, REEL, YAGEO"))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["slot"], "part_number")
        self.assertEqual(events[0]["predicted_value"], "1232")


if __name__ == "__main__":
    unittest.main()
