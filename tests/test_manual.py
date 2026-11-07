"""Manual-vs-CSV regressions and the independent six-output contract."""
import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from erp_model.company_schema import CompanySchemas
from erp_model.generation import GenerationPredictor, candidates_for
from erp_model.manual_audit import capacity_modes, capacitances
from erp_model.preprocess import preprocess, identifier_candidates
from erp_model.registry import Registry
from erp_model.schema import SLOTS

ROOT = Path(__file__).resolve().parents[1]


class ManualRulesTests(unittest.TestCase):
    def test_customer_codes_are_not_mislabeled_as_product_origin_and_state(self):
        book = CompanySchemas()
        for code in ["JSDSRPT0253--", "JSDSMPT0077--", "JSESSIVT127-0", "9920HD19-01B-"]:
            self.assertIsNone(book.parse(code), code)
        self.assertEqual(book.parse("JBMHM3827--CA")["values"]["vendor"], "C")

    def test_thermistor_has_separate_packing_position(self):
        parsed = CompanySchemas().parse("1775100D9A-B-")
        self.assertEqual(parsed["values"]["classification"], "17")
        self.assertEqual(parsed["values"]["trait"], "75")
        self.assertEqual(parsed["values"]["part_number"], "100D9A-")
        self.assertEqual(parsed["values"]["packing"], "B")
        self.assertEqual(parsed["values"]["vendor"], "-")

    def test_unknown_code_table_values_do_not_pass_assembly(self):
        book = CompanySchemas()
        values = book.parse("RCP10F1232TPY")["values"]
        values["case"] = "99"
        self.assertIsNone(book.assemble("RCP", values))
        self.assertIsNone(book.parse("DW1N4148-99TV"))
        self.assertEqual(book.parse("19SMW25010---")["category_id"], "19")

    def test_fuse_and_varistor_have_different_packing_semantics(self):
        book = CompanySchemas()
        fuse = book.parse("25712A00SR5H-")
        self.assertEqual(fuse["category_id"], "25")
        self.assertEqual(fuse["values"]["part_number"], "2A00SR5H")
        self.assertEqual(fuse["values"]["packing"], "")
        disk = book.parse("259014D471-WA")
        self.assertEqual(disk["category_id"], "2590")
        self.assertEqual(disk["values"]["case"], "14D")
        self.assertEqual(disk["values"]["part_number"], "471-")
        self.assertEqual(disk["values"]["packing"], "W")
        chip = book.parse("25911018101RN")
        self.assertEqual(chip["values"]["case"], "10")
        self.assertEqual(chip["values"]["part_number"], "18101")
        self.assertIsNone(book.parse("259110M1G26RT"))

    def test_micro_units_and_both_electrolytic_encodings(self):
        self.assertEqual(capacitances("47µF 16V"), [47000000])
        book = CompanySchemas()
        row = {"규격명": "47uF", "품목코드": "CEC1H476M5BT-"}
        self.assertEqual(capacity_modes(row, book.parse(row["품목코드"])), ["pf"])
        row["품목코드"] = "CEC1H470M5BT-"
        self.assertEqual(capacity_modes(row, book.parse(row["품목코드"])), ["uf"])

    def test_identifier_widths_are_generated_without_a_gold_code(self):
        p = preprocess("CONNECTOR", "ZX12345678")
        values = {c["value"] for c in identifier_candidates(p.normalized)}
        self.assertIn("ZX12345678", values)
        p = preprocess("LED", "W47B3311Z")
        self.assertIn("47B3311Z", {c["value"] for c in identifier_candidates(p.normalized)})


class ManualRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = GenerationPredictor(ROOT / "artifacts/generation_v5")
        cls.new_items = GenerationPredictor(ROOT / "artifacts/generation_v5",
                                             registry=Registry.build([], {"scope": "empty_test_master"}))

    def run_six(self, name, spec, scheme=None, predictor=None):
        by = {e["slot"]: e for e in (predictor or self.predictor).events(name, spec, scheme)}
        self.assertTrue(set(SLOTS) <= by.keys())
        self.assertNotEqual(by["classification"]["predicted_value"], "OTHER")
        return by

    def test_new_thermistor_uses_all_independent_slots_and_correct_positions(self):
        by = self.run_six("THERMISTOR NTC", "NTX9234 BULK", "COMPANY_17", self.new_items)
        self.assertEqual(by["assembly"]["display_code"], "1775NTX9234B-")
        self.assertEqual(by["packing"]["predicted_value"], "B")
        self.assertEqual(by["case"]["status"], "not_applicable")
        self.assertEqual(by["context"]["manual_pages"], [25])

    def test_new_ten_character_part_is_not_restricted_to_training_catalog(self):
        by = self.run_six("CONNECTOR", "ZX12345678", "COMPANY_19", self.new_items)
        self.assertEqual(by["part_number"]["predicted_value"], "ZX12345678")
        self.assertEqual(by["assembly"]["display_code"], "19ZX12345678-")
        self.assertEqual(by["assembly"]["code_source"], "model_assembled")

    def test_led_manual_transform_preserves_final_suffix(self):
        by = self.run_six("LED", "W47B3311Z TAPING VISHAY", "COMPANY_DL", self.new_items)
        self.assertEqual(by["part_number"]["predicted_value"], "47B3311Z")
        self.assertEqual(by["assembly"]["display_code"][2:10], "47B3311Z")

    def test_customer_record_preserves_code_and_emits_six_explicit_results(self):
        by = self.run_six("CHIP RES", "2012 F 49.9R RC2012F49R9CS")
        self.assertEqual(by["assembly"]["display_code"], "JSDSMPT0077--")
        self.assertEqual(by["assembly"]["code_source"], "registry_exact")
        self.assertEqual(by["classification"]["predicted_value"], "J")
        for slot in SLOTS[1:]:
            self.assertEqual(by[slot]["status"], "unsupported_schema")
            self.assertIsNone(by[slot]["predicted_value"])

    def test_classification_failure_is_isolated(self):
        def controlled(slot, prepared, state):
            if slot == "classification":
                raise RuntimeError("classification unavailable")
            return candidates_for(slot, prepared, state)
        with patch("erp_model.generation.candidates_for", side_effect=controlled):
            by = self.run_six("DIODE", "1N4148 DO-35 TAPING VISHAY", predictor=self.new_items)
        self.assertEqual(by["classification"]["status"], "error")
        self.assertTrue(all(by[s]["status"] != "error" for s in SLOTS[1:]))
        self.assertIsNone(by["assembly"]["display_code"])

    def test_common_rule_error_still_returns_six_model_results(self):
        with patch.object(self.new_items.state.bridge, "query", side_effect=RuntimeError("rules unavailable")):
            by = self.run_six("DIODE", "1N4148 DO-35 TAPING VISHAY", predictor=self.new_items)
        self.assertTrue(all(by[s]["status"] != "error" for s in SLOTS))
        self.assertIn("공통 규칙 해석 오류", " ".join(by["context"]["required_information"]))

    def test_single_slot_does_not_invoke_other_heads(self):
        predictor = GenerationPredictor(ROOT / "artifacts/generation_v5", slot="packing",
                                        registry=Registry.build([], {"scope": "empty_test_master"}))
        with patch("erp_model.generation.candidates_for", wraps=candidates_for) as calls:
            events = list(predictor.events("DIODE", "1N4148 DO-35 TAPING VISHAY"))
        self.assertEqual([c.args[0] for c in calls.call_args_list], ["packing"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["predicted_value"], "T")

    def test_fuse_current_is_converted_and_packing_is_not_fabricated(self):
        by = self.run_six("FUSE", "SERIES 217 3.15A GLASS 5X20", "COMPANY_25", self.new_items)
        self.assertEqual(by["part_number"]["predicted_value"], "2173A15-")
        self.assertEqual(by["packing"]["status"], "not_applicable")
        self.assertEqual(by["assembly"]["display_code"], "25" + "70" + "2173A15-" + "-")

    def test_varistor_shape_and_rating_follow_the_input(self):
        by = self.run_six("VARISTOR", "INR14D681K CNR", "COMPANY_2590", self.new_items)
        self.assertEqual(by["case"]["predicted_value"], "14D")
        self.assertEqual(by["part_number"]["predicted_value"], "681-")
        self.assertIsNotNone(by["assembly"]["display_code"])
        self.assertEqual(by["assembly"]["display_code"][:11], "259014D681-")

    def test_ambiguous_registry_prefixes_still_emit_six_results(self):
        rows = [{"품목명":"AMBIGUOUS PART", "규격명":"ID12345", "품목코드":code}
                for code in ["DW1N4148-D3TV", "RCP10F1232TPY"]]
        predictor = GenerationPredictor(ROOT / "artifacts/generation_v5",
                                        registry=Registry.build(rows, {"scope":"ambiguous_test"}))
        by = self.run_six("AMBIGUOUS PART", "ID12345", predictor=predictor)
        self.assertEqual(len(by["context"]["registry_candidates"]), 2)
        self.assertIsNone(by["assembly"]["display_code"])
        self.assertTrue(all(by[s]["status"] != "error" for s in SLOTS))

    def test_unicode_and_parenthesized_legacy_codes_are_returned_unchanged(self):
        for code in ["N속지(무상제작)", "B202(무지)"]:
            rows = [{"품목명":"LEGACY PACKAGING", "규격명":"REF123", "품목코드":code}]
            predictor = GenerationPredictor(ROOT / "artifacts/generation_v5",
                                            registry=Registry.build(rows, {"scope":"legacy_test"}))
            by = self.run_six("LEGACY PACKAGING", "REF123", predictor=predictor)
            self.assertEqual(by["assembly"]["display_code"], code.upper())
            self.assertEqual(by["assembly"]["code_source"], "registry_exact")
            self.assertEqual(by["trait"]["status"], "unsupported_schema")

    def test_out_of_range_dimension_is_not_silently_truncated_into_a_code(self):
        by = self.run_six("SCREW", "TAPPING T2 PAN M128X10 STEEL", "COMPANY_H", self.new_items)
        self.assertIsNone(by["case"]["predicted_value"])
        self.assertIsNone(by["assembly"]["display_code"])
        self.assertTrue(all(by[s]["status"] != "error" for s in SLOTS))

    def test_training_excludes_conflicts_and_freezes_reviewed_rules(self):
        state = self.predictor.state
        self.assertEqual(state.manifest["manual_source"]["pages"], 30)
        self.assertEqual(state.manifest["manual_source"]["sha256"], "0a50aeb8e3529da605a9a7a98c022e262771c9b56697219fd0f439636e0c6767")
        self.assertTrue(all(v["gpu_training_verified"] for v in state.manifest["slots"].values()))
        self.assertTrue(all(v["scope"] == "retained_training_only" for v in state.manifest["capacity_policy"].values()))
        counts = json.loads((state.directory / "training_counts.json").read_text())
        self.assertGreater(counts["classification"]["excluded_conflicting_input_rows"], 0)
        report = json.loads((state.directory / "manual_comparison.json").read_text())
        self.assertEqual(sum(report["status_counts"].values()), 14249)


if __name__ == "__main__":
    unittest.main()
