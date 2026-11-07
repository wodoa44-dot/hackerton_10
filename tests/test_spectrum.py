import csv,json,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from erp_model.spectrum import SpectrumPredictor,describe_code
from erp_model.code_metrics import semantic_key,numeric_checks
from erp_model.preprocess import preprocess
from erp_model.registry import Registry
from erp_model.schema import SLOTS

ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/"artifacts/generation_v6"


class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor=SpectrumPredictor(DIRECTORY,candidate_limit=10)
        cls.book=cls.predictor.state.book

    def prediction(self,name,spec):
        events=list(self.predictor.events(name,spec))
        self.assertEqual(len(events),8)
        self.assertEqual({e["slot"] for e in events},set(SLOTS)|{"context","assembly"})
        return {e["slot"]:e for e in events}

    def test_one_shared_state_and_preprocess(self):
        one=SpectrumPredictor(DIRECTORY,"part_number")
        self.assertIs(one.state,self.predictor.state)
        with patch("erp_model.spectrum.preprocess",wraps=preprocess) as prep:
            events=list(one.events("RES CHIP","1608 F 12.3K, REEL, YAGEO"))
        self.assertEqual(prep.call_count,1);self.assertEqual(len(events),1)
        self.assertEqual(events[0]["predicted_value"],"1232")

    def test_six_values_and_primary_code_agree(self):
        by=self.prediction("RES CHIP","1608 F 12.3K, REEL, YAGEO")
        a=by["assembly"];self.assertEqual(a["display_code"],"RCP10F1232TPY")
        first=a["spectrum"][0]
        self.assertEqual(self.book.assemble(first["category_id"],first["values"]),first["code"])
        for slot in SLOTS:
            if first["values"][slot]:self.assertEqual(by[slot]["predicted_value"],first["values"][slot])
        self.assertTrue(all(e.get("predicted_value")!="OTHER" for e in by.values()))

    def test_each_candidate_parts_reconstruct_full_code(self):
        a=self.prediction("DIODE","1N4148, DO-35, TAPING, VISHAY")["assembly"]
        self.assertGreater(len(a["spectrum"]),1);self.assertLessEqual(len(a["spectrum"]),10)
        codes=[]
        for c in a["spectrum"]:
            self.assertEqual("".join(p["value"] for p in c["parts"]),c["code"])
            if c["semantic_layout_verified"]:
                self.assertEqual(self.book.assemble(c["category_id"],c["values"]),c["code"])
            codes.append(c["code"])
        self.assertEqual(len(codes),len(set(codes)))

    def test_numeric_input_rule_is_locked_in_generated_spectrum(self):
        by=self.prediction("RES CHIP","1608 F 12.3K, REEL, YAGEO")
        for c in by["assembly"]["spectrum"]:
            if c["source"]=="model_assembled":
                self.assertEqual(c["values"]["part_number"],"1232")
                self.assertTrue(c["input_numeric_consistent"])

    def test_unknown_legacy_body_is_preserved_without_fake_fields(self):
        by=self.prediction("THERMISTOR","IDU,TWI/TWO/THO")
        self.assertFalse(by["context"]["registry_candidates"])
        self.assertTrue(all(c["code"]!="60WH0083541A-R" for c in by["assembly"]["spectrum"] if c["source"].startswith("registry")))
        first=describe_code(self.book,"60WH0083541A-R")
        self.assertFalse(first["semantic_layout_verified"]);self.assertEqual(first["values"]["classification"],"60")
        self.assertIsNone(first["values"]["trait"])
        self.assertEqual(first["parts"][1]["value"],"WH0083541A-R")
        self.assertEqual(by["trait"]["status"],"unsupported_schema")

    def test_ambiguous_exact_registry_returns_spectrum_without_auto_confirmation(self):
        rows=[{"품목코드":"DW1N4148-D3TV","품목명":"DIODE","규격명":"AMB123"},
              {"품목코드":"DW1N4148-D3T-","품목명":"DIODE","규격명":"AMB123"}]
        q=SpectrumPredictor(DIRECTORY,registry=Registry.build(rows,{"scope":"test"}),candidate_limit=5)
        a=list(q.events("DIODE","AMB123"))[-1]
        self.assertEqual({c["code"] for c in a["spectrum"]},{"DW1N4148-D3TV","DW1N4148-D3T-"})
        self.assertFalse(a["auto_confirmed"])
        self.assertTrue(all(not c.get("auto_confirmed") for c in a["spectrum"]))

    def test_refinement_failure_isolated_to_one_slot(self):
        from erp_model.accuracy_runtime import AccuracyPredictor
        original=AccuracyPredictor.predict_slot
        def controlled(model,slot,p):
            if slot=="case":raise RuntimeError("controlled head error")
            return original(model,slot,p)
        with patch.object(AccuracyPredictor,"predict_slot",new=controlled):
            by=self.prediction("DIODE","UNSEENX123")
        self.assertEqual(by["case"]["status"],"error")
        self.assertTrue(all(by[s]["status"]!="error" for s in SLOTS if s!="case"))

    def test_numeric_equivalence_does_not_accept_different_resistance(self):
        a="RCP10F0680TPY";b="RCP10F68R0TPY";wrong="RCP10F0690TPY"
        self.assertEqual(semantic_key(self.book,a),semantic_key(self.book,b))
        self.assertNotEqual(semantic_key(self.book,a),semantic_key(self.book,wrong))
        self.assertIsNone(semantic_key(self.book,"60WH0083541A-R"))

    def test_consistency_is_separate_from_exact_match(self):
        p=preprocess("CAP CHIP","1608 X7R 0.47uF K 50V")
        checks=numeric_checks(self.book,"CC10B474K1ER-",p)
        self.assertTrue(any(c["field"]=="volt" and not c["consistent"] for c in checks))

    def test_api_limit_and_download_allowlist(self):
        from erp_model.service import create_app
        app=create_app(DIRECTORY);client=TestClient(app)
        r=client.post("/predict",json={"name":"DIODE","spec":"1N4148 DO-35 TAPING VISHAY","candidate_limit":3})
        self.assertEqual(r.status_code,200)
        events=[json.loads(line) for line in r.text.splitlines()]
        self.assertLessEqual(len(events[-1]["spectrum"]),3)
        self.assertEqual(client.post("/predict",json={"name":"X","candidate_limit":100}).status_code,422)
        self.assertEqual(client.get("/downloads/manifest.json").status_code,404)
        self.assertEqual(client.get("/downloads/prediction_parts.csv").status_code,200)

    def test_strict_evaluation_has_no_target_code_in_retrieval_index(self):
        report=json.loads((DIRECTORY/"spectrum_metrics.json").read_text())
        self.assertFalse(report["operational_master_used"])
        self.assertEqual(report["evaluation_code_overlap_in_retrieval_index"],0)
        for k in ["1","3","5","10"]:
            self.assertLessEqual(report["manual_exact_at_k"][k]["correct"],report["company_layout_rows"])

    def test_original_export_keeps_all_rows_and_separate_parts(self):
        with (DIRECTORY/"original_code_parts.csv").open(encoding="utf-8-sig") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),14249)
        self.assertTrue(set(SLOTS)<=rows[0].keys())
        target=next(r for r in rows if r["원본코드"]=="60WH0083541A-R")
        self.assertTrue(target["미해석구간"]);self.assertFalse(target["trait"])


if __name__=="__main__":unittest.main()
