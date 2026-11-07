import csv,io,json,tempfile,unittest
from pathlib import Path
from erp_model.company_schema import CompanySchemas
from erp_model.registry import Registry
from erp_model.preprocess import preprocess
from erp_model.spectrum import SpectrumPredictor,describe_code
from erp_model.exploratory import adaptations,coverage_curve,effective_codes
from erp_model.batch_csv import CsvJobs
from test_csv_and_audit import FakePredictor,completed

ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/"artifacts/generation_v6"


class ExploratoryTests(unittest.TestCase):
    def test_pattern_adaptation_uses_only_observed_input_span_and_width(self):
        entry={"code":"48ABC1234--A","normalized_input":["MODULE","ABC1234"]}
        values=list(adaptations(entry,preprocess("MODULE","ABC5678 50V")))
        self.assertEqual([v["code"] for v in values],["48ABC5678--A"])
        self.assertEqual(values[0]["inferred_span"],{"start":2,"end":9,"value":"ABC5678"})
        self.assertFalse(list(adaptations(entry,preprocess("MODULE","ABC56789"))))
    def test_top_k_accepts_correct_second_candidate_and_deduplicates(self):
        book=CompanySchemas()
        records=[("48ABC5678--A",["48ABC1234--A","48ABC5678--A","48ABC5678--A"]),
                 ("DW1N4148-D3TV",["DW1N4148-D3TV"])]
        result=coverage_curve(records,book,3)
        self.assertEqual(result["all"]["points"][0]["rate"],.5)
        self.assertEqual(result["all"]["points"][1]["rate"],1.)
        self.assertEqual(result["all"]["points"][2]["average_candidates"],1.5)
        self.assertEqual(result["unmapped"]["points"][0]["correct"],0)
        self.assertEqual(result["unmapped"]["points"][1]["correct"],1)
    def test_unverified_input_produces_new_template_candidate_without_gold_code(self):
        training=[{"품목명":"MODULE","규격명":"ABC1234","품목코드":"48ABC1234--A"}]
        registry=Registry.build(training,{"scope":"synthetic_training_only"})
        q=SpectrumPredictor(DIRECTORY,registry=registry,candidate_limit=10)
        a=list(q.events("MODULE","ABC5678"))[-1];explore=a["exploratory"]
        self.assertTrue(explore["active"])
        expected=next(c for c in explore["candidates"] if c["code"]=="48ABC5678--A")
        self.assertEqual(expected["source"],"training_pattern_adaptation")
        self.assertFalse(expected["application_verified"]);self.assertFalse(expected["auto_confirmed"])
        self.assertIn("48ABC5678--A",effective_codes(a))
        self.assertNotIn("48ABC5678--A",{e["code"] for e in registry.entries})
    def test_missing_text_match_yields_explicit_low_evidence_candidates(self):
        q=SpectrumPredictor(DIRECTORY)
        a=list(q.events("⏣⏣⏣",""))[-1]
        self.assertIsNone(a["display_code"])
        self.assertTrue(a["exploratory"]["candidates"])
        self.assertTrue(all(c["source"]=="training_frequency_prior" for c in a["exploratory"]["candidates"]))
        self.assertFalse(any(c["auto_confirmed"] for c in a["exploratory"]["candidates"]))
    def test_thermistor_reference_traits_are_filtered_to_its_training_family(self):
        q=SpectrumPredictor(DIRECTORY);a=list(q.events("THERMISTOR","IDU,TWI/TWO/THO"))[-1]
        extra=a["exploratory"];self.assertTrue(extra["active"])
        self.assertEqual(set(extra["head_estimates"]),set(q.slots))
        self.assertNotIn("S",[v["value"] for v in extra["head_estimates"]["trait"]["candidates"]])
        self.assertTrue(all(not h["applicability_verified"] for h in extra["head_estimates"].values()))
        self.assertEqual(extra["registry_provenance"]["scope"],"retained_training_only")
    def test_csv_uses_exploratory_code_when_strict_assembly_is_empty(self):
        class Forced(FakePredictor):
            def events(self,name,spec,**kwargs):
                detail=describe_code(self.state.book,"48ABC5678--A")
                yield {"slot":"assembly","display_code":None,"spectrum":[],"auto_confirmed":False,
                       "exploratory":{"active":True,"candidates":[{"code":"48ABC5678--A","source":"training_frequency_prior",**detail}],
                                      "head_estimates":{}}}
        with tempfile.TemporaryDirectory() as directory:
            jobs=CsvJobs(directory,Forced)
            try:
                job=jobs.create(b"name,spec,code\nMODULE,ABC5678,\n")
                result=completed(jobs,job["id"]);self.assertEqual(result["filled"],1,result)
                text=(Path(directory)/"batch_jobs"/job["id"]/"result.csv").read_text(encoding="utf-8-sig")
                row=next(csv.DictReader(io.StringIO(text)))
                self.assertEqual(row["code"],"48ABC5678--A")
                self.assertEqual(row["자동처리상태"],"추정채움·규칙미검증")
                self.assertEqual(row["검토필요"],"예")
                self.assertEqual(row["규칙미확인_추정후보"],"48ABC5678--A")
            finally:jobs.close()
    def test_frozen_curve_has_fixed_denominators_and_measured_all_k(self):
        m=json.loads((DIRECTORY/"candidate_k_metrics.json").read_text())
        groups=m["groups"]
        self.assertEqual(groups["all"]["n"],2850)
        self.assertEqual(groups["manual"]["n"]+groups["unmapped"]["n"],2850)
        for group in groups.values():
            points=group["points"];self.assertEqual([p["k"] for p in points],list(range(1,11)))
            self.assertEqual([p["correct"] for p in points],sorted(p["correct"] for p in points))
            self.assertTrue(all(p["n"]==group["n"] for p in points))
        for k in range(10):
            self.assertEqual(groups["all"]["points"][k]["correct"],groups["manual"]["points"][k]["correct"]+groups["unmapped"]["points"][k]["correct"])
        self.assertFalse(m["operational_master_used"])
        self.assertEqual(m["evaluation_code_overlap_in_retrieval_index"],0)


if __name__=="__main__":unittest.main()
