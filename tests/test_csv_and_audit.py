import csv,io,json,tempfile,time,unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from erp_model.batch_csv import CsvJobs,parse_csv
from erp_model.evaluation_audit import displayed_slot,summarize_slot
from erp_model.company_schema import CompanySchemas
from erp_model.spectrum import describe_code,SpectrumPredictor
from erp_model.registry import Registry

ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/"artifacts/generation_v6"


class FakePredictor:
    def __init__(self):
        self.state=SimpleNamespace(manifest={"version":"test"},book=CompanySchemas(),registry=Registry([],{"scope":"training_only"}))
        self.calls=[]
    def events(self,name,spec,**kwargs):
        self.calls.append((name,spec))
        if spec=="ERROR":raise RuntimeError("row error")
        code=None if spec=="NONE" else "RCP10F1232TPY"
        detail=describe_code(self.state.book,code) if code else {"values":{},"parts":[]}
        yield {"slot":"assembly","display_code":code,"auto_confirmed":False,
               "predicted_values":detail["values"],"parts":detail["parts"],
               "spectrum":[{"code":code,**detail}] if code else [],"code_source":"model_assembled"}


def completed(jobs,job_id):
    for _ in range(300):
        result=jobs.get(job_id)
        if result["status"] in {"completed","failed"}:return result
        time.sleep(.01)
    raise AssertionError("job did not finish")


class CsvTests(unittest.TestCase):
    def test_encodings_and_optional_code_column(self):
        for codec in ["utf-8-sig","cp949"]:
            headers,rows,mapping=parse_csv("품목명,규격명\n저항,12.3K\n".encode(codec))
            self.assertEqual(rows[0]["품목명"],"저항")
            self.assertEqual(mapping["code"],"품목코드");self.assertEqual(rows[0]["품목코드"],"")
    def test_quoted_commas_and_blank_spec(self):
        _,rows,m=parse_csv(b'name,spec,code\nRES,"12.3K, REEL",\nRES,,00123\n')
        self.assertEqual(rows[0]["spec"],"12.3K, REEL");self.assertEqual(rows[1]["code"],"00123")
    def test_bad_headers_and_malformed_rows_rejected(self):
        for data in [b'code,spec\n,12K\n',b'name,name,code\nA,B,\n',b'name,spec\nA,B,C\n',b'']:
            with self.assertRaises(ValueError):parse_csv(data)
    def test_only_blank_codes_filled_and_original_values_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            q=FakePredictor();jobs=CsvJobs(directory,lambda:q)
            try:
                data="품목명,규격명,품목코드,메모\nRES,12K,000123,원본\nRES,12K,-,유지\nRES,12.3K,,예측\nRES,ERROR,,오류\nRES,NONE,,불가\n,12K,,빈이름\nRES,12.3K,,중복\n".encode("cp949")
                job=jobs.create(data);result=completed(jobs,job["id"])
                self.assertEqual(result["status"],"completed",result)
                self.assertEqual((result["total"],result["processed"],result["kept"],result["filled"],result["errors"],result["unfilled"]),(7,7,2,2,1,2))
                out=(Path(directory)/"batch_jobs"/job["id"]/"result.csv").read_bytes()
                self.assertTrue(out.startswith(b'\xef\xbb\xbf'))
                rows=list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
                self.assertEqual([r["품목코드"] for r in rows[:3]],["000123","-","RCP10F1232TPY"])
                self.assertEqual(rows[0]["메모"],"원본")
                self.assertEqual(rows[2]["예측_세분류품번"],"1232")
                self.assertEqual(rows[2]["검토필요"],"예")
                self.assertEqual(q.calls.count(("RES","12.3K")),1)
                self.assertFalse(q.state.registry.entries)  # Upload labels never enter the lookup index.
                jobs.jobs.clear();self.assertEqual(jobs.get(job["id"])["status"],"completed")
                with self.assertRaises(KeyError):jobs.get("../manifest")
            finally:jobs.close()
    def test_output_metadata_never_overwrites_input_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs=CsvJobs(directory,FakePredictor)
            try:
                job=jobs.create("品目,name,spec,code,自动,자동처리상태\nA,RES,12K,,B,내기록\n".encode())
                self.assertEqual(completed(jobs,job["id"])["status"],"completed")
                text=(Path(directory)/"batch_jobs"/job["id"]/"result.csv").read_text(encoding="utf-8-sig")
                row=next(csv.DictReader(io.StringIO(text)))
                self.assertEqual(row["자동처리상태"],"내기록");self.assertTrue(row["자동처리상태_2"])
            finally:jobs.close()
    def test_http_job_upload_progress_download_and_errors(self):
        from erp_model.service import create_app
        app=create_app(DIRECTORY);q=FakePredictor();app.state.material_factory=lambda slot=None:q
        with TestClient(app) as client:
            r=client.post("/batch-csv",content="품목명,규격명,품목코드\nRES,12K,\n".encode(),headers={"Content-Type":"text/csv"})
            self.assertEqual(r.status_code,202,r.text)
            job_id=r.json()["id"];done=completed(app.state.csv_jobs,job_id)
            self.assertEqual(done["status"],"completed")
            self.assertEqual(client.get("/batch-csv/"+job_id).json()["filled"],1)
            result=client.get("/batch-csv/"+job_id+"/download")
            self.assertEqual(result.status_code,200);self.assertIn("RCP10F1232TPY",result.content.decode("utf-8-sig"))
            self.assertEqual(client.post("/batch-csv",content=b'code\n123\n').status_code,422)
            self.assertEqual(client.get("/batch-csv/"+'0'*32).status_code,404)
    def test_runtime_web_registry_excludes_all_evaluation_inputs_and_codes(self):
        q=SpectrumPredictor(DIRECTORY)
        from erp_model.preprocess import preprocess
        with (ROOT/"dataset/ERP_원재료_inference.csv").open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
        self.assertFalse({preprocess(r["품목명"],r["규격명"]).key for r in rows}&{tuple(e["normalized_input"]) for e in q.state.registry.entries})
        self.assertFalse({r["품목코드"].upper() for r in rows}&{e["code"].upper() for e in q.state.registry.entries})


class EvaluationAuditTests(unittest.TestCase):
    def test_selected_unmapped_field_does_not_get_credit_from_unused_head(self):
        a={"spectrum":[{"values":{"packing":None}}]}
        self.assertEqual(displayed_slot(a,{"predicted_value":"TP"},"packing"),"")
        self.assertEqual(displayed_slot({"spectrum":[]},{"predicted_value":"TP"},"packing"),"TP")
    def test_majority_and_placeholder_metrics_keep_separate_denominators(self):
        result=summarize_slot(["-","-","W"],["-","-","Y"],Counter({"-":20,"W":3}),20)
        self.assertEqual(result["eligible_n"],3);self.assertEqual(result["coverage"],3/20)
        self.assertEqual(result["placeholder_n"],2);self.assertEqual(result["designated_n"],1)
        self.assertEqual(result["designated_accuracy"],0);self.assertEqual(result["training_majority_baseline"],2/3)
    def test_report_can_be_recomputed_from_exported_displayed_values(self):
        m=json.loads((DIRECTORY/"spectrum_metrics.json").read_text())
        with (DIRECTORY/"prediction_parts.csv").open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
        book=CompanySchemas(DIRECTORY/"company_rules.json");pairs=[]
        for row in rows:
            parsed=book.parse(row["원본정답"])
            if parsed and parsed["values"]["packing"]:pairs.append((parsed["values"]["packing"],row["packing"]))
        self.assertAlmostEqual(m["slots"]["packing"]["accuracy"],sum(t==p for t,p in pairs)/len(pairs))
        self.assertEqual(m["slot_audit"]["vendor"]["designated_n"],67)


if __name__=="__main__":unittest.main()
