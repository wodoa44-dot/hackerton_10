"""배포 전 HTTP 경로 검증. 실제 모델 대신 명시적 테스트 예측기를 사용합니다."""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from erp_model.api.service import create_app
from erp_model.encoders.rules.company_schema import CompanySchemas
from erp_model.shared.schema import SLOTS

class TestPredictor:
    def __init__(self,directory,slot=None):
        directory=Path(directory)
        self.state=SimpleNamespace(manifest=json.loads((directory/"manifest.json").read_text()),book=CompanySchemas(directory/"company_rules.json"))
    def events(self,name,spec,scheme=None):
        for slot in SLOTS:
            yield {"slot":slot,"predicted_value":"TEST","status":"review"}
        yield {"slot":"assembly","predicted_code":"DEMO_TEST_CODE","display_code":"DEMO_TEST_CODE","status":"review"}

class DeploymentHttpTests(unittest.TestCase):
    def setUp(self):
        self.environment=patch.dict(os.environ,{"ERP_SOURCE_COMMIT":"a"*40})
        self.environment.start()
        self.client=TestClient(create_app(ROOT/"artifacts/generation_v7",predictor_type=TestPredictor,public_report_only=True))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.environment.stop()

    def test_source_version_and_designated_raw_metrics(self):
        health=self.client.get("/health").json()
        self.assertEqual(health["source_commit"],"a"*40)
        info=self.client.get("/model-info").json()
        self.assertEqual(info["active_model_variant"],"encoder")
        self.assertEqual(info["encoder_comparison"]["variants"]["encoder"]["designated"]["raw_heads"]["vendor"]["n"],67)
        self.assertFalse(info["row_result_downloads"])

    def test_public_report_table_and_figure(self):
        self.assertEqual(self.client.get("/reports/encoder-topk/").status_code,200)
        table=self.client.get("/reports/encoder-topk/topk_summary.csv")
        self.assertEqual(table.status_code,200)
        self.assertIn("코드항목",table.content.decode("utf-8-sig"))
        png=self.client.get("/reports/encoder-topk/figures/raw_designated.png")
        self.assertEqual(png.status_code,200)
        self.assertTrue(png.content.startswith(b"\x89PNG"))

    def test_private_rows_and_path_escape_are_blocked(self):
        for route in ["/reports/encoder-topk/private/predictions/whole_codes.csv","/reports/encoder-topk/%2e%2e/private/predictions/whole_codes.csv","/comparison-downloads/paired_predictions.csv","/comparison-downloads/encoder/vendor_predictions.csv","/downloads/prediction_parts.csv"]:
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).status_code,404)

    def test_input_page_keeps_code_and_removes_slot_cards(self):
        text=self.client.get("/").text
        self.assertNotIn("칸별 예측 결과",text)
        self.assertIn("reports/encoder-topk/",text)
        self.assertIn("batch-csv",self.client.get("/assets/app.js").text)

    def test_prediction_event_contract_and_limit_validation(self):
        response=self.client.post("/predict",json={"name":"DEMO","spec":"TEST","candidate_limit":5})
        self.assertEqual(response.status_code,200)
        events=[json.loads(line) for line in response.text.splitlines()]
        self.assertEqual({r["slot"] for r in events},set(SLOTS)|{"assembly"})
        self.assertEqual(self.client.post("/predict",json={"name":"DEMO","candidate_limit":0}).status_code,422)

if __name__=="__main__":
    unittest.main()
