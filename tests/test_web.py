import importlib.util
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HAVE_HTTPX=importlib.util.find_spec('httpx') is not None
HAVE_QUICKJS=importlib.util.find_spec('quickjs') is not None

@unittest.skipUnless(HAVE_HTTPX,'install requirements-dev.txt for HTTP tests')
class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from erp_model.service import create_app
        cls.client=TestClient(create_app(ROOT/'artifacts/run1'))
    def test_home_and_assets(self):
        html=self.client.get('/')
        self.assertEqual(html.status_code,200)
        self.assertIn('text/html',html.headers['content-type'])
        self.assertIn('품목 정보를 코드로 연결하세요',html.text)
        for path in ['/assets/style.css','/assets/app.js']:
            response=self.client.get(path)
            self.assertEqual(response.status_code,200)
            self.assertGreater(len(response.content),1000)
    def test_real_metrics(self):
        data=self.client.get('/model-info').json()
        self.assertEqual(data['metrics']['evaluation_rows'],2850)
        self.assertEqual(data['metrics']['company_layout_rows'],473)
        self.assertEqual(data['metrics'],data['generation_metrics'])
        self.assertEqual(data['web_lookup_audit']['evaluation_input_overlap'],0)
    def test_all_stream_events(self):
        response=self.client.post('/predict',json={'name':'DIODE','spec':'1N4148 SAMYANG, PANJIT'})
        self.assertEqual(response.status_code,200)
        self.assertIn('application/x-ndjson',response.headers['content-type'])
        events=[json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(events),8)
        self.assertEqual(events[-1]['slot'],'assembly')
        vendor=next(e for e in events if e['slot']=='vendor')
        self.assertEqual(vendor['status'],'review')
        self.assertIn('conflicting_manufacturer_evidence',vendor['reason'])
    def test_single_slot_stream(self):
        response=self.client.post('/predict',json={'name':'DIODE','spec':'1N4148','slot':'part_number'})
        events=[json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(len(events),1);self.assertEqual(events[0]['slot'],'part_number')
    def test_invalid_slot(self):
        self.assertEqual(self.client.post('/predict',json={'name':'DIODE','slot':'unknown'}).status_code,422)
    def test_proxy_prefix(self):
        from fastapi.testclient import TestClient
        from erp_model.service import create_app
        client=TestClient(create_app(ROOT/'artifacts/run1',root_path='/i/OsAdz8rQR7JSXeW6'))
        self.assertEqual(client.get('/i/OsAdz8rQR7JSXeW6/').status_code,200)
        self.assertEqual(client.get('/i/OsAdz8rQR7JSXeW6/assets/app.js').status_code,200)
        self.assertEqual(client.get('/i/OsAdz8rQR7JSXeW6/model-info').status_code,200)

@unittest.skipUnless(HAVE_QUICKJS,'install requirements-dev.txt for JS syntax check')
class JavaScriptTests(unittest.TestCase):
    def test_browser_script_parses(self):
        import quickjs
        js=(ROOT/'erp_model/web/assets/app.js').read_text(encoding='utf-8')
        quickjs.Context().eval('new Function('+json.dumps(js)+')')

if __name__=='__main__':unittest.main()
