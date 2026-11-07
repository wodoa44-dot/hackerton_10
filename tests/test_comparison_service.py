"""HTTP contracts for paired reports; no lost or trained weights are required."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from erp_model.schema import SLOTS
from erp_model.service import create_app


class FakeSpectrum:
    calls=[]
    def __init__(self,directory,slot=None):
        self.directory=Path(directory); self.slot=slot
        manifest=json.loads((self.directory/'manifest.json').read_text())
        self.state=SimpleNamespace(manifest=manifest,book=SimpleNamespace(manual={},categories={}))
        self.candidate_limit=5
    def events(self,name,spec,scheme):
        self.calls.append((self.directory.name,self.slot,self.candidate_limit))
        yield {'slot':self.slot or 'assembly','model_version':self.state.manifest['version']}


class ComparisonServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        for variant,version in [('generation_v6','6.0'),('generation_v7','7.0-char-cnn-augmented')]:
            directory=self.root/variant;directory.mkdir()
            manifest={'version':version,'classification_target':'actual prefix','training_device':'cuda',
                      'slots':{slot:{} for slot in SLOTS}}
            if version.startswith('7'):manifest['encoder_augmentation']={'directory':'encoder'}
            for filename,data in [('manifest.json',manifest),('training_counts.json',{}),('data_audit.json',{})]:
                (directory/filename).write_text(json.dumps(data),encoding='utf-8')
        self.report=self.root/'encoder_comparison';self.report.mkdir()
        (self.report/'comparison.json').write_text('{"selected_k":[1,3,5,10]}')
        (self.report/'top_k_comparison.csv').write_text('k,correct,n\n1,1,2\n')
        part=self.report/'encoder';part.mkdir();(part/'case_predictions.csv').write_text('source_row,slot\n2,case\n')
        self.patch=patch('erp_model.spectrum.SpectrumPredictor',FakeSpectrum);self.patch.start()
        self.client=TestClient(create_app(self.root/'generation_v7'))
    def tearDown(self):
        self.client.app.state.csv_jobs.close();self.patch.stop();self.tmp.cleanup()
    def test_v7_uses_spectrum_and_reports_both_variants(self):
        data=self.client.get('/model-info').json()
        self.assertEqual(data['encoder_comparison']['selected_k'],[1,3,5,10])
        self.assertEqual(data['active_model_variant'],'encoder')
        self.assertTrue(data['model_variants']['encoder']['active'])
        self.assertTrue(data['model_variants']['no_encoder']['available'])
        result=self.client.post('/predict',json={'name':'DIODE','model_variant':'no_encoder','candidate_limit':10})
        self.assertEqual(result.status_code,200)
        self.assertEqual(FakeSpectrum.calls[-1],('generation_v6',None,10))
        self.assertEqual(self.client.post('/predict',json={'name':'DIODE','model_variant':'other'}).status_code,422)
    def test_exports_are_allowlisted_and_keep_variant(self):
        self.assertEqual(self.client.get('/comparison-downloads/top_k_comparison.csv').status_code,200)
        self.assertIn('2,case',self.client.get('/comparison-downloads/encoder/case_predictions.csv').text)
        for path in ['/comparison-downloads/manifest.json','/comparison-downloads/other/case_predictions.csv',
                     '/comparison-downloads/encoder/weights.pt']:
            self.assertEqual(self.client.get(path).status_code,404)
    def test_missing_comparison_is_pending_not_zero_accuracy(self):
        (self.report/'comparison.json').unlink()
        self.assertIsNone(self.client.get('/model-info').json()['encoder_comparison'])


if __name__=='__main__':unittest.main()
