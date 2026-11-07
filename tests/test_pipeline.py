import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import joblib
from erp_model.preprocess import preprocess,normalize
from erp_model.schema import labels,SLOTS
from erp_model.inference import Predictor,assembly
from erp_model.features import Features

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'artifacts/run1'
class FakeClassifier:
    issue=None
    def probabilities(self,rows,classes):return {'ensemble':np.ones((len(rows),1))}
class FakeRank:
    issue=None
    def rankings(self,rows):
        return [{'ensemble':[1. if c['value']=='1N4148-' else float(3+i) for i,c in enumerate(r.candidates)]} for r in rows]

def fixture():
    p=Predictor.__new__(Predictor);p.slots=SLOTS
    vals={'classification':'D','trait':'W','case':'D3','packing':'T','vendor':'V'}
    p.manifest={'slots':{s:{'classes':[v],'model':'ensemble','thresholds':{v:{'score':.5,'margin':.1}},'unsafe_labels':[]} for s,v in vals.items()}}
    p.manifest['slots']['part_number']={'rank_confirmation_enabled':True,'max_mean_rank':1.}
    p.models={s:FakeClassifier() for s in vals};p.models['part_number']=FakeRank()
    p.shared={'text':Features('classification').fit([preprocess('DIODE','1N4148 SAMYANG PANJIT'),preprocess('DIODE','1N4148 VISHAY')]).text}
    return p

class ContractTests(unittest.TestCase):
    def test_decomposition(self):
        actual,valid,why=labels('DW1n4148-D3TV')
        self.assertTrue(valid);self.assertEqual([actual[s] for s in SLOTS],['D','W','1N4148-','D3','T','V'])
    def test_normalization(self):
        self.assertEqual(normalize('  ｄｉｏｄｅ　DO–35  0.5A 1n4148- '),'DIODE DO-35 0.5A 1N4148-')
        self.assertEqual(preprocess('DIODE',None).normalized['규격명'],'')
    def test_candidates(self):
        p=preprocess('DIODE','GBU 606 LBAT54SLT1H 1n4148-')
        cs={c['value']:c for c in p.candidates}
        for value in ['GBU606-','BAT54S-','1N4148-']:self.assertIn(value,cs)
        self.assertTrue(any(o['matched_text']=='GBU 606' for o in cs['GBU606-']['origins']))
        self.assertTrue(any(o['matched_text']=='1n4148-' for o in cs['1N4148-']['origins']))
        self.assertEqual(len(cs),len(p.candidates))
    def test_packing_tp_boundary(self):
        self.assertEqual(preprocess('DIODE','ATPG').structured['packing']['tp'],0)
        self.assertEqual(preprocess('DIODE','TP').structured['packing']['tp'],1)
    def test_vendor_only_review_streams_other_five(self):
        p=fixture()
        with patch('erp_model.inference.preprocess',wraps=preprocess) as fn:
            events=list(p.events('DIODE','1N4148 SAMYANG, PANJIT'))
        self.assertEqual(fn.call_count,1);self.assertEqual(len(events),7)
        by={e['slot']:e for e in events}
        self.assertEqual(by['vendor']['status'],'review');self.assertIsNone(by['vendor']['value'])
        self.assertIn('conflicting',by['vendor']['reason']);self.assertEqual(len(by['vendor']['evidence']),2)
        for s in SLOTS[:-1]:self.assertEqual(by[s]['status'],'confirmed')
        self.assertEqual(events[-1]['slot'],'assembly');self.assertEqual(events[-1]['status'],'review')
    def test_review_still_returns_predicted_code(self):
        events=list(fixture().events('DIODE','1N4148 SAMYANG, PANJIT'))
        result=events[-1]
        self.assertEqual(result['status'],'review')
        self.assertIsNone(result['value'])
        self.assertEqual(result['predicted_code'],'DW1N4148-D3TV')
        self.assertEqual(result['prediction_status'],'available')
        self.assertEqual(len(result['predicted_code']),13)
    def test_missing_slot_does_not_invent_code(self):
        events=list(fixture().events('DIODE','1N4148 VISHAY'))[:-1]
        events=[e for e in events if e['slot']!='part_number']
        self.assertIsNone(assembly(events)['predicted_code'])
    def test_other_classification_has_no_general_diode_code(self):
        events=list(fixture().events('DIODE','1N4148 VISHAY'))[:-1]
        next(e for e in events if e['slot']=='classification').update(value='OTHER')
        result=assembly(events)
        self.assertIsNone(result['predicted_code'])
        self.assertEqual(result['prediction_reason'],'Classification is not D')
    def test_full_assembly(self):
        events=list(fixture().events('DIODE','1N4148 VISHAY'))
        self.assertEqual(events[-1]['value'],'DW1N4148-D3TV')
    def test_slot_error_isolation(self):
        p=fixture()
        def fail(*args):raise RuntimeError('controlled failure')
        p.models['case'].probabilities=fail
        es=list(p.events('DIODE','1N4148 VISHAY'));by={e['slot']:e for e in es}
        self.assertEqual(len(es),7);self.assertEqual(by['case']['status'],'error')
        for s in SLOTS:
            if s!='case':self.assertEqual(by[s]['status'],'confirmed')
        self.assertEqual(es[-1]['status'],'review')
    def test_invalid_assembly(self):
        es=list(fixture().events('DIODE','1N4148 VISHAY'))[:-1]
        next(e for e in es if e['slot']=='case')['value']='ZZ'
        self.assertEqual(assembly(es)['status'],'review')

@unittest.skipUnless((MODELS/'manifest.json').exists(),'trained models required')
class TrainedTests(unittest.TestCase):
    def test_reload_and_batch_identical(self):
        names=[('DIODE','1N4148 VISHAY'),('ZENER DIODE','BZT52 SAMYANG PANJIT'),('RES CHIP','10K 1/4W')]
        p=Predictor(MODELS);q=Predictor(MODELS)
        batch=p.batch([preprocess(*x) for x in names])
        for pair,b in zip(names,batch):
            a={x['slot']:x for x in p.events(*pair)};c={x['slot']:x for x in q.events(*pair)}
            self.assertEqual(a,c);self.assertEqual(a,{x['slot']:x for x in b})
    def test_single_loads_only_one_slot(self):
        real=joblib.load
        with patch('erp_model.inference.joblib.load',wraps=real) as load:
            p=Predictor(MODELS,'vendor');events=list(p.events('DIODE','1N4148 VISHAY'))
        self.assertEqual([Path(c.args[0]).name for c in load.call_args_list],['preprocessor.joblib','vendor.joblib'])
        self.assertEqual(len(events),1);self.assertEqual(events[0]['slot'],'vendor')
    def test_common_tfidf_once(self):
        p=Predictor(MODELS)
        a,b=p.shared['text']
        with patch.object(a,'transform',wraps=a.transform) as at,patch.object(b,'transform',wraps=b.transform) as bt:
            list(p.events('DIODE','1N4148 VISHAY'))
        self.assertEqual(at.call_count,1);self.assertEqual(bt.call_count,1)
    def test_disjoint_keys_and_folds(self):
        audit=json.loads((MODELS/'data_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['normalized_input_overlap_after_exclusion'],0)
        with (MODELS/'fold_assignments.csv').open(encoding='utf-8-sig') as fh:rows=list(csv.DictReader(fh))
        seen={}
        for r in rows:
            key=(r['normalized_name'],r['normalized_spec'])
            if key in seen:self.assertEqual(seen[key],r['fold'])
            seen[key]=r['fold']
        with (ROOT/'dataset/ERP_원재료_inference.csv').open(encoding='utf-8-sig') as fh:eval_keys={preprocess(r['품목명'],r['규격명']).key for r in csv.DictReader(fh)}
        self.assertFalse(set(seen)&eval_keys)
    def test_final_predictions_complete(self):
        counts={}
        with (MODELS/'final_predictions.jsonl').open(encoding='utf-8') as fh:
            for line in fh:
                r=json.loads(line);counts.setdefault(r['row'],set()).add(r['slot'])
        self.assertEqual(len(counts),2850)
        for slots in counts.values():self.assertEqual(slots,set(SLOTS+['assembly']))

if __name__=='__main__':unittest.main()
