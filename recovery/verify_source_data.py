"""Independently verify restored source/data against facts recorded before the loss."""
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from erp_model.company_schema import CompanySchemas,actual_prefix
from erp_model.preprocess import preprocess
from erp_model.accuracy_candidates import candidates
from erp_model.schema import SLOTS

def read(name):
    with (ROOT/'dataset'/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def main():
    train=read('ERP_원재료_training.csv');evaluation=read('ERP_원재료_inference.csv');full=read('ERP_원재료_전체.csv')
    assert (len(train),len(evaluation),len(full))==(11399,2850,14249)
    rows=lambda values:Counter(tuple(sorted(row.items())) for row in values)
    assert rows(train+evaluation)==rows(full)
    expected={'training':'59445267a24e1f1a5f98751cb8eb758b98b1f5b9229bc7441c29fbd486147db8',
              'inference':'fd8d8adcb201099f9ab13a054c3fdcdbff6f4ec18765ea56e2736176ad6ea9c2'}
    hashes={}
    for name,values in [('training',train),('inference',evaluation)]:
        digest=hashlib.sha256(json.dumps(sorted([sorted(row.items()) for row in values]),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert digest==expected[name],name
        hashes[name]=digest
    book=CompanySchemas(ROOT/'config/company_rules.json')
    prepared=[preprocess(r['품목명'],r['규격명']) for r in train]
    evaluation_keys={preprocess(r['품목명'],r['규격명']).key for r in evaluation}
    import re
    keep=[i for i,p in enumerate(prepared) if p.key not in evaluation_keys and re.match(r'[A-Z0-9]',train[i]['품목코드'].upper())]
    assert len(keep)==10834
    parsed=[book.parse(r['품목코드']) for r in train]
    assert sum(bool(parsed[i]) for i in keep)==1931
    assert sum(bool(book.parse(r['품목코드'])) for r in evaluation)==473
    expected_counts={'classification':10629,'trait':1521,'part_number':724,'case':1386,'packing':1423,'vendor':1883}
    found={};rank_groups=0
    for slot in SLOTS:
        eligible=keep if slot=='classification' else [i for i in keep if parsed[i] and parsed[i]['values'][slot]
                  and (slot!='part_number' or any(p['key'] in {'pn','sub','model','name'} for p in book.layout(parsed[i]['category_id'])[0][slot]))]
        label=lambda i:actual_prefix(train[i]['품목코드']) if slot=='classification' else parsed[i]['values'][slot]
        groups=defaultdict(set)
        for i in eligible:groups[prepared[i].key].add(label(i))
        fit=[i for i in eligible if len(groups[prepared[i].key])==1]
        found[slot]=len(fit)
        assert len(fit)==expected_counts[slot],(slot,len(fit))
        if slot=='part_number':
            rank_groups=sum(label(i) in {c['value'] for c in candidates(prepared[i].normalized)} for i in fit)
            assert rank_groups==546,rank_groups
    for code in ['RCP10F1232TPY','1775100D9A-B-','25712A00SR5H-','259014D471-WA','25911018101RN','DW1N4148-D3TV']:
        result=book.parse(code);assert result,code
        assert book.assemble(result['category_id'],result['values'])==code
    for p in ROOT.rglob('*.py'):
        if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8'))
    report={'status':'passed','original_csv_rows':{'training':len(train),'inference':len(evaluation),'full':len(full)},
            'csv_canonical_hashes_match_previous_release':hashes,'retained_training_rows':len(keep),
            'parseable_training_rows':1931,'parseable_inference_rows':473,'reproduced_training_label_counts':found,
            'reproduced_rank_groups':rank_groups,'python_syntax':'passed','trained_weights_recovered':False}
    (ROOT/'recovery/source_data_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
