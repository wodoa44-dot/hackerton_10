"""Prepare shared audited labels/folds before either comparison model is fitted."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from erp_model.encoders.rules.company_schema import CompanySchemas,actual_prefix
from erp_model.training.data.encoder_data import load_training_data
from erp_model.encoders.preprocess import preprocess
from erp_model.shared.schema import SLOTS
from erp_model.training.data.training_audit import audit_training
from erp_model.training.io import read_csv,write_json


def prepare(train_path,evaluation_path,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(train_path)
    # Evaluation inputs are used only to exclude duplicates; its code labels are not accessed.
    with Path(evaluation_path).open(encoding='utf-8-sig',newline='') as stream:
        eval_keys={preprocess(row['품목명'],row['규격명']).key for row in csv.DictReader(stream)}
    prepared=[preprocess(r['품목명'],r['규격명']) for r in rows]
    keep=[i for i,p in enumerate(prepared) if p.key not in eval_keys and re.match(r'[A-Z0-9]',rows[i]['품목코드'].upper())]
    book=CompanySchemas();parsed={i:book.parse(rows[i]['품목코드']) for i in keep}
    prefix=[actual_prefix(r['품목코드']) for r in rows]
    groups=[json.dumps(prepared[i].key,ensure_ascii=False) for i in keep]
    folds={}
    for fold,(_,valid) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(np.zeros(len(keep)),[prefix[i] for i in keep],groups)):
        for j in valid:folds[keep[j]]=fold
    with (out/'fold_assignments.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['source_row','normalized_name','normalized_spec','fold','classification'])
        for i in keep:writer.writerow([i+2,*prepared[i].key,folds[i],prefix[i]])
    counts={}
    for slot in SLOTS:
        ix=keep if slot=='classification' else [i for i in keep if parsed[i] and parsed[i]['values'][slot]
            and (slot!='part_number' or any(p['key'] in {'pn','sub','model','name'} for p in book.layout(parsed[i]['category_id'])[0][slot]))]
        label=lambda i:prefix[i] if slot=='classification' else parsed[i]['values'][slot]
        conflicts=defaultdict(set)
        for i in ix:conflicts[prepared[i].key].add(label(i))
        chosen=[i for i in ix if len(conflicts[prepared[i].key])==1]
        histogram=Counter(label(i) for i in chosen)
        counts[slot]={'rows':len(chosen),'classes':len(histogram),'labels':dict(histogram),
                      'excluded_conflicting_input_rows':len(ix)-len(chosen)}
    for name in ['company_rules.json','manual_source.json']:shutil.copy2(Path('config')/name,out/name)
    write_json(out/'training_counts.json',counts)
    audit=audit_training(train_path,out)
    records,classes,provenance=load_training_data(out,train_path)
    provenance.update(preparation_only=True,weights_trained=False,
                      inference_targets_used=False,evaluation_inputs_used_only_for_duplicate_exclusion=True,
                      excluded_evaluation_input_rows=sum(p.key in eval_keys for p in prepared),
                      rules_sha256=hashlib.sha256((out/'company_rules.json').read_bytes()).hexdigest(),
                      numeric_review_rows=audit['numeric_review_rows'])
    write_json(out/'learning_data_audit.json',provenance)
    print(json.dumps({'retained_rows':len(records),'label_counts':provenance['label_counts'],
                      'numeric_review_rows':audit['numeric_review_rows'],'fold_counts':provenance['fold_counts'],
                      'weights_trained':False,'inference_targets_used':False},ensure_ascii=False),flush=True)
    return records,classes,provenance


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='dataset/ERP_원재료_training.csv')
    p.add_argument('--evaluation',default='dataset/ERP_원재료_inference.csv')
    p.add_argument('--out',default='artifacts/encoder_training_data')
    a=p.parse_args();prepare(a.train,a.evaluation,a.out)
