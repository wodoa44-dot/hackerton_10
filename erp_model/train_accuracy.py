"""Grouped, training-only comparison of identifier rankers; fits the final CUDA model."""
import copy
import csv
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np

from .accuracy_candidates import candidates
from .company_schema import CompanySchemas
from .features import Features
from .models import RankBundle
from .preprocess import preprocess, identifier_candidates
from .train import read_csv, write_json
from .train_generation import cpu_inference


def score(bundle, rows, labels):
    good=covered=0
    ranked=bundle.rankings(rows)
    for row,truth,result in zip(rows,labels,ranked):
        eligible=[i for i,c in enumerate(row.candidates) if len(c["value"])==len(truth)]
        covered += truth in {row.candidates[i]["value"] for i in eligible}
        if eligible and result["ensemble"]:
            best=min(eligible,key=lambda i:result["ensemble"][i])
            good += row.candidates[best]["value"]==truth
    return {"n":len(rows),"correct":good,"coverage":covered,"accuracy":good/len(rows) if rows else None}


def main():
    start=time.time()
    base=Path("artifacts/generation_v5");out=Path("artifacts/accuracy_v6");out.mkdir(exist_ok=True)
    book=CompanySchemas(base/"company_rules.json")
    data=read_csv("dataset/ERP_원재료_training.csv")
    prepared=[preprocess(row["품목명"],row["규격명"]) for row in data]
    assignments={int(row["source_row"])-2:int(row["fold"]) for row in
                 csv.DictReader((base/"fold_assignments.csv").open(encoding="utf-8-sig"))}
    keep=sorted(assignments)
    parsed={i:book.parse(data[i]["품목코드"]) for i in keep}
    eligible=[i for i in keep if parsed[i] and parsed[i]["values"]["part_number"] and
              any(p["key"] in {"pn","sub","model","name"} for p in book.layout(parsed[i]["category_id"])[0]["part_number"])]
    values={}
    for i in eligible: values.setdefault(prepared[i].key,set()).add(parsed[i]["values"]["part_number"])
    indices=[i for i in eligible if len(values[prepared[i].key])==1]
    labels=[parsed[i]["values"]["part_number"] for i in indices]
    old=[];new=[]
    for i in indices:
        a=copy.copy(prepared[i]);a.candidates=identifier_candidates(a.normalized);old.append(a)
        b=copy.copy(prepared[i]);b.candidates=candidates(b.normalized);new.append(b)
    results=[]
    for fold in range(3):
        train=[j for j,i in enumerate(indices) if assignments[i]!=fold]
        validation=[j for j,i in enumerate(indices) if assignments[i]==fold]
        shared=Features("classification",False).fit([prepared[i] for i in keep if assignments[i]!=fold]).text
        item={"fold":fold}
        for name,rows in [("baseline",old),("improved",new)]:
            model=RankBundle(True).fit([rows[j] for j in train],[labels[j] for j in train],shared)
            item[name]=score(model,[rows[j] for j in validation],[labels[j] for j in validation])
            print("IDENTIFIER_CV",fold,name,item[name],flush=True)
        results.append(item)
    totals={name:{key:sum(item[name][key] for item in results) for key in ["n","correct","coverage"]}
            for name in ["baseline","improved"]}
    for value in totals.values():value["accuracy"]=value["correct"]/value["n"]
    selected="improved" if totals["improved"]["correct"]>=totals["baseline"]["correct"] else "baseline"
    shared=joblib.load(base/"preprocessor.joblib")["text"]
    final=RankBundle(True).fit(new if selected=="improved" else old,labels,shared)
    gpu=cpu_inference(final)
    joblib.dump(final,out/"part_number.joblib",compress=3)
    report={"training_only":True,"selection_metric":"Grouped OOF identifier accuracy, conditional on the requested identifier width",
            "final_evaluation_used_for_selection":False,"folds":results,"totals":totals,
            "selected":selected,"gpu_training_verified":gpu,"elapsed_seconds":time.time()-start}
    write_json(out/"identifier_validation.json",report)
    print("IDENTIFIER_DONE",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
