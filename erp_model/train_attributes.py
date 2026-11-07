"""Three shared grouped folds select attribute ensembles using training data only."""
import csv,json,time
from pathlib import Path
from collections import defaultdict
import joblib,numpy as np
from .attribute_models import AttributeBundle
from .company_schema import CompanySchemas,field_slot
from .features import Features
from .preprocess import preprocess
from .train import read_csv,write_json
from .label_quality import unambiguous_indices


def main():
    started=time.time();base=Path("artifacts/generation_v5");out=Path("artifacts/accuracy_v6");out.mkdir(exist_ok=True)
    book=CompanySchemas(base/"company_rules.json");raw=read_csv("dataset/ERP_원재료_training.csv")
    prepared=[preprocess(r["품목명"],r["규격명"]) for r in raw]
    folds={int(r["source_row"])-2:int(r["fold"]) for r in csv.DictReader((base/"fold_assignments.csv").open(encoding="utf-8-sig"))}
    parsed={i:book.parse(raw[i]["품목코드"]) for i in folds};valid=[i for i in folds if parsed[i]]
    # Audit targets before fitting; CV uses the same per-field eligibility mask.
    vp=[prepared[i] for i in valid];vr=[parsed[i] for i in valid]
    categories=[p["category_id"] for p in vr]
    usable={"_schema":{valid[j] for j in unambiguous_indices([p.key for p in vp],categories)}}
    quality={"_schema":{"candidate_rows":len(valid),"eligible_rows":len(usable["_schema"])}}
    for key,items in AttributeBundle.field_records(vr,book).items():
        indices,values=zip(*items)
        selected=unambiguous_indices([(vp[j].key,categories[j]) for j in indices],list(values))
        usable[key]={valid[indices[j]] for j in selected}
        quality[key]={"candidate_rows":len(items),"eligible_rows":len(usable[key])}
    for value in quality.values():value["excluded_conflicting_input_rows"]=value["candidate_rows"]-value["eligible_rows"]
    write_json(out/"attribute_label_quality.json",{"training_only":True,"fields":quality})
    scores=defaultdict(lambda:defaultdict(lambda:[0,0]))
    for fold in range(3):
        ti=[i for i in valid if folds[i]!=fold];vi=[i for i in valid if folds[i]==fold]
        shared=Features("classification",False).fit([prepared[i] for i in folds if folds[i]!=fold]).text
        model=AttributeBundle().fit([prepared[i] for i in ti],[parsed[i] for i in ti],book,shared)
        x=model.matrices([prepared[i] for i in vi],[parsed[i]["category_id"] for i in vi])
        sx=model.matrices([prepared[i] for i in vi],[None]*len(vi))
        for key,group in [("_schema",model.router),*model.groups.items()]:
            probs=model.distributions(group,sx if key=="_schema" else x)
            for n,i in enumerate(vi):
                if i not in usable[key]:continue
                p=parsed[i];cat=p["category_id"]
                if key=="_schema":
                    truth=cat;field=None
                else:
                    category=book.categories[cat];field=next((f for f in category["fields"] if f["k"]==key),None)
                    if not field:continue
                    slot=field_slot(category,key)
                    parts=book.parts(cat,slot,p["values"][slot])
                    match=next((part["value"] for part in parts or [] if part["key"]==key),None)
                    if match is None:continue
                    truth=match
                for kind,values in probs.items():
                    choices=[(values[n,j],value) for j,value in enumerate(group["classes"])
                             if field is None or not book.field_error(cat,field,value)]
                    guess=max(choices)[1] if choices else None
                    scores[key][kind][0]+=guess==truth;scores[key][kind][1]+=1
        print("ATTRIBUTE_FOLD",fold,"TRAIN",len(ti),"VALID",len(vi),flush=True)
    report={key:{kind:{"correct":v[0],"n":v[1],"accuracy":v[0]/v[1] if v[1] else 0} for kind,v in kinds.items()} for key,kinds in scores.items()}
    selection={key:max(kinds,key=lambda kind:(kinds[kind]["accuracy"],kind=="ensemble")) for key,kinds in report.items()}
    shared=joblib.load(base/"preprocessor.joblib")["text"]
    model=AttributeBundle().fit([prepared[i] for i in valid],[parsed[i] for i in valid],book,shared)
    model.selection=selection;gpu=model.for_cpu()
    joblib.dump(model,out/"attributes.joblib",compress=3)
    result={"training_only":True,"final_evaluation_used_for_selection":False,
            "scope":"Attribute accuracy conditional on the requested manual schema; schema router scored without target schema features",
            "selection":selection,"scores":report,"label_quality":quality,"gpu_training_verified":gpu,"elapsed_seconds":time.time()-started}
    write_json(out/"attribute_validation.json",result)
    print("ATTRIBUTE_DONE",json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
