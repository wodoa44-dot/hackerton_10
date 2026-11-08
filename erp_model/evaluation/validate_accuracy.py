"""Select refinement policies on a held-out training fold, never the final CSV."""
import copy,csv,json,shutil,time
from pathlib import Path
from collections import defaultdict
import joblib,numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from erp_model.encoders.features import Features
from erp_model.models.tabular import ClassifierBundle,RankBundle
from erp_model.prediction.generation import GenerationPredictor
from erp_model.prediction.accuracy_runtime import AccuracyPredictor
from erp_model.models.attributes import AttributeBundle
from erp_model.encoders.candidates import candidates
from erp_model.encoders.preprocess import preprocess,identifier_candidates
from erp_model.encoders.rules.company_schema import CompanySchemas,actual_prefix
from erp_model.models.registry import Registry
from erp_model.shared.schema import SLOTS
from erp_model.training.train import read_csv,write_json
from erp_model.training.train_generation import cpu_inference


def main():
    started=time.time();base=Path("artifacts/generation_v5");output=Path("artifacts/accuracy_v6")
    bdir=output/"validation_base";ndir=output/"validation_refined";bdir.mkdir(exist_ok=True);ndir.mkdir(exist_ok=True)
    book=CompanySchemas(base/"company_rules.json");data=read_csv("dataset/ERP_원재료_training.csv")
    p=[preprocess(r["품목명"],r["규격명"]) for r in data]
    folds={int(r["source_row"])-2:int(r["fold"]) for r in csv.DictReader((base/"fold_assignments.csv").open(encoding="utf-8-sig"))}
    ti=[i for i in folds if folds[i]!=0];vi=[i for i in folds if folds[i]==0 and book.parse(data[i]["품목코드"])]
    parsed={i:book.parse(data[i]["품목코드"]) for i in folds}
    shared=Features("classification",False).fit([p[i] for i in ti]).text
    manifest=json.loads((base/"manifest.json").read_text());manifest["version"]="5.0"
    for slot in SLOTS:
        eligible=[i for i in ti if slot=="classification" or parsed[i] and parsed[i]["values"][slot] and
                  (slot!="part_number" or any(v["key"] in {"pn","sub","model","name"} for v in book.layout(parsed[i]["category_id"])[0][slot]))]
        labels=lambda i:actual_prefix(data[i]["품목코드"]) if slot=="classification" else parsed[i]["values"][slot]
        bykey=defaultdict(set)
        for i in eligible:bykey[p[i].key].add(labels(i))
        indices=[i for i in eligible if len(bykey[p[i].key])==1]
        if slot=="part_number":
            for i in indices:p[i].candidates=identifier_candidates(p[i].normalized)
            model=RankBundle(True).fit([p[i] for i in indices],[labels(i) for i in indices],shared)
        else:model=ClassifierBundle(slot,True).fit([p[i] for i in indices],[labels(i) for i in indices],shared)
        cpu_inference(model);joblib.dump(model,bdir/(slot+".joblib"),compress=3)
        if slot!="part_number":manifest["slots"][slot]["classes"]=model.classes
        print("VALIDATION_BASE_FIT",slot,len(indices),flush=True)
    joblib.dump({"text":shared},bdir/"preprocessor.joblib",compress=3)
    refs=[data[i] for i in ti];matrix=normalize(hstack([v.transform([p[i].normalized[c] for i in ti]) for v,c in zip(shared,["품목명","규격명"])],format="csr"))
    joblib.dump({"rows":refs,"matrix":matrix},bdir/"reference.joblib",compress=3)
    Registry.build(refs,{"scope":"training_fold_only"}).save(bdir/"operational_master.json")
    write_json(bdir/"manifest.json",manifest);write_json(bdir/"scheme_values.json",{})
    for name in ["company_rules.json","manual_source.json"]:shutil.copy2(base/name,bdir/name)
    for path in bdir.iterdir():shutil.copy2(path,ndir/path.name)
    attr=AttributeBundle().fit([p[i] for i in ti if parsed[i]],[parsed[i] for i in ti if parsed[i]],book,shared)
    report=json.loads((output/"attribute_validation.json").read_text());attr.selection=report["selection"];attr.for_cpu()
    joblib.dump(attr,ndir/"attributes.joblib",compress=3);shutil.copy2(output/"attribute_validation.json",ndir/"attribute_validation.json")
    ids=[i for i in ti if parsed[i] and parsed[i]["values"]["part_number"] and
         any(v["key"] in {"pn","sub","model","name"} for v in book.layout(parsed[i]["category_id"])[0]["part_number"])]
    bykey=defaultdict(set)
    for i in ids:bykey[p[i].key].add(parsed[i]["values"]["part_number"])
    ids=[i for i in ids if len(bykey[p[i].key])==1]
    for i in ids:p[i].candidates=candidates(p[i].normalized)
    rank=RankBundle(True).fit([p[i] for i in ids],[parsed[i]["values"]["part_number"] for i in ids],shared);cpu_inference(rank)
    joblib.dump(rank,ndir/"part_number.joblib",compress=3)
    manifest["version"]="6.0";write_json(ndir/"manifest.json",manifest)
    empty=Registry.build([],{"scope":"no_operational_master"})
    runs=[("baseline",GenerationPredictor(bdir,registry=Registry.load(bdir/"operational_master.json")))]
    for name,attrs,router in [("rank_and_rules",False,False),("with_attributes",True,False),("with_router",True,True)]:
        predictor=AccuracyPredictor(ndir,registry=empty,mode="manual")
        predictor.use_attributes=attrs;predictor.use_router=router;runs.append((name,predictor))
    scores={}
    for name,predictor in runs:
        rows=[preprocess(data[i]["품목명"],data[i]["규격명"]) for i in vi]
        for pp in rows:
            pp.candidates=identifier_candidates(pp.normalized) if name=="baseline" else candidates(pp.normalized);pp._identifier_pool_ready=True
        for slot in SLOTS:
            model=predictor.state.model(slot);cfg=predictor.state.manifest["slots"][slot]
            if slot=="part_number":
                for start in range(0,len(rows),64):
                    for pp,ranked in zip(rows[start:start+64],model.rankings(rows[start:start+64])):pp._generation_ranking=ranked["ensemble"]
            else:
                prob=model.probabilities(rows,cfg["classes"])[cfg["model"]]
                for pp,value in zip(rows,prob):
                    if not hasattr(pp,"_generation_probabilities"):pp._generation_probabilities={}
                    pp._generation_probabilities[slot]=value
        correct=0;slots={s:0 for s in SLOTS};den={s:0 for s in SLOTS}
        for index,(i,pp) in enumerate(zip(vi,rows)):
            events={e["slot"]:e for e in predictor.events(data[i]["품목명"],data[i]["규격명"],prepared=pp)}
            correct+=events["assembly"]["display_code"]==data[i]["품목코드"].upper()
            for s,value in parsed[i]["values"].items():
                if value:slots[s]+=events[s]["predicted_value"]==value;den[s]+=1
        scores[name]={"n":len(vi),"correct":correct,"accuracy":correct/len(vi),"slot_correct":slots,"slot_n":den}
        print("TRAINING_PIPELINE_VALIDATION",name,scores[name],flush=True)
    selected=max([name for name in scores if name!="baseline"],key=lambda name:scores[name]["correct"])
    result={"training_only":True,"final_evaluation_used_for_selection":False,"validation_fold":0,
            "scores":scores,"selected":selected,"improved":scores[selected]["correct"]>scores["baseline"]["correct"],
            "elapsed_seconds":time.time()-started}
    write_json(output/"pipeline_validation.json",result);print("PIPELINE_DONE",json.dumps(result),flush=True)


if __name__=="__main__":main()
