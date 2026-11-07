"""Bounded candidate-set evaluation with training-only retrieval and part exports."""
import csv,json,math,re,time
from collections import Counter
from pathlib import Path
from .spectrum import SpectrumPredictor,describe_code
from .accuracy_candidates import candidates
from .accuracy_runtime import resistance
from .company_schema import actual_prefix
from .preprocess import preprocess
from .registry import Registry
from .schema import SLOTS
from .train import read_csv,write_json,metrics
from .evaluation_audit import displayed_slot,audit_slots
from .exploratory import effective_codes,coverage_curve

KS=(1,3,5,10)


from .code_metrics import decode_res,semantic_key,numeric_checks


def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export_original(book,directory):
    output=[]
    for file in ["ERP_원재료_training.csv","ERP_원재료_inference.csv"]:
        for i,row in enumerate(read_csv("dataset/"+file)):
            code=row["품목코드"].strip().upper();detail=describe_code(book,code)
            output.append({"원본파일":file,"원본행":i+2,"품목명":row["품목명"],"규격명":row["규격명"],"원본코드":code,
                **{s:detail["values"][s] for s in SLOTS},"규칙상태":book.inspect(code)["status"],
                "미해석구간":"" if detail["semantic_layout_verified"] else code[len(actual_prefix(code)):],
                "세부파트":json.dumps(detail["parts"],ensure_ascii=False)})
    write_csv(directory/"original_code_parts.csv",output)
    return len(output)


def main():
    started=time.time();directory=Path("artifacts/generation_v6")
    registry=Registry.load(directory/"evaluation_registry.json")
    predictor=SpectrumPredictor(directory,registry=registry,candidate_limit=10)
    rows=read_csv("dataset/ERP_원재료_inference.csv");book=predictor.state.book
    corpus_codes={e["code"].upper() for e in registry.entries}
    overlap=sum(r["품목코드"].upper() in corpus_codes for r in rows)
    pps=[preprocess(r["품목명"],r["규격명"]) for r in rows]
    for p in pps:p.candidates=candidates(p.normalized);p._identifier_pool_ready=True
    for slot in SLOTS:
        cfg=predictor.state.manifest["slots"][slot];model=predictor.state.model(slot)
        if slot=="part_number":
            for start in range(0,len(pps),64):
                for p,rank in zip(pps[start:start+64],model.rankings(pps[start:start+64])):p._generation_ranking=rank["ensemble"]
        else:
            probabilities=model.probabilities(pps,cfg["classes"])[cfg["model"]]
            for p,prob in zip(pps,probabilities):
                if not hasattr(p,"_generation_probabilities"):p._generation_probabilities={}
                p._generation_probabilities[slot]=prob
        print("BATCH",slot,flush=True)
    total=len(rows);exact={k:0 for k in KS};manual={k:0 for k in KS};semantic={k:0 for k in KS}
    truths,preds={s:[] for s in SLOTS},{s:[] for s in SLOTS}
    part_hits={s:{k:0 for k in KS} for s in SLOTS};marginal={s:{k:0 for k in KS} for s in SLOTS}
    sizes=Counter();sources=Counter();errors=Counter();mrr=0.;manual_n=0;top_rows=[];long_rows=[];available=0
    check_n=check_ok=truth_conflicts=truth_check_rows=0
    curve_records=[];exploratory_rows=[]
    with (directory/"spectrum_predictions.jsonl").open("w",encoding="utf-8") as f:
        for i,(row,p) in enumerate(zip(rows,pps)):
            by={e["slot"]:e for e in predictor.events(row["품목명"],row["규격명"],prepared=p)}
            assembly=by["assembly"];spectrum=assembly["spectrum"];codes=[c["code"] for c in spectrum]
            truth=row["품목코드"].strip().upper();parsed=book.parse(truth);manual_n+=bool(parsed)
            curve_records.append((truth,effective_codes(assembly)))
            if assembly.get("exploratory",{}).get("active"):
                for c in assembly["exploratory"].get("candidates",[]):
                    exploratory_rows.append({"원본행":i+2,"품목명":row["품목명"],"규격명":row["규격명"],"원본정답":truth,
                        "후보순위":c["rank"],"추정후보코드":c["code"],"출처":c["source"],"정답일치":c["code"]==truth,
                        "코드적용검증":False,"칸별참고후보":json.dumps(assembly["exploratory"].get("head_estimates",{}),ensure_ascii=False)})
            sizes[len(codes)]+=1;available+=bool(codes);sources[assembly.get("code_source") or "no_code"]+=1
            if truth in codes:mrr+=1/(codes.index(truth)+1)
            truth_key=semantic_key(book,truth)
            semantic_keys=[semantic_key(book,c) for c in codes]
            for k in KS:
                hit=truth in codes[:k];exact[k]+=hit;manual[k]+=bool(parsed and hit)
                semantic[k]+=bool(truth_key is not None and truth_key in semantic_keys[:k])
            for s in SLOTS:
                errors[s]+=by[s]["status"]=="error"
                target=actual_prefix(truth) if s=="classification" else parsed["values"][s] if parsed else None
                if not target:continue
                prediction=displayed_slot(assembly,by[s],s)
                truths[s].append(target);preds[s].append(prediction)
                for k in KS:
                    part_hits[s][k]+=target in [c["values"].get(s) for c in spectrum[:k]]
                    marginal[s][k]+=target in [o["value"] for o in assembly["part_spectra"].get(s,[])[:k]]
            checks=numeric_checks(book,codes[0],p,spectrum[0].get("capacity_mode")) if codes else []
            check_n+=len(checks);check_ok+=sum(c["consistent"] for c in checks)
            truth_checks=numeric_checks(book,truth,p);truth_check_rows+=bool(truth_checks)
            truth_conflicts+=any(not c["consistent"] for c in truth_checks)
            top_rows.append({"원본행":i+2,"품목명":row["품목명"],"규격명":row["규격명"],"원본정답":truth,
                "1순위코드":codes[0] if codes else "","원문일치":bool(codes and codes[0]==truth),
                **{s:assembly.get("predicted_values",{}).get(s) for s in SLOTS},
                "후보수":len(codes),"Top5포함":truth in codes[:5],"출처":assembly.get("code_source"),
                "세부파트":json.dumps(assembly.get("parts",[]),ensure_ascii=False)})
            for c in spectrum[:5]:
                long_rows.append({"원본행":i+2,"품목명":row["품목명"],"규격명":row["규격명"],"원본정답":truth,
                    "후보순위":c["rank"],"후보코드":c["code"],**{s:c["values"][s] for s in SLOTS},
                    "상대점수":c["relative_score"],"출처":c["source"],"구간규칙확인":c["semantic_layout_verified"],
                    "세부파트":json.dumps(c["parts"],ensure_ascii=False)})
            f.write(json.dumps({"source_row":i+2,"truth":truth,"assembly":assembly,"numeric_checks":checks},ensure_ascii=False)+"\n")
            if i%250==0:print("EVALUATED",i,flush=True)
    result={"version":"6.0","evaluation_rows":total,"company_layout_rows":manual_n,"operational_master_used":False,
        "evaluation_reference_scope":registry.provenance,"evaluation_code_overlap_in_retrieval_index":overlap,
        "exact_at_k":{str(k):{"correct":exact[k],"rate":exact[k]/total} for k in KS},
        "manual_exact_at_k":{str(k):{"correct":manual[k],"n":manual_n,"rate":manual[k]/manual_n} for k in KS},
        "semantic_at_k":{str(k):{"correct":semantic[k],"n":manual_n,"rate":semantic[k]/manual_n} for k in KS},
        "mean_reciprocal_rank":mrr/total,"average_candidates_at_5":sum(min(k,5)*v for k,v in sizes.items())/total,
        "candidate_size_histogram":dict(sizes),"code_available":available,"code_available_rate":available/total,
        "full_code_correct":exact[1],"full_code_exact_rate":exact[1]/total,"manual_full_code_correct":manual[1],
        "manual_full_code_exact_rate":manual[1]/manual_n,"code_sources":dict(sources),"slot_errors":dict(errors),
        "six_output_rows":total,"model_generated_codes":sources["model_assembled"],
        "slots":{s:{**metrics(truths[s],preds[s],s),
                    "spectrum_at_k":{str(k):part_hits[s][k]/len(truths[s]) if truths[s] else None for k in KS},
                    "marginal_at_k":{str(k):marginal[s][k]/len(truths[s]) if truths[s] else None for k in KS}}
                 for s in SLOTS},
        "slot_audit":audit_slots(book,registry,truths,preds,total),
        "evaluation_scope":"Displayed final values; rules + learned models + training-only retrieval. Detail labels are available on a subset.",
        "numeric_input_consistency":{"checks":check_n,"consistent":check_ok,"rate":check_ok/check_n if check_n else None,
            "truth_rows_with_readable_constraints":truth_check_rows,"truth_rows_conflicting_with_input":truth_conflicts,
            "fields":["resistance","capacitance when unit convention is defined","voltage"],
            "note":"Input consistency is not accuracy; unavailable/ambiguous constraints are unscored."},
        "semantic_definition":"Same validated category and every field equal, except equivalent resistance number encoding.",
        "spectrum_protocol":{"limits":list(KS),"joint_codes":"whole-code candidates","part_coverage":"values in the first K joint code candidates",
                             "marginal":"independent per-head candidates, not a guarantee of joint coverage"},
        "note":"Final set previously inspected: diagnostic only. All selection uses grouped training validation. Candidate scores are not calibrated probabilities.",
        "elapsed_seconds":time.time()-started}
    curves={"max_k":10,"groups":coverage_curve(curve_records,book,10),
            "definition":"Hit@K: original truth code is among first K unique full-code candidates. Candidate choice never uses truth.",
            "candidate_policy":"Use expanded unverified spectrum only when the predicted layout is unsupported; otherwise use validated spectrum.",
            "evaluation_reference_scope":registry.provenance,"operational_master_used":False,
            "evaluation_code_overlap_in_retrieval_index":overlap,
            "note":"Previously inspected diagnostic evaluation; not an untouched blind test."}
    write_json(directory/"candidate_k_metrics.json",curves)
    write_csv(directory/"candidate_k_curve.csv",[
        {"평가범위":name,"K":d["k"],"정답포함행수":d["correct"],"전체행수":d["n"],"정답포함률":d["rate"],"평균후보수":d["average_candidates"]}
        for name,g in curves["groups"].items() for d in g["points"]])
    write_csv(directory/"exploratory_predictions.csv",exploratory_rows)
    write_json(directory/"spectrum_metrics.json",result);write_json(directory/"evaluation_metrics.json",result)
    write_csv(directory/"prediction_parts.csv",top_rows);write_csv(directory/"spectrum_parts_top5.csv",long_rows)
    result["original_parts_rows"]=export_original(book,directory)
    write_json(directory/"spectrum_metrics.json",result)
    print("DONE",json.dumps({k:v for k,v in result.items() if k not in {"slots","evaluation_reference_scope"}},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
