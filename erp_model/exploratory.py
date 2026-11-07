"""Exploratory candidates for unverified layouts, from training evidence only."""
import re
from collections import Counter,defaultdict
from functools import lru_cache
from .accuracy_candidates import ATOM,PHYSICAL,PACKAGE,SIZES
from .company_schema import actual_prefix
from .generation import candidates_for
from .schema import SLOTS


def tokens(text):
    result=[]
    for m in ATOM.finditer(text.upper()):
        raw=m.group()
        if len(raw)<4 or len(raw)>24 or not any(c.isdigit() for c in raw):continue
        if PHYSICAL.fullmatch(raw) or PACKAGE.fullmatch(raw) or raw in SIZES:continue
        if raw.isdigit() and len(raw)<6:continue
        for value in [raw,re.sub(r"[^A-Z0-9]","",raw)]:
            if len(value)>=4 and value not in result:result.append(value)
    return result


def adaptations(entry,p):
    """Replace an observed input identifier at its learned code span; never invent serials."""
    base=entry["code"].upper();prefix=actual_prefix(base);old=tokens(" ".join(entry["normalized_input"]))
    fresh=tokens(" ".join(p.normalized.values()));seen=set()
    for before in old:
        start=base.find(before,len(prefix))
        if start<0:continue
        for after in fresh:
            if before==after:continue
            # Keep the observed code width. Unknown padding/truncation is not assumed.
            if len(before)!=len(after):continue
            code=base[:start]+after+base[start+len(before):]
            if code in seen or len(code)>64 or not re.fullmatch(r"[A-Z0-9-]+",code):continue
            seen.add(code)
            yield {"code":code,"source_code":base,"old_identifier":before,"input_identifier":after,
                   "inferred_span":{"start":start,"end":start+len(after),"value":after}}


@lru_cache(maxsize=8)
def training_statistics(state,corpus):
    allowed=defaultdict(lambda:defaultdict(set));widths=defaultdict(set)
    prior=defaultdict(Counter)
    for e in corpus.entries:
        family=e["component_family"];code=e["code"].upper();prefix=actual_prefix(code)
        allowed[family]["classification"].add(prefix);prior[prefix][code]+=1
        parsed=state.book.parse(code)
        if not parsed:continue
        for slot,value in parsed["values"].items():
            if value:allowed[family][slot].add(value)
        if parsed["values"]["part_number"]:widths[family].add(len(parsed["values"]["part_number"]))
    return {f:dict(v) for f,v in allowed.items()},dict(widths),dict(prior)


def head_estimates(p,predictor):
    family=p.generation_context["component_family"]
    allowed,widths,_=training_statistics(predictor.state,predictor.corpus)
    groups=allowed.get(family,{}) if family!="UNKNOWN" else {}
    out={}
    for slot in SLOTS:
        try:
            values=candidates_for(slot,p,predictor.state)
            if slot=="part_number" and family in widths:
                values=[c for c in values if len(c["value"]) in widths[family]]
            elif slot!="part_number" and family!="UNKNOWN":
                values=[c for c in values if c["value"] in groups.get(slot,set())]
            out[slot]={"candidates":values[:3],"applicability_verified":False,
                       "reason":"같은 입력의 독립 모델 참고 후보입니다. 이 코드 형식의 실제 구간 의미는 검증하지 못했습니다."}
        except Exception as error:
            out[slot]={"candidates":[],"applicability_verified":False,"reason":type(error).__name__+": "+str(error)}
    return out


def exploratory_spectrum(p,predictor,main):
    from .spectrum import describe_code
    book=predictor.state.book
    active=not main.get("display_code") or not book.parse(main["display_code"])
    if not active:return {"active":False,"candidates":[],"head_estimates":{}}
    estimates=head_estimates(p,predictor);pool={};limit=predictor.candidate_limit
    def add(code,score,source,extra=None):
        if not code:return
        code=code.upper()
        value={"code":code,"relative_score":float(score),"source":source,
               "score_type":"relative_rank_score_not_probability","auto_confirmed":False,
               **describe_code(book,code),**(extra or {})}
        # Syntactic parseability does not validate applying a template to this input.
        value["application_verified"]=False
        if code not in pool or pool[code]["relative_score"]<score:pool[code]=value
    for i,c in enumerate(main.get("spectrum",[])):
        add(c["code"],1. if i==0 else .60/(1+i*.15),c["source"])
    matches=predictor.lookup.search(p,limit=40)
    for entry in matches:
        score=min(.75,float(entry.get("retrieval_score",0))*.65)
        add(entry["code"],score,"training_reference",
            {"registered_name":entry["registered_name"],"registered_spec":entry["registered_spec"]})
        if not book.parse(entry["code"]):
            for candidate in adaptations(entry,p):
                add(candidate["code"],score+.12,"training_pattern_adaptation",candidate)
    if not pool:
        # Explicitly weak predictions when the input has no matching reference.
        _,_,prior=training_statistics(predictor.state,predictor.corpus)
        prefixes=[v["value"] for v in estimates["classification"]["candidates"]]
        if not prefixes:prefixes=sorted(prior,key=lambda k:-sum(prior[k].values()))[:3]
        for index,prefix in enumerate(prefixes):
            for code,count in sorted(prior.get(prefix,{}).items(),key=lambda v:(-v[1],v[0]))[:limit]:
                add(code,.01/(1+index),"training_frequency_prior",
                    {"reason":"입력 유사 근거가 없어 학습 분포에서 만든 낮은 근거의 후보입니다."})
    ranked=sorted(pool.values(),key=lambda c:(-c["relative_score"],c["code"]))[:limit]
    for i,c in enumerate(ranked):c["rank"]=i+1
    return {"active":True,"candidates":ranked,"head_estimates":estimates,
            "status":"unverified_estimates","registry_provenance":predictor.corpus.provenance,
            "reason":"구간 규칙 미확인 품목도 후보를 출력합니다. 코드의 적용 가능성과 칸별 의미는 검증되지 않았습니다.",
            "score_note":"후보 순위 점수이며 정답 확률이 아닙니다."}


def effective_codes(assembly):
    exploratory=assembly.get("exploratory",{})
    candidates=exploratory.get("candidates") if exploratory.get("active") else assembly.get("spectrum",[])
    return list(dict.fromkeys(c["code"] for c in candidates or []))


def coverage_curve(records,book,max_k=10):
    """Denominators are fixed; candidate choice is never conditioned on truth."""
    groups={name:{"n":0,"hits":[0]*max_k,"sizes":[0]*max_k,"predicted_rows":0}
            for name in ["all","manual","unmapped"]}
    for truth,codes in records:
        truth=truth.strip().upper();codes=list(dict.fromkeys(codes))
        subset="manual" if book.parse(truth) else "unmapped"
        for name in ["all",subset]:
            g=groups[name];g["n"]+=1;g["predicted_rows"]+=bool(codes)
            for i in range(max_k):
                g["hits"][i]+=truth in codes[:i+1];g["sizes"][i]+=min(i+1,len(codes))
    return {name:{"n":g["n"],"predicted_rows":g["predicted_rows"],
                  "points":[{"k":i+1,"correct":g["hits"][i],"n":g["n"],
                             "rate":g["hits"][i]/g["n"] if g["n"] else None,
                             "average_candidates":g["sizes"][i]/g["n"] if g["n"] else 0}
                            for i in range(max_k)]}
            for name,g in groups.items()}
