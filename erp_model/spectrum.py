"""One shared preprocessing/ensemble pipeline producing a bounded code spectrum."""
from functools import lru_cache
from pathlib import Path
import math
from .accuracy_runtime import AccuracyPredictor
from .existing_lookup import ExistingLookup,FixedRegistry
from .generation import GenerationPredictor,load_state
from .company_schema import actual_prefix
from .preprocess import preprocess
from .schema import SLOTS


@lru_cache(maxsize=2)
def operational_lookup(directory):
    return ExistingLookup(load_state(directory))


def describe_code(book,code):
    """Decode only verified layouts; never invent meanings for legacy serials."""
    parsed=book.parse(code)
    values={s:None for s in SLOTS};parts=[]
    if parsed:
        category=parsed["category_id"];values.update(parsed["values"])
        definitions={f["k"]:f for f in book.categories[category]["fields"]}
        for slot in SLOTS:
            for part in book.parts(category,slot,values[slot]) or []:
                definition=definitions.get(part["key"],{})
                meaning=next((o[1] for o in definition.get("o",[]) if o[0]==part["value"]),None)
                parts.append({**part,"slot":slot,"meaning":meaning,
                              "position_start":part["start"]+1,"position_end":part["end"]})
        for fixed in book.layout(category)[1]:
            parts.append({**fixed,"slot":"fixed","position_start":fixed["start"]+1,"position_end":fixed["end"]})
        return {"values":values,"parts":sorted(parts,key=lambda p:p["position_start"]),
                "category_id":category,"semantic_layout_verified":True}
    prefix=actual_prefix(code);values["classification"]=prefix
    parts=[{"slot":"classification","key":"_major","label":"분류 접두값","value":prefix,
            "start":0,"end":len(prefix),"position_start":1,"position_end":len(prefix)},
           {"slot":"unmapped","key":"_unmapped","label":"구간 의미 미확인","value":code[len(prefix):],
            "start":len(prefix),"end":len(code),"position_start":len(prefix)+1,"position_end":len(code)}]
    return {"values":values,"parts":parts,"category_id":None,"semantic_layout_verified":False}


class SpectrumPredictor(AccuracyPredictor):
    def __init__(self,directory="artifacts/generation_v6",slot=None,registry=None,candidate_limit=5):
        super().__init__(directory,slot,registry,mode="manual")
        self.candidate_limit=max(1,min(10,int(candidate_limit)))
        self.corpus=registry if registry is not None else self.state.registry
        self.lookup=(ExistingLookup(self.state,registry) if registry is not None else
                     operational_lookup(str(self.state.directory.resolve())))

    def prepare(self,name,spec,scheme=None,prepared=None):
        p=prepared if prepared is not None else preprocess(name,spec)
        matches=self.lookup.search(p,limit=8)
        exact=[m for m in matches if m.get("retrieval_basis")=="normalized_name_and_specification"]
        # Exact master rows are evidence in the same pipeline, not another trained model.
        if exact and not scheme:
            codes={m["code"].upper() for m in exact}
            selected=dict(exact[0],code_source="registry_exact" if len(codes)==1 else "registry_candidate")
            self.registry=FixedRegistry([selected],self.corpus.provenance)
            p,context=GenerationPredictor.prepare(self,name,spec,scheme,p)
            p._accuracy_overrides={}
            p.generation_plan["auto"]=len(codes)==1
            if len(codes)>1:
                p.generation_plan["issues"]=["동일한 품목명·규격명에 여러 등록 코드가 있습니다. 후보별 파트를 비교하세요."]
        else:
            p,context=super().prepare(name,spec,scheme,p)
            cat=p.generation_plan.get("category_id")
            identifier_needed=bool(cat and any(s["key"] in {"pn","sub","model","name"} for s in self.state.book.layout(cat)[0]["part_number"]))
            missing_identifier=identifier_needed and not p.generation_plan.get("code") and not any(
                self.state.book.valid_slot(cat,"part_number",c["value"]) for c in p.candidates)
            if (not context["generation_supported"] or missing_identifier) and matches and not scheme:
                selected=dict(matches[0],code_source="registry_candidate")
                self.registry=FixedRegistry([selected],self.corpus.provenance)
                p,context=GenerationPredictor.prepare(self,name,spec,scheme,p)
                p._accuracy_overrides={}
        p._spectrum_lookup=matches
        strict_matches=self.corpus.search(p,context["component_family"])
        context.update(prediction_mode="spectrum",registry_candidates=strict_matches,lookup_candidates=matches,
                       spectrum_limit=self.candidate_limit,registry_provenance=self.corpus.provenance)
        return p,context

    def predict_slot(self,slot,p):
        try:
            return super().predict_slot(slot,p)
        except Exception as error:
            return {"slot":slot,"value":None,"predicted_value":None,"status":"error",
                    "candidates":[],"top_candidates":[],"fields":[],"evidence":[],
                    "reason":type(error).__name__+": "+str(error),
                    "code_scheme_id":p.generation_context["code_scheme_id"]}

    def slot_spectrum(self,result,p):
        slot=result["slot"];book=self.state.book;cat=p.generation_plan["category_id"]
        primary=result.get("predicted_value")
        if primary is None or not cat or p.generation_plan["source"]!="model_assembled":
            return [{"value":primary,"relative_score":1.,"basis":"selected_value"}] if primary is not None else []
        spans=book.layout(cat)[0][slot]
        if not spans:return [{"value":"","relative_score":1.,"basis":"not_applicable"}]
        primary_parts={v["key"]:v for v in result.get("fields",[])}
        locked={key:v["value"] for key,v in primary_parts.items()
                if v.get("source") in {"input_rule","schema","manufacturer_conflict"}}
        options={primary:{"value":primary,"relative_score":1.,"basis":"selected_value"}}
        def add(value,score,basis):
            pieces=book.parts(cat,slot,value)
            if not pieces:return
            value="".join(locked.get(part["key"],part["value"]) for part in pieces)
            if not book.valid_slot(cat,slot,value):return
            if value not in options or options[value]["relative_score"]<score:
                options[value]={"value":value,"relative_score":float(score),"basis":basis}
        for rank,c in enumerate(result.get("candidates",[])[:8]):
            if c["value"]!=primary:add(c["value"],.78/(1+rank*.25),"independent_head")
        definitions={f["k"]:f for f in book.categories[cat]["fields"]}
        for span in spans:
            key=span["key"]
            if key in locked or key not in definitions or not self.use_attributes:continue
            selected=self.attribute_report["selection"].get(key)
            if self.attribute_report["scores"].get(key,{}).get(selected,{}).get("accuracy",0)<.6:continue
            alternatives=self.attributes.field_candidates(p,cat,definitions[key],book) or []
            for rank,(prob,value) in enumerate(alternatives[:3]):
                pieces=[value if s["key"]==key else primary_parts.get(s["key"],{}).get("value","")
                        for s in spans]
                add("".join(pieces),.80/(1+rank*.3),"attribute_head")
        return [options[primary]]+sorted([v for k,v in options.items() if k!=primary],
                                         key=lambda v:(-v["relative_score"],v["value"]))[:4]

    def assemble(self,results,p):
        main=super().assemble(results,p);book=self.state.book
        spectra={r["slot"]:self.slot_spectrum(r,p) for r in results}
        candidates=[];seen=set()
        def add(code,score,source,metadata=None):
            if not code:return
            code=code.upper()
            if code in seen:return
            detail=describe_code(book,code)
            from .code_metrics import numeric_checks
            checks=numeric_checks(book,code,p,p.generation_context.get("capacity_mode") if source=="model_assembled" else None)
            detail["input_numeric_checks"]=checks
            detail["input_numeric_consistent"]=all(c["consistent"] for c in checks) if checks else None
            # Newly composed codes must pass every manual field table.
            if source=="model_assembled" and not detail["semantic_layout_verified"]:return
            seen.add(code)
            candidates.append({"code":code,"relative_score":float(score),"source":source,
                               "score_type":"relative_rank_score_not_probability",
                               "capacity_mode":p.generation_context.get("capacity_mode") if source=="model_assembled" else None,
                               **detail,**(metadata or {})})
        lookup=p._spectrum_lookup
        exact=[m for m in lookup if m.get("retrieval_basis")=="normalized_name_and_specification"]
        if exact:
            for m in exact:
                add(m["code"],1.,"registry_exact",{"registered_name":m["registered_name"],
                    "registered_spec":m["registered_spec"],"auto_confirmed":len({x["code"] for x in exact})==1})
        add(main.get("display_code"),1.,main.get("code_source") or "model_assembled")
        cat=p.generation_plan["category_id"]
        if cat and p.generation_plan["source"]=="model_assembled" and main.get("display_code"):
            beam=[(0.,{})]
            for slot in SLOTS:
                options=spectra[slot] or ([{"value":"","relative_score":1.}] if not book.layout(cat)[0][slot] else [])
                expanded=[(score+math.log(max(option["relative_score"],1e-9)),{**values,slot:option["value"]})
                          for score,values in beam for option in options]
                beam=sorted(expanded,key=lambda row:-row[0])[:80]
            for score,values in beam:
                add(book.assemble(cat,values),math.exp(score),"model_assembled")
        # Retrieval supplies additional hypotheses in the same ranked result.
        # It never silently replaces a generated code based on text similarity alone.
        for m in lookup:
            add(m["code"],min(.65,float(m.get("retrieval_score",0))*.6),"registry_candidate",
                {"registered_name":m["registered_name"],"registered_spec":m["registered_spec"],
                 "retrieval_basis":m.get("retrieval_basis"),"auto_confirmed":False})
        candidates=candidates[:self.candidate_limit]
        for i,c in enumerate(candidates):c["rank"]=i+1
        if candidates:
            first=candidates[0]
            main.update(display_code=first["code"],predicted_code=first["code"],code_length=len(first["code"]),
                        parts=first["parts"],predicted_values=first["values"],code_source=first["source"])
        main.update(spectrum=candidates,part_spectra=spectra,spectrum_size=len(candidates),
                    prediction_mode="spectrum",score_note="후보의 상대 순위 점수이며 정답 확률이 아닙니다.",
                    registry_provenance=self.corpus.provenance)
        from .exploratory import exploratory_spectrum
        try:
            main["exploratory"]=exploratory_spectrum(p,self,main)
        except Exception as error:
            main["exploratory"]={"active":True,"status":"error","candidates":[],"head_estimates":{},
                                 "reason":type(error).__name__+": "+str(error)}
        return main
