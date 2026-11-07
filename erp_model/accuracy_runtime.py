"""Internal six-head refinement used by the unified spectrum predictor."""
import json,re,threading
from functools import lru_cache
from pathlib import Path
import joblib
from .generation import GenerationPredictor,candidates_for
from .accuracy_candidates import candidates
from .company_schema import actual_prefix,field_slot
from .existing_lookup import ExistingLookup,FixedRegistry
from .material_schema import silver_family
from .preprocess import preprocess
from .schema import SLOTS


def resistance(text):
    m=re.search(r"(?<![A-Z0-9.])(\d+(?:\.\d+)?)\s*(K|M|R|OHM)(?=[BDFGJK](?![A-Z])|(?:OHM)?(?![A-Z0-9]))",text)
    if m:return float(m[1])*{"K":1e3,"M":1e6,"R":1,"OHM":1}[m[2]]
    m=re.search(r"(?:^|[ ,])([BDFGJK])(?:급)?\s*[/ ]*\s*(\d+(?:\.\d+)?)(?![A-Z0-9.])",text)
    if m:return float(m[2])
    m=re.search(r"(?:^|[ ,])(\d+(?:\.\d+)?)\s+([BDFGJK])(?:$|[ ,])",text)
    return float(m[1]) if m else None


def resistance_code(value,width):
    if value is None or value<0:return None
    if value==0:return "0"*width
    if width==3:
        if value<10:return f"{value:.1f}".replace(".","R")
        exponent=max(0,len(str(int(value)))-2);return str(round(value/10**exponent))+str(exponent)
    if value<.01:
        raw=f"{value*1000:g}".replace(".","M")
        return (raw if "M" in raw else raw+"M").ljust(4,"0")[:4]
    if value<10 and value!=int(value):return f"{value:.2f}".replace(".","R")
    if value<100 and value!=int(value):return f"{value:.1f}".replace(".","R")
    for exponent in range(10):
        mantissa=value/10**exponent
        if mantissa<1000 and abs(mantissa-round(mantissa))<1e-7:return str(round(mantissa)).zfill(3)+str(exponent)
    return None


def field_overrides(p,category,book):
    text=" ".join(p.normalized.values());out={}
    if category.startswith("R"):
        f=next((f for f in book.categories[category]["fields"] if f["k"]=="res"),None)
        if f:
            code=resistance_code(resistance(text),f["n"])
            if code and not book.field_error(category,f,code):out["res"]=code
        tol=re.search(r"(?<![A-Z])([BDFGJK])급",text)
        if tol:out["tol"]=tol[1]
    if category=="H":
        tip=re.search(r"([1-4])\s*종",text)
        if tip:out["tip"]=tip[1]
    if category in {"CA","CB","CC"}:
        # Vendor capacitor labels encode property/voltage/capacitance/tolerance/size.
        m=re.search(r"([A-Z])([0-9][A-Z])([0-9]{3}|[0-9]R[0-9])([BCDFGJKMZ])([0-9]{2})(?:[A-Z0-9]|$)",text)
        if m:
            for key,value in zip(["tc","volt","cap","tol","dia" if category=="CA" else "shape"],m.groups()):
                f=next((f for f in book.categories[category]["fields"] if f["k"]==key),None)
                if f and not book.field_error(category,f,value):out[key]=value
    return out


@lru_cache(maxsize=2)
def refinement(directory):
    d=Path(directory)
    model=joblib.load(d/"attributes.joblib")
    report=json.loads((d/"attribute_validation.json").read_text())
    return model,report


@lru_cache(maxsize=2)
def lookup_for(directory):
    from .generation import load_state
    return ExistingLookup(load_state(directory))


class AccuracyPredictor(GenerationPredictor):
    def __init__(self,directory="artifacts/generation_v6",slot=None,registry=None,mode="auto"):
        super().__init__(directory,slot,registry)
        self.mode=mode
        self.allow_operational_lookup=registry is None
        self.original_registry=self.registry
        self.attributes,self.attribute_report=refinement(str(self.state.directory.resolve()))
        self.attributes.text=self.state.shared
        policy=self.state.manifest.get("refinement_policy",{})
        self.use_attributes=policy.get("use_attributes",True)
        self.use_router=policy.get("use_router",True)

    def prepare(self,name,spec,scheme=None,prepared=None):
        p=prepared if prepared is not None else preprocess(name,spec)
        p.candidates=candidates(p.normalized);p._identifier_pool_ready=True
        lookup=[]
        if self.allow_operational_lookup and self.mode!="manual":
            lookup=lookup_for(str(self.state.directory.resolve())).search(p)
        exact=lookup and lookup[0].get("retrieval_basis")=="normalized_name_and_specification"
        strong=lookup and lookup[0].get("strong_match") and (len(lookup)==1 or lookup[0]["retrieval_score"]-lookup[1]["retrieval_score"]>=.08)
        choose_existing=bool(lookup and (self.mode=="existing" or self.mode=="auto" and (exact or strong)))
        if choose_existing:
            matches=lookup if exact else lookup[:1]
            self.registry=FixedRegistry(matches,{"scope":"operational_master_lookup","model_prediction":False})
            p,context=super().prepare(name,spec,scheme,p)
        else:
            self.registry=FixedRegistry([],{"scope":"manual_generation_without_operational_lookup"})
            family=silver_family(p)[0] or "UNKNOWN"
            name=p.normalized["품목명"]
            if family=="UNKNOWN":
                if re.search(r"WAFER|HEADER|HOUSING|CONNECT",name):family="CONNECTOR"
                elif re.search(r"ENCORDER|ENCODER",name):family="SWITCH"
                elif re.search(r"PHOTO.?COUPLER|PHOTO.?THYRISTOR|REGULATOR|MICOM|MCU",name):family="IC"
                elif re.search(r"E/?C.?CAP|E.?CAP",name):family="CAPACITOR"
            allowed=self.state.book.allowed_categories(family)
            score,schema=self.attributes.schema(p,allowed) if allowed and self.use_router else (0,None)
            chosen=scheme or ("COMPANY_"+schema if schema and score>=.4 else None)
            p,context=super().prepare(name,spec,chosen,p)
            if p.generation_plan["source"]=="reference_candidate":
                p.generation_plan.update(code=None,category_id=None,source=None,
                    issues=["해당 품목의 신규 코드 규칙을 확인해야 합니다. 등록 코드 찾기에서 기존 품목 후보를 확인하세요."])
                context.update(generation_supported=False,slot_layout={},expected_code_length=None)
        context["prediction_mode"]="existing" if choose_existing else "manual"
        context["requested_mode"]=self.mode
        context["lookup_candidates"]=lookup
        if not choose_existing:context["registry_candidates"]=lookup
        p._accuracy_overrides=field_overrides(p,p.generation_plan["category_id"],self.state.book) if p.generation_plan["category_id"] else {}
        return p,context

    def predict_slot(self,slot,p):
        if p.generation_plan.get("code") and p.generation_plan["source"] in {"registry_exact","registry_candidate","reference_candidate"}:
            code=p.generation_plan["code"];parsed=self.state.book.parse(code)
            value=actual_prefix(code) if slot=="classification" else parsed["values"][slot] if parsed else None
            status=("confirmed" if p.generation_plan["auto"] else "review") if value else "not_applicable" if parsed else "unsupported_schema"
            fields=self.state.book.parts(parsed["category_id"],slot,value) if parsed and value else []
            return {"slot":slot,"value":value if status=="confirmed" else None,"predicted_value":value or None,
                "status":status,"candidates":[{"value":value,"score":None,"score_type":"registered_field"}] if value else [],
                "top_candidates":[],"fields":fields or [],"evidence":[],
                "reason":"등록 코드의 실제 값을 구간별로 읽었습니다." if value else "구간 의미가 미확인되었거나 해당 구간이 없습니다.",
                "code_scheme_id":p.generation_context["code_scheme_id"]}
        if slot=="part_number" and not hasattr(p,"_generation_ranking"):
            p._generation_ranking=self.state.model(slot).rankings([p])[0]["ensemble"]
        result=super().predict_slot(slot,p)
        plan=p.generation_plan;category=plan["category_id"];book=self.state.book
        if result["status"] in {"error","unsupported_schema","not_applicable"} or not category or plan["source"]!="model_assembled":
            return result
        if slot=="classification":return result
        if result.get("requires_input_review"):return result
        parts=result.get("fields",[])
        current={part["key"]:part["value"] for part in parts}
        fields={f["k"]:f for f in book.categories[category]["fields"]}
        overrides=p._accuracy_overrides
        if slot=="part_number":
            identifier=any(span["key"] in {"pn","sub","model","name"} for span in book.layout(category)[0][slot])
            preserve=any(plan["sources"].get(span["key"]) in {"rule","manual_transform"} for span in book.layout(category)[0][slot])
            if identifier and not preserve:
                ranked=[c for c in candidates_for(slot,p,self.state) if book.valid_slot(category,slot,c["value"])]
                if ranked:
                    current={v["key"]:v["value"] for v in book.parts(category,slot,ranked[0]["value"])}
                    result["candidates"]=ranked[:8]
        output=[]
        for span in book.layout(category)[0][slot]:
            key=span["key"];value=current.get(key);source="independent_slot_model";prob=None
            if key=="_subtype":value=book.categories[category]["prefix"][len(actual_prefix(book.categories[category]["prefix"])):];source="schema"
            elif key in overrides:value=overrides[key];source="input_rule"
            elif key in fields:
                field=fields[key];explicit=plan["sources"].get(key) in {"rule","manual_transform"}
                # Several voltage options can share a magnitude (AC/DC); infer only when not explicit.
                if explicit:source="input_rule"
                else:
                    selected=self.attribute_report["selection"].get(key)
                    stats=self.attribute_report["scores"].get(key,{}).get(selected,{})
                    guess=self.attributes.field(p,category,field,book) if self.use_attributes and stats.get("accuracy",0)>=.6 else None
                    if guess:prob,value=guess;source="attribute_model"
                if key=="vend" and result.get("reason")=="conflicting_manufacturer_evidence":
                    value="-";source="manufacturer_conflict"
            if value is None:return result
            output.append({**span,"value":value,"source":source,"probability":prob})
        combined="".join(part["value"] for part in output)
        if book.valid_slot(category,slot,combined):
            result.update(predicted_value=combined,fields=output)
            if not result["candidates"] or result["candidates"][0]["value"]!=combined:
                result["candidates"]=[{"value":combined,"score":None,"score_type":"independent_attribute_composition"}]+result["candidates"][:7]
            result["top_candidates"]=result["candidates"]
        return result

    def assemble(self,results,p):
        result=super().assemble(results,p)
        code=result["display_code"]
        parts=[]
        if code:
            parsed=self.state.book.parse(code)
            if parsed:
                for slot in SLOTS:
                    for part in self.state.book.parts(parsed["category_id"],slot,parsed["values"][slot]) or []:
                        parts.append({**part,"slot":slot,"position_start":part["start"]+1,"position_end":part["end"]})
                for fixed in self.state.book.layout(parsed["category_id"])[1]:
                    parts.append({**fixed,"slot":"fixed","position_start":fixed["start"]+1,"position_end":fixed["end"]})
            else:
                prefix=actual_prefix(code)
                parts=[{"slot":"classification","label":"분류 접두값","value":prefix,"position_start":1,"position_end":len(prefix)},
                       {"slot":"unmapped","label":"구간 의미 미확인","value":code[len(prefix):],"position_start":len(prefix)+1,"position_end":len(code)}]
        result["parts"]=sorted(parts,key=lambda x:x["position_start"])
        result["prediction_mode"]=p.generation_context["prediction_mode"]
        return result
