"""Conditional attribute estimators; each of the six public slots owns its fields."""
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize as norm_matrix
from .models import classifier
from .label_quality import unambiguous_indices


def raw_features(prepared, category=None):
    result={}
    for slot, values in prepared.structured.items():
        for key,value in values.items():result[slot+":"+key]=value
    if category:result["schema:"+category]=1
    text=" ".join(prepared.normalized.values())
    for size in re.findall(r"(?<!\d)(0603|1005|1608|2012|3216|3225|3528|6032|7343)(?!\d)",text):
        result["size:"+size]=1
    dimensions=re.search(r"(?<![A-Z0-9])(?:[MØΦ])?(\d+(?:\.\d+)?)\s*[*X×]\s*(\d+(?:\.\d+)?)",text)
    if dimensions:
        result["dimension:diameter"]=float(dimensions[1]);result["dimension:length"]=float(dimensions[2])
    for token in re.findall(r"(?<![A-Z0-9])([A-Z]{2,4})(?![A-Z0-9])",text):
        result["series:"+token]=1
    return result


class AttributeBundle:
    def __init__(self):
        self.text=None;self.dv=DictVectorizer(dtype=np.float32);self.groups={};self.router=None;self.selection={}

    def matrices(self, prepared, categories):
        base=hstack([v.transform([p.normalized[col] for p in prepared])
                     for v,col in zip(self.text,["품목명","규격명"])],format="csr",dtype=np.float32)
        extra=self.dv.transform([raw_features(p,c) for p,c in zip(prepared,categories)])
        return hstack([base,extra],format="csr",dtype=np.float32)

    def fit(self, prepared, parsed, book, shared):
        self.text=shared
        cats=[p["category_id"] for p in parsed]
        self.dv.fit([raw_features(p,c) for p,c in zip(prepared,cats)]+[raw_features(p) for p in prepared])
        x=self.matrices(prepared,cats);router_x=self.matrices(prepared,[None]*len(prepared))
        self.router=self.fit_group(router_x,cats,[p.key for p in prepared])
        if self.router is None:raise ValueError("No unambiguous schema-router labels")
        records=self.field_records(parsed,book)
        for key,items in records.items():
            indices,values=zip(*items)
            group=self.fit_group(x[list(indices)],list(values),[(prepared[i].key,cats[i]) for i in indices])
            if group is not None:self.groups[key]=group
        return self

    @staticmethod
    def field_records(parsed,book):
        records={}
        for i,record in enumerate(parsed):
            category=book.categories[record["category_id"]]
            for field in category["fields"]:
                if field["t"] not in {"sel"} and not (field["k"]=="ver" and field["n"]==1):continue
                slot=__import__("erp_model.company_schema",fromlist=["field_slot"]).field_slot(category,field["k"])
                if slot in record.get('exclude_slots',[]):continue
                if slot=="part_number":continue
                parts=book.parts(record["category_id"],slot,record["values"][slot]) or []
                value=next(p["value"] for p in parts if p["key"]==field["k"])
                records.setdefault(field["k"],[]).append((i,value))
        return records

    @staticmethod
    def fit_group(x, labels, keys=None):
        before=len(labels)
        if keys is not None:
            eligible=unambiguous_indices(keys,labels)
            x=x[eligible];labels=[labels[i] for i in eligible]
        if not labels:return None
        classes=sorted(set(labels));group={"classes":classes,"models":{},"rows":len(labels),
            "candidate_rows":before,"excluded_conflicting_input_rows":before-len(labels)}
        if len(classes)>1:
            y=np.array([classes.index(v) for v in labels])
            for kind in ["xgb","lgb"]:group["models"][kind]=classifier(kind).fit(x,y)
        return group

    @staticmethod
    def distributions(group,x):
        classes=group["classes"]
        if not group["models"]:return {k:np.ones((x.shape[0],1)) for k in ["xgb","lgb","ensemble"]}
        out={k:model.predict_proba(x) for k,model in group["models"].items()}
        out["ensemble"]=(out["xgb"]+out["lgb"])/2
        return out

    def schema(self,p,allowed=None):
        x=self.matrices([p],[None])
        proba=self.distributions(self.router,x)[self.selection.get("_schema","ensemble")][0]
        options=[(float(proba[i]),c) for i,c in enumerate(self.router["classes"]) if not allowed or c in allowed]
        options.sort(reverse=True)
        return options[0] if options else (0,None)

    def field(self,p,category,field,book):
        options=self.field_candidates(p,category,field,book)
        return options[0] if options else None

    def field_candidates(self,p,category,field,book):
        key=field["k"]
        if key not in self.groups:return None
        cache=getattr(p,"_attribute_matrix",{})
        cache_key=(id(self),category)
        if cache_key not in cache:
            cache[cache_key]=self.matrices([p],[category]);p._attribute_matrix=cache
        group=self.groups[key];kind=self.selection.get(key,"ensemble")
        probs=self.distributions(group,cache[cache_key])[kind][0]
        options=[(float(probs[i]),value) for i,value in enumerate(group["classes"])
                 if not book.field_error(category,field,value)]
        return sorted(options,reverse=True)

    def for_cpu(self):
        gpu=False
        import json
        for group in [self.router,*self.groups.values()]:
            for model in group["models"].values():
                if model.__class__.__module__.startswith("xgboost"):
                    gpu |= json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"].startswith("cuda")
                    model.set_params(device="cpu",n_jobs=2)
                elif hasattr(model,"set_params"):model.set_params(n_jobs=2)
        return gpu
