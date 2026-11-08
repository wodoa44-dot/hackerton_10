"""Existing-code retrieval, evaluated as lookup over a known operational master."""
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
from erp_model.encoders.candidates import ATOM,PHYSICAL,PACKAGE
from erp_model.encoders.rules.material_schema import silver_family


def sku_tokens(text):
    return {re.sub(r"[^A-Z0-9]","",m.group()) for m in ATOM.finditer(text)
            if len(m.group())>=5 and any(c.isalpha() for c in m.group()) and any(c.isdigit() for c in m.group())
            and not PHYSICAL.fullmatch(m.group()) and not PACKAGE.fullmatch(m.group())}


class ExistingLookup:
    def __init__(self,state,registry=None):
        self.state=state;self.registry=registry if registry is not None else state.registry;self.entries=self.registry.entries
        self.matrix=normalize(hstack([v.transform([e["normalized_input"][col] for e in self.entries])
                                     for v,col in zip(state.shared,[0,1])],format="csr"))
        self.skus=[sku_tokens(e["normalized_input"][1]) for e in self.entries]

    def search(self,p,limit=8):
        family=silver_family(p)[0] or "UNKNOWN"
        exact=self.registry.search(p,family)
        if exact and exact[0]["code_source"]=="registry_exact":
            return [dict(e,retrieval_score=1.,retrieval_basis="normalized_name_and_specification",strong_match=True) for e in exact]
        query=normalize(self.state.text_features(p))
        scores=(query@self.matrix.T).toarray().ravel()
        q_skus=sku_tokens(p.normalized["규격명"])
        pool=np.argsort(-scores,kind="stable")[:100]
        ranked=[]
        for i in pool:
            if scores[i]<.18:continue
            e=self.entries[i];agreement=bool(q_skus & self.skus[i])
            score=float(scores[i])
            if agreement:score+=.18
            elif q_skus and self.skus[i]:score*=.65
            if family!="UNKNOWN" and e["component_family"] not in {family,"UNKNOWN"}:score*=.6
            ranked.append((score,int(i),agreement))
        ranked.sort(reverse=True)
        results=[]
        for score,i,agreement in ranked[:limit]:
            e=self.entries[i];same=p.key[0]==e["normalized_input"][0]
            results.append({**e,"input_name":p.raw["품목명"],"input_spec":p.raw["규격명"],
                "name_differs":not same,"normalized_name_matches":same,
                "code_source":"registry_candidate","auto_confirmed":False,"status":"review",
                "retrieval_score":score,"strong_match":agreement and score>=.65,
                "retrieval_basis":"manufacturer_part_agreement" if agreement else "name_and_spec_text_similarity",
                "match_basis":["part number agrees"] if agreement else ["text similarity"],
                "reason":"Existing registered code candidate; review the original name and specification"})
        return results


class FixedRegistry:
    def __init__(self,results,provenance):
        self.results=results;self.provenance=provenance
    def search(self,p,family):
        return self.results
