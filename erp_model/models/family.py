import numpy as np
from sklearn.linear_model import LogisticRegression
from erp_model.encoders.features import Features
from erp_model.encoders.rules.material_schema import silver_family,family_evidence

class FamilyModel:
    def fit(self,rows,labels):
        self.feature=Features('classification',False).fit(rows)
        self.model=LogisticRegression(max_iter=500,C=3,class_weight='balanced',random_state=42).fit(self.feature.transform(rows),labels)
        return self
    def probabilities(self,rows):return self.model.predict_proba(self.feature.transform(rows))
    def identify(self,p):
        family,evidence=silver_family(p)
        if family:return {'value':family,'status':'confirmed','source':'name_alias','evidence':evidence,'candidates':[{'value':family,'score':1.,'score_type':'rule_match_not_probability'}]}
        conflicts={e['family'] for e in evidence}
        if len(conflicts)>1:return {'value':'UNKNOWN','status':'review','source':'conflicting_name_aliases','evidence':evidence,'candidates':[{'value':f,'score':None} for f in sorted(conflicts)]}
        ps=self.probabilities([p])[0];order=np.argsort(-ps)
        candidates=[{'value':str(self.model.classes_[i]),'score':float(ps[i]),'score_type':'probability_uncalibrated_weak_labels'} for i in order[:5]]
        certain=ps[order[0]]>=.8 and (ps[order[0]]-ps[order[1]]>=.2 if len(order)>1 else True)
        return {'value':candidates[0]['value'] if certain else 'UNKNOWN','status':'review','source':'model_from_name_silver_labels','evidence':[], 'candidates':candidates}
