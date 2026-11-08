import os
import numpy as np
from scipy.stats import rankdata
from xgboost import XGBClassifier, XGBRanker
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import RandomForestClassifier
from erp_model.encoders.features import Features
from erp_model.shared.schema import SLOTS

WEIGHTS={'xgb':{'xgb':1.},'lgb':{'lgb':1.},'ensemble':{'xgb':.5,'lgb':.5},'forest':{'forest':1.},'ensemble_forest':{'xgb':.4,'lgb':.4,'forest':.2}}
THREADS=max(1, int(os.environ.get('ERP_TRAIN_THREADS', '2')))
TREES=max(10, int(os.environ.get('ERP_TRAIN_TREES', '90')))

def classifier(kind):
    if kind=='xgb':return XGBClassifier(n_estimators=TREES,max_depth=3,learning_rate=.08,n_jobs=THREADS,random_state=42,tree_method='hist',device=os.environ.get('ERP_XGB_DEVICE','cpu'),verbosity=0)
    if kind=='lgb':return LGBMClassifier(n_estimators=TREES,max_depth=4,num_leaves=12,min_child_samples=3,learning_rate=.06,n_jobs=THREADS,random_state=42,verbosity=-1)
    return RandomForestClassifier(n_estimators=160,min_samples_leaf=1,class_weight='balanced_subsample',n_jobs=THREADS,random_state=42)

class ClassifierBundle:
    def __init__(self,slot,structured=True):self.slot=slot;self.feature=Features(slot,structured);self.models={};self.issue=None
    def fit(self,rows,y,shared_text=None):
        self.classes=sorted(set(y));self.feature.fit(rows,shared_text);x=self.feature.transform(rows)
        if len(self.classes)<2:
            self.issue='single training class; model fitting impossible';return self
        yy=np.array([self.classes.index(v) for v in y])
        for kind in (['xgb','lgb','forest'] if self.slot in ['classification','packing'] else ['xgb','lgb']):
            self.models[kind]=classifier(kind).fit(x,yy)
        return self
    def probabilities(self,rows,classes):
        x=self.feature.transform(rows);result={}
        for name in (['xgb','lgb','forest'] if self.slot in ['classification','packing'] else ['xgb','lgb']):
            out=np.zeros((len(rows),len(classes)))
            if self.models:
                local=self.models[name].predict_proba(x)
                for j,label in enumerate(self.classes):out[:,classes.index(label)]=local[:,j]
            else:out[:,classes.index(self.classes[0])]=1.
            result[name]=out
        for combo,weights in WEIGHTS.items():
            if all(k in result for k in weights):result[combo]=sum(result[k]*w for k,w in weights.items())
        return result

class RankBundle:
    def __init__(self,structured=True):self.slot='part_number';self.feature=Features(self.slot,structured);self.models={};self.issue=None
    def fit(self,rows,truth,shared_text=None):
        self.feature.fit(rows,shared_text)
        eligible=[(r,y) for r,y in zip(rows,truth) if len(r.candidates)>1 and y in {c['value'] for c in r.candidates}]
        self.train_groups=len(eligible)
        if not eligible:self.issue='no mixed positive/negative candidate groups';return self
        rr,yy=zip(*eligible);x,counts=self.feature.candidate_matrix(list(rr))
        targets=np.array([int(c['value']==y) for r,y in eligible for c in r.candidates])
        qid=np.repeat(np.arange(len(counts)),counts)
        self.models['xgb']=XGBRanker(objective='rank:pairwise',n_estimators=TREES,max_depth=3,learning_rate=.08,n_jobs=THREADS,random_state=42,tree_method='hist',device=os.environ.get('ERP_XGB_DEVICE','cpu')).fit(x,targets,qid=qid)
        self.models['lgb']=LGBMRanker(objective='lambdarank',n_estimators=TREES,num_leaves=12,min_child_samples=3,learning_rate=.06,n_jobs=THREADS,random_state=42,verbosity=-1).fit(x,targets,group=counts)
        return self
    def rankings(self,rows):
        x,counts=self.feature.candidate_matrix(rows)
        if x is None:return [{'xgb':[],'lgb':[],'ensemble':[]} for _ in rows]
        scores={k:m.predict(x) for k,m in self.models.items()}
        out=[];offset=0
        for count in counts:
            d={}
            for k in ['xgb','lgb']:
                # Average ranks preserve genuine score ties.
                d[k]=rankdata(-scores[k][offset:offset+count],method='average').tolist() if k in scores else [float(count+1)]*count
            d['ensemble']=((np.array(d['xgb'])+np.array(d['lgb']))/2).tolist()
            out.append(d);offset+=count
        return out
