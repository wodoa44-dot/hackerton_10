import csv
import hashlib
import json
import platform
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from .preprocess import preprocess, VERSION
from .schema import labels, SLOTS, ALLOWED
from .models import ClassifierBundle, RankBundle, WEIGHTS
from .features import Features

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        data=list(csv.DictReader(f))
    required={'품목명','규격명','품목코드'}
    if not data or not required<=data[0].keys():raise ValueError('Missing CSV columns')
    return data

def metrics(y,pred,slot):
    labs=sorted(set(y)|set(pred)); report=classification_report(y,pred,labels=labs,output_dict=True,zero_division=0)
    out={'n':len(y),'accuracy':accuracy_score(y,pred),'macro_f1':report['macro avg']['f1-score'],'per_class':{k:report[k] for k in labs}}
    if slot=='classification':out['D_precision']=report.get('D',{}).get('precision',0);out['D_recall']=report.get('D',{}).get('recall',0);out['D_f1']=report.get('D',{}).get('f1-score',0)
    return out

def score(m,slot):return m['D_f1'] if slot=='classification' else m['macro_f1']
def forest_adopt(base,mix,slot):
    if slot=='classification':return mix['D_precision']>=base['D_precision'] and mix['D_recall']>=base['D_recall'] and score(mix,slot)>score(base,slot)
    return mix['macro_f1']>base['macro_f1']

def thresholds(y,probs,classes,slot):
    order=np.argsort(-probs,axis=1);best=probs[np.arange(len(y)),order[:,0]];second=probs[np.arange(len(y)),order[:,1]] if len(classes)>1 else np.zeros(len(y))
    pred=np.array(classes)[order[:,0]];y=np.array(y);out={}
    for label in classes:
        target=.98 if slot=='classification' else .9
        eligible=pred==label;winner=None
        for cut in [0.,.4,.5,.6,.7,.8,.85,.9,.95,.98,.99]:
            for gap in [0.,.05,.1,.2,.3,.5]:
                mask=eligible&(best>=cut)&(best-second>=gap); n=int(mask.sum())
                if n<3:continue
                precision=float((pred[mask]==y[mask]).mean())
                if precision>=target and (winner is None or (n,precision,-cut,-gap)>winner[0]):winner=((n,precision,-cut,-gap),{'score':cut,'margin':gap,'oof_accepted':n,'oof_accuracy':precision,'target':target})
        out[label]=winner[1] if winner else {'score':1.01,'margin':1.01,'oof_accepted':0,'reason':'fewer than 3 correct-confidence validation predictions at target precision','target':target}
    return out

def candidate_audit(prepared,truth):
    out={'n':len(prepared)}
    for stage in [1,2,3]:
        counts=[sum(c['stage']<=stage for c in p.candidates) for p in prepared]
        hits=[y in {c['value'] for c in p.candidates if c['stage']<=stage} for p,y in zip(prepared,truth)]
        out[str(stage)]={'included':sum(hits),'coverage':float(np.mean(hits)) if hits else 0,'candidate_mean':float(np.mean(counts)) if counts else 0,'candidate_median':float(np.median(counts)) if counts else 0,'candidate_max':max(counts,default=0)}
    return out

def train(train_path, eval_path, out_dir, full_path=None):
    """Restored compatibility entrypoint; use the current company-code pipeline."""
    from .train_generation import train as train_company_codes
    return train_company_codes(train_path, eval_path, out_dir, full_path)
