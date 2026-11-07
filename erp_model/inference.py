import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
from scipy.sparse import hstack
from .preprocess import preprocess
from .schema import SLOTS, ALLOWED, PATTERN
from .models import WEIGHTS

def valid_code(values):
    if any(not isinstance(v,str) for v in values):return False
    by=dict(zip(SLOTS,values))
    return (all(len(v)==n for v,n in zip(values,[1,1,7,2,1,1]))
            and bool(PATTERN.fullmatch(''.join(values)))
            and all(by[s] in ALLOWED[s] for s in ALLOWED)
            and bool(by['part_number'].strip('-')))

def assembly(results):
    by={r['slot']:r for r in results}
    predicted={}
    for slot in SLOTS:
        r=by.get(slot,{})
        candidates=r.get('candidates',[])
        predicted[slot]=None if r.get('status')=='error' else r.get('value') or (candidates[0]['value'] if candidates else None)
    values=[predicted[s] for s in SLOTS]
    prediction_valid=valid_code(values)
    result={'slot':'assembly','value':None,'status':'review','scope':'general diode format',
            'predicted_code':''.join(values) if prediction_valid else None,
            'predicted_values':predicted,
            'prediction_status':'available' if prediction_valid else 'unavailable'}
    missing=[s for s in SLOTS if predicted[s] is None]
    result['prediction_reason']=('Missing predictions: '+', '.join(missing) if missing else
        'Classification is not D' if predicted['classification']!='D' else
        'Top predictions validated against code tables and lengths' if prediction_valid else 'Code table/length validation failed')
    pending=[s for s in SLOTS if by.get(s,{}).get('status')!='confirmed']
    if pending:result['reason']='Unconfirmed slots: '+', '.join(pending)
    elif by['classification']['value']!='D':result['reason']='Classification is not D'
    elif valid_code([by[s]['value'] for s in SLOTS]):
        result.update(value=''.join(by[s]['value'] for s in SLOTS),status='confirmed',reason='Validated code tables and 1+1+7+2+1+1 lengths')
    else:result['reason']='Code table/length validation failed'
    return result

class Predictor:
    def __init__(self, directory, slot=None):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        self.slots=[slot] if slot else SLOTS
        self.shared=joblib.load(self.directory/'preprocessor.joblib')
        self.models={s:joblib.load(self.directory/(s+'.joblib')) for s in self.slots}
        for model in self.models.values():model.feature.text=self.shared['text']
    def predict_slot(self,slot,p,computed=None):
        try:
            result=self._predict_slot(slot,p,computed)
            result['predicted_value']=result['value'] or (result['candidates'][0]['value'] if result['candidates'] else None)
            return result
        except Exception as exc:return {'slot':slot,'value':None,'status':'error','candidates':[],'evidence':[],'reason':type(exc).__name__+': '+str(exc),'scope':'general diode format' if slot!='classification' else 'all ERP items'}
    def _predict_slot(self,slot,p,computed=None):
        model=self.models[slot];cfg=self.manifest['slots'][slot]
        result={'slot':slot,'value':None,'status':'review','candidates':[],'evidence':[e for e in p.evidence if e['slot'] in [slot,'physical']],'reason':'','scope':'general diode format' if slot!='classification' else 'all ERP items'}
        if slot=='part_number':
            if not p.candidates:result['reason']='no_candidates';return result
            ranks=(model.rankings([p])[0] if computed is None else computed)['ensemble'];order=np.argsort(ranks,kind='stable')
            result['candidates']=[{'value':p.candidates[i]['value'],'score':float(ranks[i]),'score_type':'mean_rank_lower_is_better','evidence':p.candidates[i]['origins']} for i in order]
            best=int(order[0]); c=p.candidates[best];result['evidence']=c['origins']
            if model.issue:result['reason']='ranking_model_unavailable: '+model.issue
            elif len(order)>1 and ranks[order[0]]==ranks[order[1]]:result['reason']='rank_tie'
            elif c['features']['looks_unit'] or c['features']['looks_case'] or len(c['value'].rstrip('-'))<4:result['reason']='insufficient_candidate_evidence'
            elif not cfg['rank_confirmation_enabled']:result['reason']='insufficient_oof_rank_accuracy'
            elif ranks[best]>cfg['max_mean_rank']:result['reason']='weak_rank_agreement'
            else:result.update(value=c['value'],status='confirmed',reason='Unique rank winner with input evidence and OOF-validated agreement')
            return result
        classes=cfg['classes']; probs=(model.probabilities([p],classes) if computed is None else computed)[cfg['model']][0];order=np.argsort(-probs,kind='stable');idx=int(order[0]);value=classes[idx]
        score=float(probs[idx]);margin=score-float(probs[order[1]]) if len(order)>1 else score
        result['candidates']=[{'value':classes[i],'score':float(probs[i]),'score_type':'probability_uncalibrated'} for i in order]
        policy=cfg['thresholds'].get(value,{'score':1.01,'margin':1.01})
        reasons=[]
        if model.issue:reasons.append('model_unavailable: '+model.issue)
        if value in cfg['unsafe_labels']:reasons.append('label_absent_in_cv_training_fold')
        if score<policy['score'] or margin<policy['margin']:reasons.append('below_oof_confidence_threshold')
        if slot=='vendor':
            vendors={e['normalized_value'] for e in p.evidence if e['slot']=='vendor'}
            if len(vendors)>1:reasons.append('conflicting_manufacturer_evidence')
            elif not vendors:reasons.append('no_explicit_manufacturer_evidence')
            elif next(iter(vendors))!=value:reasons.append('manufacturer_evidence_disagrees_with_model_or_unmapped')
        if reasons:result['reason']='; '.join(reasons)
        else:result.update(value=value,status='confirmed',reason='OOF score and margin thresholds passed')
        return result
    def events(self,name,spec=''):
        p=preprocess(name,spec)
        # Compute common name/spec TF-IDF exactly once before independent slot workers.
        p._text_cache=(id(self.shared['text'][0]), hstack([v.transform([p.normalized[c]]) for v,c in zip(self.shared['text'],['품목명','규격명'])],format='csr'))
        if len(self.slots)==1:
            yield self.predict_slot(self.slots[0],p);return
        results=[]
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs={pool.submit(self.predict_slot,s,p):s for s in self.slots}
            for f in as_completed(jobs):
                r=f.result();results.append(r);yield r
        yield assembly(results)
    def batch(self, prepared):
        """Batch scoring shares the same policies; optimized model calls via cached outputs."""
        # Matrix inference is computed once per slot, then policies are applied unchanged.
        saved={};failed={}
        for slot,m in self.models.items():
            try:
                saved[slot]=m.rankings(prepared) if slot=='part_number' else m.probabilities(prepared,self.manifest['slots'][slot]['classes'])
            except Exception as exc:
                failed[slot]={'slot':slot,'value':None,'status':'error','candidates':[],'evidence':[],'reason':type(exc).__name__+': '+str(exc),'scope':'general diode format' if slot!='classification' else 'all ERP items'}
        result=[]
        for i,p in enumerate(prepared):
            events=[]
            for slot in self.slots:
                if slot in failed:events.append(dict(failed[slot]));continue
                vals=saved[slot]
                computed=vals[i] if slot=='part_number' else {k:a[i:i+1] for k,a in vals.items()}
                events.append(self.predict_slot(slot,p,computed))
            if len(self.slots)>1:events.append(assembly(events))
            result.append(events)
        return result
