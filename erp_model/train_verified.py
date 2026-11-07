"""Scan all materials, learn only semantically verified code fields. Never train OTHER/family as code classification."""
import csv,json,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from .schema import SLOTS
from .code_rules import CodeRules
from .preprocess import preprocess
from .candidates_v2 import expanded_preprocess
from .features import Features
from .models import ClassifierBundle,RankBundle,WEIGHTS
from .train import read_csv,write_json,metrics,thresholds,forest_adopt,score
from .train_material import rank_metrics,top_predictions

def train(directory='artifacts/verified_v3',rule_path='config/code_rules.json'):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);rules=CodeRules(rule_path)
    tr=read_csv('dataset/ERP_원재료_training.csv');ev=read_csv('dataset/ERP_원재료_inference.csv')
    tp=[expanded_preprocess(r['품목명'],r['규격명']) for r in tr];ep=[preprocess(r['품목명'],r['규격명']) for r in ev]
    ekeys={p.key for p in ep};keep=[i for i,p in enumerate(tp) if p.key not in ekeys]
    parsed=[rules.parse(r['품목코드']) for r in tr]
    # This marker stratifies unknown rows; it is NEVER an estimator class or a prediction value.
    strata=[parsed[i]['labels'].get('classification','__UNLABELED_FOR_SPLIT_ONLY__') for i in keep]
    groups=[json.dumps(tp[i].key,ensure_ascii=False) for i in keep];folds={}
    for f,(_,vi) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(np.zeros(len(keep)),strata,groups)):
        for j in vi:folds[keep[j]]=f
    for f in range(3):assert not {tp[i].key for i in keep if folds[i]==f}&{tp[i].key for i in keep if folds[i]!=f}
    with (out/'fold_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['source_row','normalized_name','normalized_spec','fold','actual_classification_label'])
        for i in keep:w.writerow([i+2,*tp[i].key,folds[i],parsed[i]['labels'].get('classification','')])
    shared={f:Features('classification',False).fit([tp[i] for i in keep if folds[i]!=f]).text for f in range(3)}
    shared_final=Features('classification',False).fit([tp[i] for i in keep]).text
    manifest={'version':'3.0','seed':42,'actual_code_labels_only':True,'binary_D_OTHER_removed':True,
        'rules_source':str(rule_path),'feature_inputs':['품목명','규격명'],'slots':{},
        'pending_rules':'Only DIODE_GENERAL_13 has verified semantic positions. Other observed prefixes are NOT training labels.'}
    comparison={};training_counts={};label_review=[]
    for i,r in enumerate(tr):
        if parsed[i]['issues']:label_review.append({'source_row':i+2,'name':r['품목명'],'spec':r['규격명'],'code':r['품목코드'],**parsed[i]})
    write_json(out/'training_label_review.json',label_review)
    for slot in SLOTS:
        ix=[i for i in keep if slot in parsed[i]['labels']];rows=[tp[i] for i in ix];y=[parsed[i]['labels'][slot] for i in ix]
        classes=sorted(set(y));training_counts[slot]={'rows':len(ix),'classes':len(classes),'labels':dict(Counter(y)),'label_source':'Verified code slices and per-field allowed values only',
            'full_input_rows_scanned':len(tr),'remaining_input_rows_scanned':len(keep),'label_coverage_remaining':len(ix)/len(keep),
            'scheme_ids':dict(Counter(parsed[i]['code_scheme_id'] for i in ix))}
        print('Verified slot',slot,len(ix),'rows',len(classes),'classes',flush=True)
        if not ix:manifest['slots'][slot]={'model_status':'no_verified_labels','classes':[]};continue
        if slot=='part_number':
            pred=['']*len(ix);foldinfo=[]
            for f in range(3):
                ti=[j for j,i in enumerate(ix) if folds[i]!=f];vi=[j for j,i in enumerate(ix) if folds[i]==f]
                model=RankBundle(True).fit([rows[j] for j in ti],[y[j] for j in ti],shared[f])
                for j,v in zip(vi,top_predictions(model,[rows[j] for j in vi])):pred[j]=v
                foldinfo.append({'fold':f,'train_rows':len(ti),'valid_rows':len(vi),'ranking_groups':model.train_groups,'issue':model.issue})
            model=RankBundle(True).fit(rows,y,shared_final)
            cfg={'model':'mean_rank_ensemble','weights':{'xgb_rank':.5,'lgb_rank':.5},'rank_confirmation_enabled':False,'max_mean_rank':0,'model_status':'trained',
                'confirmation_reason':'No independently validated automatic confirmation policy for the new expanded label set'}
            comparison[slot]={'oof':rank_metrics(rows,y,pred),'folds':foldinfo}
        else:
            kinds=['xgb','lgb','ensemble']+(['forest','ensemble_forest'] if slot in ['classification','packing'] else [])
            ps={k:np.zeros((len(ix),len(classes))) for k in kinds};unsafe=set();fi=[]
            for f in range(3):
                ti=[j for j,i in enumerate(ix) if folds[i]!=f];vi=[j for j,i in enumerate(ix) if folds[i]==f]
                model=ClassifierBundle(slot,True).fit([rows[j] for j in ti],[y[j] for j in ti],shared[f])
                result=model.probabilities([rows[j] for j in vi],classes)
                for k in kinds:ps[k][vi]=result[k]
                absent=set(classes)-set(model.classes);unsafe.update(absent)
                fi.append({'fold':f,'train_rows':len(ti),'valid_rows':len(vi),'absent_training_labels':sorted(absent),'untrainable':model.issue})
            mm={k:metrics(y,np.array(classes)[p.argmax(axis=1)].tolist(),slot) for k,p in ps.items()}
            meaningful=len(classes)>=2
            choices=['xgb','lgb','ensemble']
            if slot in ['classification','packing']:
                accepted=meaningful and forest_adopt(mm['ensemble'],mm['ensemble_forest'],slot)
                # Generic actual classification uses macro F1; legacy D precision is never a selection rule here.
                if slot=='classification':accepted=meaningful and mm['ensemble_forest']['macro_f1']>mm['ensemble']['macro_f1']
                mm['forest_adopted_over_ensemble']=accepted
                if accepted:choices.append('ensemble_forest')
            chosen=max(choices,key=lambda k:(mm[k]['macro_f1'],mm[k]['accuracy'],k=='ensemble'))
            cfg={'classes':classes,'model':chosen,'weights':WEIGHTS[chosen],'unsafe_labels':sorted(unsafe),
                'thresholds':thresholds(y,ps[chosen],classes,slot),
                'model_status':'trained' if meaningful else 'single_class_no_multiclass_fit',
                'meaningful_multiclass_evaluation':meaningful}
            model=ClassifierBundle(slot,True).fit(rows,y,shared_final)
            comparison[slot]={'models':mm,'folds':fi,'selected':chosen,'meaningful_multiclass_evaluation':meaningful}
        joblib.dump(model,out/(slot+'.joblib'),compress=3);manifest['slots'][slot]=cfg
    joblib.dump({'text':shared_final},out/'preprocessor.joblib',compress=3)
    write_json(out/'manifest.json',manifest);write_json(out/'training_counts.json',training_counts);write_json(out/'cv_comparison.json',comparison)
    shutil.copyfile(rule_path,out/'code_rules.json')
    write_json(out/'frozen_selection.json',{'selection_source':'training OOF only; no evaluation labels used',
        'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.suffix=='.joblib' or f.name in ['manifest.json','code_rules.json']}})
    audit={'train_original':len(tr),'final_rows':len(ev),'excluded_overlap':len(tr)-len(keep),'remaining_train':len(keep),
        'normalized_key_overlap_after_exclusion':0,'unsupported_or_partly_invalid_train_rows':len(label_review),
        'full_data_scan':True,'global_diode_107_filter_removed':True,'actual_classification_classes':training_counts['classification']['labels'],
        'verified_rule_count':len(rules.rules),'required_to_train_real_multiclass':'At least one additional verified code classification label and semantic slice rule',
        'label_rule_validation':rules.document,'source_hashes':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in ['dataset/ERP_원재료_training.csv','dataset/ERP_원재료_inference.csv']}}
    write_json(out/'data_audit.json',audit)
    print('Verified code-slot training complete. Actual multiclass classification is data-blocked with one verified class.',flush=True)

if __name__=='__main__':train()
