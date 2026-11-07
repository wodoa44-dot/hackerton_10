"""Recreated artifact assembly from the recorded v5/v6 workflow; performs no training."""
import argparse
import json
import shutil
from pathlib import Path


def assemble(base, refinement, output, activate=False):
    base,refinement,output=map(Path,(base,refinement,output))
    required=['manifest.json','preprocessor.joblib','reference.joblib','evaluation_registry.json',
              'operational_master.json','company_rules.json','manual_source.json','scheme_values.json',
              'classification.joblib','trait.joblib','case.joblib','packing.joblib','vendor.joblib']
    extra=['part_number.joblib','attributes.joblib','identifier_validation.json','attribute_validation.json']
    missing=[str(base/f) for f in required if not (base/f).is_file()]
    missing += [str(refinement/f) for f in extra if not (refinement/f).is_file()]
    if missing:raise FileNotFoundError('Train the required artifacts first: '+', '.join(missing))
    if (output/'manifest.json').exists():raise FileExistsError('Choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    excluded={'evaluation_predictions.csv','evaluation_predictions.jsonl','evaluation_metrics.json','release_validation.json'}
    for p in base.iterdir():
        if p.is_file() and p.name not in excluded:shutil.copy2(p,output/p.name)
    for name in extra:shutil.copy2(refinement/name,output/name)
    optional=refinement/'pipeline_validation.json'
    if optional.is_file():shutil.copy2(optional,output/optional.name)
    m=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    rank=json.loads((refinement/'identifier_validation.json').read_text(encoding='utf-8'))
    m.update(version='6.0',base_model=base.name,refinement_policy={'use_attributes':True,'use_router':True},
             accuracy_inputs=['품목명','규격명'],prediction_contract='one_shared_pipeline_six_heads_bounded_spectrum',
             spectrum_max_candidates=10,web_lookup_scope='retained_training_only',evaluation_rows_in_web_lookup=False)
    m['slots']['part_number']['candidate_version']='atomic_v6' if rank['selected']=='improved' else 'baseline'
    m['slots']['part_number']['gpu_training_verified']=bool(rank.get('gpu_training_verified'))
    (output/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    if activate:
        (output.parent/'active_generation.json').write_text(json.dumps({'directory':output.name,'version':'6.0'},indent=2),encoding='utf-8')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='artifacts/generation_v5')
    parser.add_argument('--refinement',default='artifacts/accuracy_v6')
    parser.add_argument('--out',default='artifacts/generation_v6')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    print(assemble(args.base,args.refinement,args.out,args.activate))
