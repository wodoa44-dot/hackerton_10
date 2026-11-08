import argparse
import json
import os
from pathlib import Path
from erp_model.shared.schema import SLOTS

def main():
    p=argparse.ArgumentParser(description='ERP full-material code generation and six independent field predictors')
    sub=p.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--train',required=True);tr.add_argument('--evaluation',required=True);tr.add_argument('--full');tr.add_argument('--out',default='artifacts/generation_v5');tr.add_argument('--legacy-diode',action='store_true')
    pr=sub.add_parser('predict');pr.add_argument('--models',default='artifacts/run1');pr.add_argument('--name',required=True);pr.add_argument('--spec',default='');pr.add_argument('--slot',choices=SLOTS);pr.add_argument('--materials',default='artifacts/material_v2');pr.add_argument('--legacy-diode',action='store_true');pr.add_argument('--code-scheme')
    sv=sub.add_parser('serve');sv.add_argument('--models',default='artifacts/run1');sv.add_argument('--host',default='0.0.0.0');sv.add_argument('--port',type=int,default=8000);sv.add_argument('--root-path',default='')
    a=p.parse_args()
    if a.command=='train':
        if a.legacy_diode:
            from erp_model.training.train import train
            train(a.train,a.evaluation,a.out,a.full)
        else:
            from erp_model.training.train_generation import train
            train(a.train,a.evaluation,a.out,a.full)
    elif a.command=='predict':
        if a.legacy_diode:
            from erp_model.prediction.inference import Predictor
            events=Predictor(a.models,a.slot).events(a.name,a.spec)
        else:
            from erp_model.prediction.generation import GenerationPredictor
            from erp_model.shared.artifact_paths import generation_directory
            current=generation_directory(a.models)
            version=json.loads((current/"manifest.json").read_text())["version"]
            if str(version).startswith(("6", "7")):
                from erp_model.prediction.spectrum import SpectrumPredictor
                events=SpectrumPredictor(current,a.slot).events(a.name,a.spec,a.code_scheme)
            else:
                events=GenerationPredictor(current,a.slot).events(a.name,a.spec,a.code_scheme)
        for event in events:print(json.dumps(event,ensure_ascii=False),flush=True)
    else:
        import uvicorn
        from erp_model.api.service import create_app
        uvicorn.run(create_app(a.models,a.root_path),host=a.host,port=a.port)
if __name__=='__main__':main()
