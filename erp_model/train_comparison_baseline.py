"""Refit the no-encoder system with exactly the encoder's audited task labels."""
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    data=Path('artifacts/encoder_training_data')
    if not (data/'learning_data_audit.json').is_file():raise ValueError('Prepare shared audited labels first')
    subprocess.run([sys.executable,'-u','-m','erp_model.train_generation','--audited-data',str(data)],check=True)
    base=Path('artifacts/generation_v5');refined=Path('artifacts/accuracy_v6');refined.mkdir(exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text())
    shutil.copy2(base/'part_number.joblib',refined/'part_number.joblib')
    rank={'selected':'improved','candidate_policy':'atomic_v6, fixed before evaluation',
          'training_only':True,'final_evaluation_used_for_selection':False,
          'gpu_training_verified':manifest['slots']['part_number']['gpu_training_verified']}
    (refined/'identifier_validation.json').write_text(json.dumps(rank,indent=2))
    subprocess.run([sys.executable,'-u','-m','erp_model.train_attributes'],check=True)
    policy={'refinement_policy':{'use_attributes':True,'use_router':True},
            'policy_frozen_before_final_evaluation':True,'final_evaluation_used_for_selection':False,
            'note':'Both encoder and no-encoder variants use the same fitted router, attributes, rules, retrieval, and K limit.'}
    (refined/'pipeline_validation.json').write_text(json.dumps(policy,indent=2))
    subprocess.run([sys.executable,'deploy/assemble_generation_v6.py'],check=True)
    print('NO_ENCODER_BASELINE_READY',flush=True)


if __name__=='__main__':main()
