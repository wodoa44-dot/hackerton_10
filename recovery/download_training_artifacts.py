"""Back up only this task's learned artifacts via the existing SSH connection."""
import argparse
from pathlib import Path
import subprocess
import tarfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    parser.add_argument('--scope',choices=['encoder','results'],default='encoder')
    parser.add_argument('--extract',action='store_true')
    args=parser.parse_args()
    if args.scope=='encoder':
        files=['artifacts/encoder_v7/'+name for name in ['weights.pt','encoder_config.json','completion.json',
               'grouped_validation.json','training_data_audit.json','tokenizer','validation_history.json','full_training_history.json']]
    else:
        files=['artifacts/'+name for name in ['generation_v6','generation_v7','encoder_v7','encoder_comparison','encoder_comparison_job.json']]
    command='tar --exclude=*/batch_jobs -czf - -C /data/erp_encoder_20261107_v7/source '+' '.join(files)
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    with args.archive.open('wb') as output:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                        'ff9ebb72-6ae84c15.hackathon',command],stdout=output,check=True)
    if args.extract:
        with tarfile.open(args.archive) as archive:
            if any(not member.name.startswith('artifacts/') for member in archive.getmembers()):
                raise ValueError('Unexpected artifact entry')
            archive.extractall(Path(__file__).resolve().parents[1],filter='data')
    print('DOWNLOADED',args.scope,args.archive.stat().st_size,'bytes')


if __name__=='__main__':main()
