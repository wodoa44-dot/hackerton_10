"""Durable, bounded-resource GPU training and paired evaluation, without downloads."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'artifacts/encoder_comparison_job.json'
STEPS = [
    ('prepare', ['-m', 'erp_model.prepare_encoder_data']),
    ('encoder', ['-m', 'erp_model.train_encoder', '--epochs', '24', '--batch-size', '64', '--cpu-threads', '2', '--max-gpu-memory-gib', '8']),
    ('no_encoder', ['-m', 'erp_model.train_comparison_baseline']),
    ('encoder_trees', ['-m', 'erp_model.train_encoder_augmented', '--device', 'cuda']),
    ('evaluation', ['-m', 'erp_model.compare_encoders']),
]


def main():
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    state = {'status': 'running', 'started_at': time.time(), 'steps': {},
             'source_commit': os.environ.get('ERP_SOURCE_COMMIT'),
             'evaluation_used_for_fit_or_selection': False}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               ERP_TRAIN_THREADS='2', ERP_XGB_DEVICE='cuda', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', MALLOC_ARENA_MAX='2')

    def save():
        pending = STATUS.with_suffix('.pending')
        pending.write_text(json.dumps(state, indent=2), encoding='utf-8')
        pending.replace(STATUS)

    save()
    try:
        for name, command in STEPS:
            state['phase'] = name
            state['steps'][name] = {'started_at': time.time(), 'status': 'running'}
            save()
            print('PIPELINE_PHASE', name, flush=True)
            with (ROOT/'artifacts'/('job_'+name+'.log')).open('ab', buffering=0) as log:
                subprocess.run([sys.executable, '-u', *command], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state['steps'][name].update(status='complete', finished_at=time.time())
            save()
        state.update(status='complete', phase='complete', finished_at=time.time())
        save()
        print('ENCODER_COMPARISON_COMPLETE', flush=True)
    except Exception as error:
        state.update(status='failed', error=type(error).__name__+': '+str(error), finished_at=time.time())
        if state.get('phase') in state['steps']:
            state['steps'][state['phase']]['status'] = 'failed'
        save()
        raise


if __name__ == '__main__':
    main()
