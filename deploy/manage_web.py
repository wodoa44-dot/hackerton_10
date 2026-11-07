#!/usr/bin/env python3
"""Start/stop the deployed web process independently of an interactive terminal."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PID=ROOT/'artifacts/web.pid'
def alive():
    if not PID.exists():return None
    try:
        pid=int(PID.read_text());os.kill(pid,0)
        proc=Path('/proc')/str(pid)
        if proc.exists():
            cmd=(proc/'cmdline').read_bytes()
            if b'erp_model' not in cmd or b'serve' not in cmd:return None
        return pid
    except (OSError,ValueError):return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['start','stop','status'])
    ap.add_argument('--port',type=int,default=8501);ap.add_argument('--prefix',default='/i/OsAdz8rQR7JSXeW6')
    a=ap.parse_args();pid=alive()
    if a.action=='status':print(json.dumps({'running':bool(pid),'pid':pid}));return
    if a.action=='stop':
        if pid:os.kill(pid,signal.SIGTERM);PID.unlink(missing_ok=True);print('Stopped',pid)
        else:print('No managed web process')
        return
    if pid:print('Already running',pid);return
    python=ROOT/'.venv/bin/python'
    if not python.exists():python=ROOT/'.venv/Scripts/python.exe'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUTF8='1')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    with (ROOT/'artifacts/web.log').open('ab') as log:
        proc=subprocess.Popen([str(python),'-m','erp_model','serve','--models',str(ROOT/'artifacts/run1'),'--host','0.0.0.0','--port',str(a.port),'--root-path',a.prefix],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    PID.write_text(str(proc.pid))
    for _ in range(40):
        if proc.poll() is not None:raise RuntimeError('Web startup failed; inspect artifacts/web.log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:'+str(a.port)+a.prefix+'/health',timeout=1) as r:
                if r.status==200:print('Started',proc.pid,'port',a.port,'prefix',a.prefix);return
        except OSError:pass
        time.sleep(.25)
    raise RuntimeError('Web readiness timeout; inspect artifacts/web.log')
if __name__=='__main__':main()
