#!/usr/bin/env python3
"""공통 실행 진입점. 서버에 준비된 가상환경과 로컬 실행 설정을 자동으로 사용합니다."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def prepare_runtime():
    environment=os.environ.copy()
    settings=ROOT/".deployment/environment.json"
    if settings.is_file():
        saved=json.loads(settings.read_text(encoding="utf-8"))
        allowed={"PATH","LD_LIBRARY_PATH","TMPDIR","CUDA_CACHE_PATH","CUDA_CACHE_MAXSIZE","TRITON_CACHE_DIR","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONUTF8","PYTHONDONTWRITEBYTECODE","MALLOC_ARENA_MAX","ERP_SOURCE_COMMIT","ERP_PUBLIC_REPORTS","ERP_RUNTIME_ENVIRONMENT"}
        environment.update({key:str(value) for key,value in saved.items() if key in allowed})
    candidates=[ROOT/".venv/bin/python",ROOT/".venv/Scripts/python.exe"]
    python=next((p for p in candidates if p.is_file()),Path(sys.executable))
    needs_venv=python!=Path(sys.executable) and (ROOT/".venv").is_dir() and Path(sys.prefix).resolve()!=(ROOT/".venv").resolve()
    changed=any(os.environ.get(key)!=value for key,value in environment.items())
    if environment.get("ERP_RUNTIME_READY")!=str(ROOT) and (changed or needs_venv):
        environment["ERP_RUNTIME_READY"]=str(ROOT)
        os.execve(str(python),[str(python),str(ROOT/"run.py"),*sys.argv[1:]],environment)

if __name__=="__main__":
    os.chdir(ROOT)
    prepare_runtime()
    from erp_model.cli import main
    main()
