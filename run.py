#!/usr/bin/env python3
"""프로젝트 루트에서 웹·학습·평가·리포트를 실행하는 공통 진입점."""
import os
from pathlib import Path
from erp_model.cli import main
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
