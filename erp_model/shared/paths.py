"""소스 이동과 무관하게 고정되는 프로젝트 경로."""
from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
