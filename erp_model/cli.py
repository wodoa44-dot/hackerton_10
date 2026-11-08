"""웹·학습·평가·결과 리포트의 단일 명령 진입점."""
import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from erp_model.shared.paths import PROJECT_ROOT, PACKAGE_ROOT

TRAINING = {
    "baseline": "erp_model.training.train_comparison_baseline",
    "encoder": "erp_model.training.train_encoder",
    "augmented": "erp_model.training.train_encoder_augmented",
    "attributes": "erp_model.training.train_attributes",
    "pipeline": "erp_model.training.run_encoder_comparison",
}
EVALUATION = {
    "encoder": "erp_model.evaluation.compare_encoders",
    "cnn-only": "erp_model.evaluation.evaluate_encoder",
    "generation": "erp_model.evaluation.evaluate_generation",
    "spectrum": "erp_model.evaluation.evaluate_spectrum",
}

def _module(name, arguments):
    sys.argv = [name, *arguments]
    runpy.run_module(name, run_name="__main__")

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["web"]
    parser = argparse.ArgumentParser(description="ERP 원재료 품목코드: 웹·학습·평가·CSV/그래프")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("web", aliases=["serve"], help="선택된 모델로 웹/API 실행")
    web.add_argument("--models", default="artifacts/run1")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8501)
    web.add_argument("--root-path", default="")
    pred = sub.add_parser("predict", help="품목명·규격명으로 코드 예측")
    pred.add_argument("--models", default="artifacts/run1")
    pred.add_argument("--name", required=True)
    pred.add_argument("--spec", default="")
    pred.add_argument("--slot", choices=["classification","trait","part_number","case","packing","vendor"])
    pred.add_argument("--code-scheme")
    pred.add_argument("--legacy-diode", action="store_true", help=argparse.SUPPRESS)
    pred.add_argument("--materials", default="artifacts/material_v2", help=argparse.SUPPRESS)
    train = sub.add_parser("train", help="학습 단계 선택")
    train.add_argument("stage", choices=[*TRAINING, "generation"])
    evaluation = sub.add_parser("evaluate", help="모델을 다시 추론하여 평가")
    evaluation.add_argument("stage", choices=list(EVALUATION))
    sub.add_parser("prepare", help="공통 규칙 검증·학습 라벨 마스크 준비")
    report = sub.add_parser("report", help="저장된 결과에서 CSV·표·PNG·SVG 생성")
    report.add_argument("--input", default="artifacts/encoder_comparison")
    report.add_argument("--out", default="artifacts/reports/encoder_topk")
    report.add_argument("--runtime")
    sub.add_parser("check", help="폴더 구조·모듈 호환 경로 점검")
    managed = sub.add_parser("manage-web", help="백그라운드 웹 시작·중지·상태")
    managed.add_argument("action", choices=["start","stop","status"])
    managed.add_argument("--port", type=int, default=8501)
    managed.add_argument("--prefix", default="")
    # 이전 python -m erp_model train --train ... 호출을 계속 지원합니다.
    if arguments[0] == "train" and len(arguments)>1 and arguments[1].startswith("--") and arguments[1] != "--help":
        from erp_model.shared.legacy_cli import main as legacy_main
        sys.argv = ["erp_model", *arguments]
        return legacy_main()
    args, remaining = parser.parse_known_args(arguments)
    if remaining and args.command not in {"train","evaluate","prepare"}:
        parser.error("알 수 없는 인자: " + " ".join(remaining))
    try:
        if args.command in {"web","serve"}:
            import uvicorn
            from erp_model.api.service import create_app
            uvicorn.run(create_app(args.models,args.root_path),host=args.host,port=args.port)
        elif args.command == "predict":
            from erp_model.shared.legacy_cli import main as legacy_main
            sys.argv = ["erp_model", *arguments]
            legacy_main()
        elif args.command == "train":
            if args.stage == "generation":
                from erp_model.shared.legacy_cli import main as legacy_main
                sys.argv = ["erp_model", "train", *remaining]
                legacy_main()
            else:
                _module(TRAINING[args.stage],remaining)
        elif args.command == "prepare":
            _module("erp_model.training.data.prepare_encoder_data",remaining)
        elif args.command == "evaluate":
            _module(EVALUATION[args.stage],remaining)
        elif args.command == "check":
            from erp_model.shared.diagnostics import check
            print(json.dumps(check(),ensure_ascii=False,indent=2))
        elif args.command == "manage-web":
            command=[sys.executable,str(PROJECT_ROOT/"deploy/manage_web.py"),args.action,"--port",str(args.port),"--prefix",args.prefix]
            subprocess.run(command,check=True,cwd=PROJECT_ROOT)
        elif args.command == "report":
            node=shutil.which("node")
            if not node:
                parser.error("Node.js 실행 환경이 필요합니다.")
            env=os.environ.copy()
            if args.runtime:
                env["ERP_REPORT_RUNTIME"]=str(Path(args.runtime).resolve())
            script=PACKAGE_ROOT/"evaluation/report/generate_encoder_report.mjs"
            subprocess.run([node,str(script),"--input",args.input,"--out",args.out],check=True,cwd=PROJECT_ROOT,env=env)
    except ModuleNotFoundError as error:
        parser.exit(2,"실행 의존성이 없습니다: "+str(error.name)+". requirements.txt 및 모델에 맞는 requirements-encoder.txt를 확인하세요.\n")

if __name__ == "__main__":
    main()
