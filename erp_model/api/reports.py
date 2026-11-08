"""공개 집계 리포트만 제공하며 행별 private 결과는 노출하지 않습니다."""
import json
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

def register_report_routes(app, root):
    root=Path(root)

    def allowed_files():
        try:
            manifest=json.loads((root/"run_manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError,json.JSONDecodeError):
            return set()
        return {"report.html","run_manifest.json"} | {
            row["path"] for row in manifest.get("outputs",[])
            if not row["path"].startswith(("private/","/")) and ".." not in Path(row["path"]).parts
        }

    @app.get("/reports/encoder-topk/",include_in_schema=False)
    def report_home():
        target=root/"report.html"
        if not target.is_file():
            raise HTTPException(404,"평가 보고서가 아직 준비되지 않았습니다.")
        return FileResponse(target,headers={"Cache-Control":"no-store"})

    @app.get("/reports/encoder-topk/{filename:path}",include_in_schema=False)
    def report_file(filename: str):
        if filename not in allowed_files():
            raise HTTPException(404,"공개 집계 파일이 아닙니다.")
        target=(root/filename).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise HTTPException(404,"파일을 찾을 수 없습니다.")
        return FileResponse(target,headers={"Cache-Control":"no-store"})
