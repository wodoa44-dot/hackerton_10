"""HTTP adapter for whole-material code generation and independent slot events."""
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .schema import SLOTS


class Input(BaseModel):
    name: str
    spec: str = ""
    slot: str | None = None
    code_scheme_id: str | None = None
    candidate_limit: int = Field(default=5, ge=1, le=10)


def create_app(directory, root_path=""):
    from .generation import GenerationPredictor

    from .artifact_paths import generation_directory
    current = generation_directory(directory)
    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            application.state.csv_jobs.close()

    app = FastAPI(root_path=root_path, lifespan=lifespan, title="ERP 원재료 품목코드 생성",
                  description="품목명과 규격명에서 실제 코드 분류값과 여섯 칸을 예측하고 회사 규칙에 맞춰 코드를 생성합니다.")
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest["version"]).startswith("6"):
        from .spectrum import SpectrumPredictor
        predictor_type = SpectrumPredictor
    else:
        predictor_type = GenerationPredictor
    predictor = predictor_type(current)
    app.state.material_factory = lambda slot=None: predictor_type(current, slot)
    web = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/model-info")
    def model_info():
        manifest = predictor.state.manifest
        metrics_path = current / "evaluation_metrics.json"
        legacy = Path(directory) / "final_metrics.json"
        return {
            "version": manifest["version"],
            "csv_upload": True,
            "candidate_k_metrics": json.loads((current / "candidate_k_metrics.json").read_text(encoding="utf-8")) if (current / "candidate_k_metrics.json").exists() else None,
            "spectrum_metrics": json.loads((current / "spectrum_metrics.json").read_text(encoding="utf-8")) if (current / "spectrum_metrics.json").exists() else None,
            "pipeline_validation": json.loads((current / "pipeline_validation.json").read_text(encoding="utf-8")) if (current / "pipeline_validation.json").exists() else None,
            "downloads": [name for name in ["original_code_parts.csv","prediction_parts.csv","spectrum_parts_top5.csv","exploratory_predictions.csv"] if (current / name).exists()],
            "generation_metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None,
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None,
            "web_lookup_audit": json.loads((current / "web_lookup_audit.json").read_text(encoding="utf-8")) if (current / "web_lookup_audit.json").exists() else None,
            "training_counts": json.loads((current / "training_counts.json").read_text(encoding="utf-8")),
            "data_audit": json.loads((current / "data_audit.json").read_text(encoding="utf-8")),
            "selection": {slot: {key: value for key, value in config.items() if key in {"model", "weights", "gpu_training_verified"}}
                          for slot, config in manifest["slots"].items()},
            "training_device": manifest["training_device"], "actual_code_classification": True,
            "manual_source": predictor.state.book.manual,
            "capacity_policy": manifest.get("capacity_policy", {}),
            "manual_comparison": json.loads((current / "manual_comparison.json").read_text(encoding="utf-8")) if (current / "manual_comparison.json").exists() else None,
            "other_class_used": False, "classification_target": manifest["classification_target"],
            "schemes": [{"id": "COMPANY_" + key, "title": row["title"], "prefix": row["prefix"],
                         "length": predictor.state.book.layout(key)[2]}
                        for key, row in predictor.state.book.categories.items() if row.get("generation_enabled", True)],
        }

    @app.get("/downloads/{filename}")
    def download(filename: str):
        if filename not in {"original_code_parts.csv", "prediction_parts.csv", "spectrum_parts_top5.csv", "exploratory_predictions.csv", "candidate_k_curve.csv"}:
            raise HTTPException(404, "Unknown download")
        target = current / filename
        if not target.exists():
            raise HTTPException(404, "Result not ready")
        return FileResponse(target, media_type="text/csv; charset=utf-8", filename=filename)

    @app.get("/health")
    def health():
        return {"status": "ok", "models": str(current), "version": predictor.state.manifest["version"]}

    @app.post("/predict")
    def predict(data: Input):
        if data.slot is not None and data.slot not in SLOTS:
            raise HTTPException(422, "Unknown slot")
        if not data.name.strip():
            raise HTTPException(422, "품목명을 입력하세요.")
        current_predictor = app.state.material_factory(data.slot)
        if hasattr(current_predictor, "candidate_limit"):
            current_predictor.candidate_limit = data.candidate_limit
        def stream():
            for event in current_predictor.events(data.name, data.spec, data.code_scheme_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    from .batch_csv import register_csv_routes
    register_csv_routes(app, current, lambda: app.state.material_factory(None))
    return app