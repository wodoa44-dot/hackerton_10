from erp_model.shared.paths import PACKAGE_ROOT
"""HTTP adapter for whole-material code generation and independent slot events."""
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from erp_model.shared.schema import SLOTS


class Input(BaseModel):
    name: str
    spec: str = ""
    slot: str | None = None
    code_scheme_id: str | None = None
    candidate_limit: int = Field(default=5, ge=1, le=10)
    model_variant: str | None = None


def create_app(directory, root_path=""):
    from erp_model.prediction.generation import GenerationPredictor

    from erp_model.shared.artifact_paths import generation_directory
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
    if str(manifest["version"]).startswith(("6", "7")):
        from erp_model.prediction.spectrum import SpectrumPredictor
        predictor_type = SpectrumPredictor
    else:
        predictor_type = GenerationPredictor
    predictor = predictor_type(current)
    app.state.material_factory = lambda slot=None: predictor_type(current, slot)
    comparison_root = current.parent / "encoder_comparison"
    variant_paths = {"no_encoder": current.parent / "generation_v6",
                     "encoder": current.parent / "generation_v7"}

    def variants():
        return {key: {"available": (path / "manifest.json").is_file(),
                      "active": path.resolve() == current.resolve()}
                for key, path in variant_paths.items()}

    def comparison_report():
        target = comparison_root / "comparison.json"
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    web = PACKAGE_ROOT / "web"
    app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/model-info")
    def model_info(response: Response):
        response.headers["Cache-Control"] = "no-store"
        manifest = predictor.state.manifest
        metrics_path = current / "evaluation_metrics.json"
        comparison = comparison_report()
        active_variant = "encoder" if manifest.get("encoder_augmentation") else "no_encoder"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
        measured = (comparison or {}).get("variants", {}).get(active_variant)
        if measured:
            groups = measured["candidate_k_metrics"]["groups"]
            first = lambda name: next(p for p in groups[name]["points"] if p["k"] == 1)
            metrics = {**measured, "company_layout_rows": measured["manual_parseable_rows"],
                       "code_available": groups["all"]["predicted_rows"],
                       "code_available_rate": groups["all"]["predicted_rows"] / measured["evaluation_rows"] if measured["evaluation_rows"] else None,
                       "full_code_correct": first("all")["correct"], "full_code_exact_rate": first("all")["rate"],
                       "manual_full_code_correct": first("manual")["correct"], "manual_full_code_exact_rate": first("manual")["rate"]}
        return {
            "version": manifest["version"],
            "csv_upload": True,
            "encoder_comparison": comparison,
            "model_variants": variants(),
            "active_model_variant": active_variant,
            "candidate_k_metrics": json.loads((current / "candidate_k_metrics.json").read_text(encoding="utf-8")) if (current / "candidate_k_metrics.json").exists() else None,
            "spectrum_metrics": json.loads((current / "spectrum_metrics.json").read_text(encoding="utf-8")) if (current / "spectrum_metrics.json").exists() else None,
            "pipeline_validation": json.loads((current / "pipeline_validation.json").read_text(encoding="utf-8")) if (current / "pipeline_validation.json").exists() else None,
            "downloads": [name for name in ["original_code_parts.csv","prediction_parts.csv","spectrum_parts_top5.csv","exploratory_predictions.csv"] if (current / name).exists()],
            "generation_metrics": metrics,
            "metrics": metrics,
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

    @app.get("/comparison-downloads/{filename}")
    def comparison_download(filename: str):
        allowed = {"comparison.json", "candidate_k_curve.csv", "candidate_k_curve.svg",
                   "top_k_comparison.csv", "six_slot_comparison.csv", "paired_predictions.csv"}
        if filename not in allowed or not (comparison_root / filename).is_file():
            raise HTTPException(404, "Comparison result not ready")
        return FileResponse(comparison_root / filename, filename=filename)

    @app.get("/comparison-downloads/{variant}/{filename}")
    def comparison_parts(variant: str, filename: str):
        allowed = {slot + "_predictions.csv" for slot in SLOTS} | {"prediction_parts.csv", "six_slot_predictions.csv"}
        target = comparison_root / variant / filename
        if variant not in variant_paths or filename not in allowed or not target.is_file():
            raise HTTPException(404, "Comparison result not ready")
        return FileResponse(target, media_type="text/csv; charset=utf-8", filename=variant + "_" + filename)

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
        if data.model_variant is None:
            current_predictor = app.state.material_factory(data.slot)
        elif data.model_variant in variant_paths and variants()[data.model_variant]["available"]:
            from erp_model.prediction.spectrum import SpectrumPredictor
            current_predictor = SpectrumPredictor(variant_paths[data.model_variant], data.slot)
        else:
            raise HTTPException(422, "선택한 모델의 학습 결과가 아직 준비되지 않았습니다.")
        if hasattr(current_predictor, "candidate_limit"):
            current_predictor.candidate_limit = data.candidate_limit
        def stream():
            for event in current_predictor.events(data.name, data.spec, data.code_scheme_id):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    from erp_model.api.batch_csv import register_csv_routes
    register_csv_routes(app, current, lambda: app.state.material_factory(None))
    return app
