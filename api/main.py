"""
Phase 7: FastAPI service — full pipeline with transparency reporting API.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from adversarial_safeguards.bundle import load_serving_bundle
from adversarial_safeguards.config import CIFAR_CLASSES, CIFAR_MEAN, CIFAR_STD, INPUT_SIZE
from adversarial_safeguards.inference_pipeline import RobustInferencePipeline

REPORTS: dict[str, dict[str, Any]] = {}
MAX_REPORTS = 500


def _tensor_from_upload(im: Image.Image) -> torch.Tensor:
    from torchvision import transforms

    im = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    t = transforms.ToTensor()(im)
    t = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)(t)
    return t.unsqueeze(0)


def create_app() -> FastAPI:
    app = FastAPI(title="Adversarial Safeguards API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bundle_path = Path(os.getenv("SERVING_BUNDLE", "./artifacts/baseline_bundle.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_defense = os.getenv("USE_INPUT_DEFENSE", "1") not in ("0", "false", "False")

    if not bundle_path.is_file():
        app.state.pipeline = None
        app.state.load_error = f"Missing bundle: {bundle_path.resolve()}"
        app.state.bundle_meta = {}
    else:
        model, det, meta = load_serving_bundle(bundle_path, device)
        log_path = Path(os.getenv("REQUEST_LOG_PATH", "./logs/requests.jsonl"))
        app.state.pipeline = RobustInferencePipeline(
            model=model,
            detector_state=det,
            device=device,
            use_input_defense=use_defense,
            model_name=str(meta.get("kind", "cifar_cnn")),
            model_version=str(meta.get("model_version", "1.0.0")),
            log_path=log_path,
        )
        app.state.load_error = None
        app.state.bundle_meta = meta

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if app.state.pipeline else "degraded",
            "device": str(device),
            "bundle": str(bundle_path.resolve()),
            "error": app.state.load_error,
            "classes": list(CIFAR_CLASSES),
            "bundle_meta": getattr(app.state, "bundle_meta", {}),
        }

    @app.post("/v1/predict")
    async def predict(file: UploadFile = File(...), include_gradcam: bool = True) -> dict[str, Any]:
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail=app.state.load_error)
        raw = await file.read()
        try:
            im = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e
        x = _tensor_from_upload(im)
        report, extras = app.state.pipeline.run(x, include_gradcam=include_gradcam)
        payload = report.to_json_dict()
        rid = payload["request_id"]
        REPORTS[rid] = payload
        if len(REPORTS) > MAX_REPORTS:
            for k in list(REPORTS.keys())[: len(REPORTS) - MAX_REPORTS]:
                del REPORTS[k]
        return {"transparency": payload, "extras": extras}

    @app.get("/v1/transparency/{request_id}")
    def transparency(request_id: str) -> dict[str, Any]:
        if request_id not in REPORTS:
            raise HTTPException(status_code=404, detail="Unknown request_id")
        return REPORTS[request_id]

    @app.get("/v1/transparency-report/latest")
    def latest() -> dict[str, Any]:
        if not REPORTS:
            raise HTTPException(status_code=404, detail="No reports yet")
        last = list(REPORTS.keys())[-1]
        return REPORTS[last]

    return app


app = create_app()
