from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from adversarial_safeguards.detection.statistical import DetectorState, detector_state_from_dict, detector_state_to_dict
from adversarial_safeguards.models.cifar_cnn import CifarCNN


def save_serving_bundle(
    path: Path | str,
    model: CifarCNN,
    detector_state: DetectorState,
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "detector": detector_state_to_dict(detector_state),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_serving_bundle(path: Path | str, device: torch.device) -> tuple[CifarCNN, DetectorState, dict[str, Any]]:
    path = Path(path)
    # Load weights on CPU first so state_dict always applies cleanly, then move the full module.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = CifarCNN()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    det = detector_state_from_dict(ckpt["detector"])
    meta = ckpt.get("meta") or {}
    return model, det, meta
