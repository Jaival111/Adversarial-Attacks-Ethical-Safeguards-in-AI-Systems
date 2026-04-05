from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from adversarial_safeguards.config import CIFAR_CLASSES, CIFAR_MEAN, CIFAR_STD
from adversarial_safeguards.defenses.input_transform import defense_input_pipeline
from adversarial_safeguards.detection.statistical import DetectorState, score_sample
from adversarial_safeguards.models.cifar_cnn import CifarCNN
from adversarial_safeguards.monitoring.gradcam import GradCAM, cam_to_heatmap_rgba
from adversarial_safeguards.monitoring.logger import JsonlLogger, RequestLogEntry, new_request_id, utc_now_iso
from adversarial_safeguards.risk.framework import DEFAULT_LIMITATIONS, TransparencyReport, tier_from_signals


class RobustInferencePipeline:
    """Input → detector → (optional input defense) → model → monitoring → risk report."""

    def __init__(
        self,
        model: CifarCNN,
        detector_state: DetectorState,
        device: torch.device,
        use_input_defense: bool = True,
        model_name: str = "cifar_cnn",
        model_version: str = "1.0.0",
        log_path: Path | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.detector_state = detector_state
        self.device = device
        self.use_input_defense = use_input_defense
        self.model_name = model_name
        self.model_version = model_version
        self.logger = JsonlLogger(log_path)

    @torch.no_grad()
    def _predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(
        self,
        x: torch.Tensor,
        request_id: str | None = None,
        include_gradcam: bool = True,
    ) -> tuple[TransparencyReport, dict[str, Any]]:
        rid = request_id or new_request_id()
        x = x.to(self.device, non_blocking=True)
        det = score_sample(self.model, x, self.detector_state, self.device)

        x_in = x
        if self.use_input_defense:
            x_in = defense_input_pipeline(x, CIFAR_MEAN, CIFAR_STD)

        logits = self._predict_logits(x_in)
        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        conf = float(probs.max().item())

        tier, rationale = tier_from_signals(
            det.flags,
            confidence=conf,
            kl_to_ref=det.kl_to_ref,
            kl_threshold_soft=self.detector_state.kl_clean_p95,
        )

        cam_payload: dict[str, Any] | None = None
        if include_gradcam:
            x_cam = x_in.detach().clone().requires_grad_(True)
            gc = GradCAM(self.model, self.model.cam_layer())
            try:
                cam, cam_cls = gc.compute(x_cam, class_idx=pred)
                cam_payload = {
                    "class_index": cam_cls,
                    "heatmap_rgba": cam_to_heatmap_rgba(cam.squeeze(0)),
                }
            finally:
                gc.remove_hooks()

        notes: list[str] = []
        if self.use_input_defense:
            notes.append("Input JPEG + mild Gaussian smoothing applied before inference.")
        if any(det.flags.values()):
            notes.append("Detector raised one or more statistical flags; see transparency report.")

        report = TransparencyReport(
            request_id=rid,
            model_name=self.model_name,
            model_version=self.model_version,
            risk_tier=tier,
            risk_rationale=rationale,
            detector={
                "scores": {
                    "max_prob": det.max_prob,
                    "entropy": det.entropy,
                    "kl_to_ref_mean": det.kl_to_ref,
                    "confidence_delta_noisy": det.confidence_delta_noisy,
                },
                "flags": det.flags,
                "calibration": {
                    "clean_conf_p05": self.detector_state.clean_conf_p05,
                    "kl_clean_p95": self.detector_state.kl_clean_p95,
                },
            },
            prediction={
                "class_index": pred,
                "class_name": CIFAR_CLASSES[pred],
                "confidence": conf,
            },
            monitoring={
                "grad_cam": cam_payload,
                "input_defense_enabled": self.use_input_defense,
                "notes": notes,
            },
            ethics={
                "purpose": "Demonstrate accountable logging and transparency for adversarial risk.",
                "human_review_recommended": tier != "low",
            },
            limitations=list(DEFAULT_LIMITATIONS),
        )

        self.logger.append(
            RequestLogEntry(
                request_id=rid,
                ts_iso=utc_now_iso(),
                predicted_class=pred,
                label_name=CIFAR_CLASSES[pred],
                confidence=conf,
                detector_flags=det.flags,
                detector_scores={
                    "max_prob": det.max_prob,
                    "entropy": det.entropy,
                    "kl_to_ref": det.kl_to_ref,
                    "confidence_delta_noisy": det.confidence_delta_noisy,
                },
                risk_tier=tier,
                input_anomaly_notes=notes,
            )
        )

        extras = {
            "logits": logits.detach().cpu().tolist(),
            "defended_input_tensor_shape": list(x_in.shape),
        }
        return report, extras
