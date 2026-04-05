from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RiskTier = Literal["high", "medium", "low"]


@dataclass
class TransparencyReport:
    """Accountable, machine-readable disclosure for a single inference request."""

    request_id: str
    model_name: str
    model_version: str
    risk_tier: RiskTier
    risk_rationale: list[str]
    detector: dict[str, Any]
    prediction: dict[str, Any]
    monitoring: dict[str, Any]
    ethics: dict[str, Any]
    limitations: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def tier_from_signals(
    detector_flags: dict[str, bool],
    confidence: float,
    kl_to_ref: float,
    kl_threshold_soft: float,
) -> tuple[RiskTier, list[str]]:
    """Map heuristic detector outputs to accountable risk tiers."""
    reasons: list[str] = []
    n_flags = sum(1 for v in detector_flags.values() if v)

    if detector_flags.get("low_confidence"):
        reasons.append("Max class probability is unusually low relative to clean calibration.")
    if detector_flags.get("distribution_shift_kl"):
        reasons.append("Output distribution diverges (KL) from the clean validation prior.")
    if detector_flags.get("confidence_instability"):
        reasons.append("Large confidence change under light input noise (possible adversarial sensitivity).")

    if kl_to_ref > kl_threshold_soft * 2.0:
        reasons.append("KL divergence exceeds twice the clean 95th percentile.")

    if n_flags >= 2 or confidence < 0.35:
        return "high", reasons
    if n_flags >= 1 or confidence < 0.55 or kl_to_ref > kl_threshold_soft * 1.5:
        return "medium", reasons
    return "low", reasons


DEFAULT_LIMITATIONS = [
    "Statistical detectors are not certified guarantees; adaptive attackers may evade them.",
    "CIFAR-10 models do not generalize to arbitrary natural images uploaded via the UI.",
    "Reports support governance and incident review; they are not a substitute for formal security audits.",
]
