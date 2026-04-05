from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DetectorState:
    """Calibrated on clean validation data."""

    ref_probs_mean: torch.Tensor  # (C,) prior over classes from clean val
    clean_conf_p05: float  # 5th percentile max prob on clean
    clean_conf_p95: float
    kl_clean_p95: float  # 95th percentile KL(p || ref_mean) on clean


def detector_state_to_dict(state: DetectorState) -> dict:
    return {
        "ref_probs_mean": state.ref_probs_mean.detach().cpu().tolist(),
        "clean_conf_p05": state.clean_conf_p05,
        "clean_conf_p95": state.clean_conf_p95,
        "kl_clean_p95": state.kl_clean_p95,
    }


def detector_state_from_dict(d: dict) -> DetectorState:
    return DetectorState(
        ref_probs_mean=torch.tensor(d["ref_probs_mean"], dtype=torch.float32),
        clean_conf_p05=float(d["clean_conf_p05"]),
        clean_conf_p95=float(d["clean_conf_p95"]),
        kl_clean_p95=float(d["kl_clean_p95"]),
    )


@torch.no_grad()
def collect_detector_state(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> DetectorState:
    model.eval()
    all_probs: list[torch.Tensor] = []
    kls: list[float] = []
    confs: list[float] = []
    count = 0
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        all_probs.append(p.cpu())
        confs.extend(p.max(dim=1).values.cpu().tolist())
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    stacked = torch.cat(all_probs, dim=0)
    ref_mean = stacked.mean(dim=0)
    ref_mean = ref_mean / ref_mean.sum().clamp_min(1e-8)
    # Re-scan for KL vs ref (cheap for demo subset)
    for pbatch in all_probs:
        for i in range(pbatch.shape[0]):
            pi = pbatch[i : i + 1].to(device).clamp_min(1e-10)
            ref = ref_mean.to(device).unsqueeze(0)
            kls.append(float(F.kl_div(pi.log(), ref, reduction="batchmean").item()))
    conf_arr = np.array(confs)
    kl_arr = np.array(kls)
    return DetectorState(
        ref_probs_mean=ref_mean.cpu(),
        clean_conf_p05=float(np.percentile(conf_arr, 5)),
        clean_conf_p95=float(np.percentile(conf_arr, 95)),
        kl_clean_p95=float(np.percentile(kl_arr, 95)),
    )


@dataclass
class DetectionScores:
    max_prob: float
    entropy: float
    kl_to_ref: float
    confidence_delta_noisy: float
    flags: dict[str, bool]


def score_sample(
    model: nn.Module,
    x: torch.Tensor,
    state: DetectorState,
    device: torch.device,
    noise_std: float = 0.02,
) -> DetectionScores:
    """Heuristic adversarial / OOD cues (not a certified detector)."""
    model.eval()
    x = x.to(device)
    logits = model(x)
    p = F.softmax(logits, dim=1)
    max_prob = float(p.max().item())
    entropy = float((-(p * (p.clamp_min(1e-8).log())).sum()).item())
    ref = state.ref_probs_mean.to(device).unsqueeze(0)
    p_safe = p.clamp_min(1e-10)
    kl = float(F.kl_div(p_safe.log(), ref, reduction="batchmean").item())

    noise = torch.randn_like(x) * noise_std
    logits2 = model((x + noise).clamp(-3, 3))
    p2 = F.softmax(logits2, dim=1)
    conf_delta = abs(float(p.max().item() - p2.max().item()))

    low_conf = max_prob < state.clean_conf_p05
    high_kl = kl > state.kl_clean_p95 * 1.5
    unstable = conf_delta > 0.25

    flags = {
        "low_confidence": low_conf,
        "distribution_shift_kl": high_kl,
        "confidence_instability": unstable,
    }
    return DetectionScores(
        max_prob=max_prob,
        entropy=entropy,
        kl_to_ref=kl,
        confidence_delta_noisy=conf_delta,
        flags=flags,
    )
