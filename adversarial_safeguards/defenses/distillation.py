from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_targets(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.log_softmax(logits / temperature, dim=1)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Hinton KD: alpha * KL(student || teacher) + (1-alpha) * CE(student, hard labels)."""
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce


@torch.no_grad()
def teacher_predict_logits(teacher: nn.Module, x: torch.Tensor, temperature: float) -> torch.Tensor:
    return teacher(x) / max(temperature, 1e-6)
