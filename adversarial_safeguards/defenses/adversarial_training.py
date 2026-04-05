from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adversarial_safeguards.attacks.pgd_fgsm import pgd_attack


def adversarial_loss_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    pgd_steps: int,
    clean_weight: float = 0.5,
) -> torch.Tensor:
    x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=pgd_steps, random_start=True)
    logits_c = model(x)
    logits_a = model(x_adv)
    loss_c = F.cross_entropy(logits_c, y)
    loss_a = F.cross_entropy(logits_a, y)
    return clean_weight * loss_c + (1.0 - clean_weight) * loss_a
