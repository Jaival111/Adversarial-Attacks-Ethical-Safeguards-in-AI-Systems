from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp_tensor(x: torch.Tensor, center: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, center + eps), center - eps)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + eps * grad.sign()
    return _clamp_tensor(x_adv, x, eps).detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Projected Gradient Descent (l_inf), same recipe as CleverHans / Madry et al."""
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = _clamp_tensor(x_adv, x, eps)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + alpha * grad.sign()).detach()
        x_adv = _clamp_tensor(x_adv, x, eps)
    return x_adv


def pgd_with_torchattacks(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
) -> torch.Tensor:
    """Optional wrapper around torchattacks.PGD for parity checks."""
    from torchattacks import PGD

    atk = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
    return atk(x, y)
