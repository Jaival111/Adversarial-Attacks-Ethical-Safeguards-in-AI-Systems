from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def denormalize_cifar_tensor(x: torch.Tensor, mean, std, clamp01: bool = True) -> torch.Tensor:
    """x: (N,3,H,W) normalized."""
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    out = x * s + m
    if clamp01:
        out = out.clamp(0, 1)
    return out


def normalize_01_tensor(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


def jpeg_compress_tensor_batch(
    x_norm: torch.Tensor,
    mean,
    std,
    quality: int = 75,
) -> torch.Tensor:
    """Simulate JPEG via PIL on denormalized uint8 images; reduces adversarial high-frequency noise."""
    x01 = denormalize_cifar_tensor(x_norm, mean, std, clamp01=True)
    out = torch.empty_like(x_norm)
    for i in range(x_norm.shape[0]):
        arr = (x01[i].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        rec = np.array(Image.open(buf)).astype(np.float32) / 255.0
        t = torch.from_numpy(rec.transpose(2, 0, 1)).to(x_norm.device)
        out[i] = normalize_01_tensor(t.unsqueeze(0), mean, std).squeeze(0)
    return out


def gaussian_smooth(x: torch.Tensor, kernel_size: int = 3, sigma: float = 0.8) -> torch.Tensor:
    """Light spatial smoothing as an additional input transformation."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k2d = (g[:, None] @ g[None, :]).view(1, 1, kernel_size, kernel_size)
    k = k2d.expand(3, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=3)


def defense_input_pipeline(x: torch.Tensor, mean, std, jpeg_quality: int = 75, smooth: bool = True) -> torch.Tensor:
    x = jpeg_compress_tensor_batch(x, mean, std, quality=jpeg_quality)
    if smooth:
        x = gaussian_smooth(x)
    return x
