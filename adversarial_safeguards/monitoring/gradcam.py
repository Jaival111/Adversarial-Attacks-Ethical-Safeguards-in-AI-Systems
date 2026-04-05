from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _inp, out) -> None:
        self.activations = out.detach()

    def _save_grad(self, _m, _gi, go) -> None:
        self.gradients = go[0].detach()

    def remove_hooks(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def compute(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, int]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.squeeze(), class_idx


def cam_to_heatmap_rgba(cam_hw: torch.Tensor, colormap: str = "viridis") -> list[list[list[float]]]:
    """Return HxWx4 RGBA in [0,1] for JSON/API (matplotlib optional)."""
    import numpy as np

    arr = cam_hw.detach().float().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    try:
        import matplotlib

        cmap = matplotlib.colormaps[colormap]
        rgba = cmap(arr / 255.0)
        return rgba[..., :4].tolist()
    except Exception:
        # Grayscale fallback
        h, w = arr.shape
        out = []
        for i in range(h):
            row = []
            for j in range(w):
                v = arr[i, j] / 255.0
                row.append([v, v, v, 1.0])
            out.append(row)
        return out
