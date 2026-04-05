from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    """Lightweight CNN for CIFAR-10 (fast training for research / demos)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def cam_layer(self) -> nn.Module:
        """Last conv block for Grad-CAM."""
        return self.conv3


class DistilledCifarCNN(CifarCNN):
    """Same architecture; used as student in defensive distillation (temperature at train time)."""

    def forward_with_temperature(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        logits = self.forward(x)
        return logits / max(temperature, 1e-6)
