from __future__ import annotations
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
