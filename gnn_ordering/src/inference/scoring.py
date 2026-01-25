from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from ..models.train import device_auto

@torch.no_grad()
def score_taus_batch(model, taus: np.ndarray, F_obs: np.ndarray, *, device: torch.device | None = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = device_auto()
    model = model.to(device)
    model.eval()
    B, _ = taus.shape
    F = np.broadcast_to(F_obs[None, :], (B, F_obs.shape[0]))
    X = np.concatenate([taus.astype(np.float32), F.astype(np.float32)], axis=1)
    Xt = torch.from_numpy(X).to(device)
    logits = model(Xt).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs
