from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    y = y.float()
    tp = ((preds == 1) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def train_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if device is None:
        device = device_auto()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.from_numpy(X_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)

    hist = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_state = None
    best_val = float("inf")
    bad = 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = loss_fn(val_logits, yva)
            val_metrics = compute_metrics(val_logits, yva)

        hist["train_loss"].append(float(loss.item()))
        hist["val_loss"].append(float(val_loss.item()))
        hist["val_metrics"].append(val_metrics)

        if float(val_loss.item()) < best_val - 1e-6:
            best_val = float(val_loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_val_loss": best_val,
        "epochs_ran": len(hist["train_loss"]),
        "history": hist,
        "device": str(device),
    }
    return model, info

def save_checkpoint(path: str, model: nn.Module, meta: Dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)

def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    if device is None:
        device = device_auto()
    return torch.load(path, map_location=device)
