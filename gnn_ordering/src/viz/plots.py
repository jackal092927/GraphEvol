from __future__ import annotations
from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def plot_training_curves(history: Dict[str, Any], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = [m["acc"] for m in history["val_metrics"]]

    plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    out_path2 = out_path.replace(".png", "_valacc.png")
    plt.figure()
    plt.plot(val_acc, label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=160)
    plt.close()

def plot_tau_ranks(tau: np.ndarray, out_path: str, title: str = "tau ranks") -> None:
    ensure_dir(os.path.dirname(out_path))
    order = np.argsort(tau)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(tau)), ranks)
    plt.title(title)
    plt.xlabel("node")
    plt.ylabel("rank (0=earliest)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_annealing_history(hist: Dict[str, Any], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.plot(hist["cur_logit"], label="cur logit")
    plt.plot(hist["best_logit"], label="best logit")
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("logit")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
