from __future__ import annotations
import os, json, argparse
import numpy as np
from ..utils_seed import seed_all
from ..models.mlp import MLPClassifier
from ..models.train import train_model, save_checkpoint
from ..viz.plots import plot_training_curves

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    Xtr, ytr = load_npz(os.path.join(args.data_dir, "train.npz"))
    Xva, yva = load_npz(os.path.join(args.data_dir, "val.npz"))

    model = MLPClassifier(in_dim=int(Xtr.shape[1]), hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout)
    model, info = train_model(
        model, Xtr, ytr, Xva, yva,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience
    )

    ckpt_path = os.path.join(args.out_dir, "model.pt")
    save_checkpoint(ckpt_path, model, meta={"train_info": info, "in_dim": int(Xtr.shape[1])})

    with open(os.path.join(args.out_dir, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    plot_training_curves(info["history"], os.path.join(args.out_dir, "training_curves.png"))
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Best val loss: {info['best_val_loss']}, epochs ran: {info['epochs_ran']}")

if __name__ == "__main__":
    main()
