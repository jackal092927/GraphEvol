from __future__ import annotations
import os, json, argparse, pickle
import numpy as np
from scipy.stats import kendalltau

from ..models.mlp import MLPClassifier
from ..models.train import load_checkpoint, device_auto
from ..search.annealing import simulated_annealing_ordering
from ..data.split_io import load_split, apply_split

def tau_to_order(tau: np.ndarray) -> np.ndarray:
    return np.argsort(tau).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instances_pkl", required=True)
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--which", choices=["trainval", "test"], required=True)

    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_eval", required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T0", type=float, default=1.0)
    ap.add_argument("--Tend", type=float, default=1e-3)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_eval), exist_ok=True)

    with open(args.instances_pkl, "rb") as f:
        instances = pickle.load(f)

    split = load_split(args.split_path)
    train_inst, val_inst, test_inst = apply_split(instances, split)
    if args.which == "trainval":
        target = train_inst + val_inst
    else:
        target = test_inst

    device = device_auto()
    obj = load_checkpoint(args.ckpt, device=device)
    model = MLPClassifier(in_dim=int(obj["meta"]["in_dim"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device); model.eval()

    kendalls = []
    out_lines = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for idx, inst in enumerate(target):
            n = inst.graph.number_of_nodes()
            F_obs = np.array(inst.F_hat, dtype=np.float32)

            init_order = tau_to_order(np.array(inst.tau_hat, dtype=np.float64))
            best, _hist = simulated_annealing_ordering(
                model, F_obs, init_order,
                steps=args.steps, T0=args.T0, Tend=args.Tend, seed=args.seed + idx
            )

            order_pred = np.array(best["order"], dtype=int)
            order_true = tau_to_order(np.array(inst.tau_true, dtype=np.float64))

            # Kendall tau on rank vectors
            r_pred = np.empty(n, dtype=int); r_pred[order_pred] = np.arange(n)
            r_true = np.empty(n, dtype=int); r_true[order_true] = np.arange(n)
            kt = kendalltau(r_true, r_pred).correlation
            if kt is None:
                kt = 0.0
            kendalls.append(float(kt))

            rec = {
                "graph_id": inst.graph_id,
                "which": args.which,
                "order": order_pred.tolist(),
                "tau": best["tau"],
                "logit": best["logit"],
                "prob": best["prob"],
                "kendall_tau_vs_true": float(kt),
            }
            f.write(json.dumps(rec) + "\n")
            out_lines += 1

    eval_obj = {
        "which": args.which,
        "num_graphs": out_lines,
        "kendall_tau_mean": float(np.mean(kendalls)) if kendalls else None,
        "kendall_tau_std": float(np.std(kendalls)) if kendalls else None,
        "kendall_tau_min": float(np.min(kendalls)) if kendalls else None,
        "kendall_tau_max": float(np.max(kendalls)) if kendalls else None,
    }
    with open(args.out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=2)

    print("Wrote:", args.out_jsonl)
    print("Eval :", args.out_eval)
    print(eval_obj)

if __name__ == "__main__":
    main()
