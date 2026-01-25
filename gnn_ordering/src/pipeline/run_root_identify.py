from __future__ import annotations

import os, json, argparse, pickle
import numpy as np

from ..data.split_io import load_split, apply_split
from ..models.mlp import MLPClassifier
from ..models.train import load_checkpoint, device_auto
from ..search.annealing import simulated_annealing_ordering


def tau_to_order(tau: np.ndarray) -> np.ndarray:
    return np.argsort(tau).astype(int)


def make_init_order_with_root(base_order: np.ndarray, root: int) -> np.ndarray:
    # base_order is a permutation of 0..n-1
    rest = [x for x in base_order.tolist() if x != root]
    return np.array([root] + rest, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--instances_pkl", required=True)
    ap.add_argument("--split_path", required=True)

    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_eval", required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--T0", type=float, default=1.0)
    ap.add_argument("--Tend", type=float, default=1e-3)

    # optional speed knobs
    ap.add_argument("--roots", type=str, default="all",
                    help="all or comma-separated list of roots to consider (e.g., '0,1,2')")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_eval), exist_ok=True)

    with open(args.instances_pkl, "rb") as f:
        instances = pickle.load(f)

    split = load_split(args.split_path)
    _train, _val, test_inst = apply_split(instances, split)

    # load model
    device = device_auto()
    obj = load_checkpoint(args.ckpt, device=device)
    model = MLPClassifier(in_dim=int(obj["meta"]["in_dim"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    per_graph = []

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for gi, inst in enumerate(test_inst):
            n = inst.graph.number_of_nodes()
            F_obs = np.array(inst.F_hat, dtype=np.float32)

            # define true root as earliest in tau_true
            order_true = tau_to_order(np.array(inst.tau_true, dtype=np.float64))
            root_true = int(order_true[0])

            base_order = tau_to_order(np.array(inst.tau_hat, dtype=np.float64))

            if args.roots == "all":
                roots = list(range(n))
            else:
                roots = [int(x) for x in args.roots.split(",") if x.strip() != ""]

            root_scores = []
            best_root = None
            best_logit = None
            best_prob = None
            best_order = None

            # For each candidate root, run constrained annealing:
            for r in roots:
                init_order = make_init_order_with_root(base_order, r)

                best, _hist = simulated_annealing_ordering(
                    model, F_obs, init_order,
                    steps=args.steps, T0=args.T0, Tend=args.Tend,
                    seed=args.seed + gi * 1000 + r,
                    fixed_prefix_len=1,  # <-- root locked at position 0
                )

                logit = float(best["logit"])
                prob = float(best["prob"])
                order = np.array(best["order"], dtype=int)

                root_scores.append({
                    "root": int(r),
                    "best_logit": logit,
                    "best_prob": prob,
                })

                if best_root is None or logit > best_logit:
                    best_root = int(r)
                    best_logit = logit
                    best_prob = prob
                    best_order = order.tolist()

            pred = int(best_root)
            ok = (pred == root_true)
            total += 1
            correct += int(ok)

            rec = {
                "graph_id": inst.graph_id,
                "root_true": root_true,
                "root_pred": pred,
                "root_pred_best_logit": best_logit,
                "root_pred_best_prob": best_prob,
                "order_mle_given_root_pred": best_order,
                "root_scores": root_scores,
            }
            fout.write(json.dumps(rec) + "\n")
            per_graph.append(ok)

    eval_obj = {
        "num_graphs": total,
        "root_acc": float(correct / total) if total > 0 else None,
    }
    with open(args.out_eval, "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=2)

    print("Wrote:", args.out_jsonl)
    print("Eval :", args.out_eval)
    print(eval_obj)


if __name__ == "__main__":
    main()
