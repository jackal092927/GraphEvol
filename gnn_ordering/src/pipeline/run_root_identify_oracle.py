from __future__ import annotations
import os, json, argparse, pickle
import numpy as np

from ..models.mlp import MLPClassifier
from ..models.train import load_checkpoint, device_auto
from ..data.split_io import load_split, apply_split

from .run_anneal_batch_oracle import simulated_annealing_ordering_oracle, tau_to_order  # reuse


def make_init_order_with_root(base_order: np.ndarray, root: int) -> np.ndarray:
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

    ap.add_argument("--roots", type=str, default="all",
                    help="all or comma-separated list of roots to consider (e.g., '0,1,2')")

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_eval), exist_ok=True)

    with open(args.instances_pkl, "rb") as f:
        instances = pickle.load(f)

    split = load_split(args.split_path)
    _train, _val, test_inst = apply_split(instances, split)

    # load pi config from meta.json next to instances.pkl
    data_dir = os.path.dirname(args.instances_pkl)
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found next to instances.pkl: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    pi_cfg = meta["pi"]

    device = device_auto()
    obj = load_checkpoint(args.ckpt, device=device)
    model = MLPClassifier(in_dim=int(obj["meta"]["in_dim"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device); model.eval()

    correct = 0
    total = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for gi, inst in enumerate(test_inst):
            n = inst.graph.number_of_nodes()
            G = inst.graph

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

            for r in roots:
                init_order = make_init_order_with_root(base_order, r)

                best, _hist = simulated_annealing_ordering_oracle(
                    model, G, init_order,
                    pi_cfg=pi_cfg,
                    steps=args.steps, T0=args.T0, Tend=args.Tend,
                    seed=args.seed + gi * 1000 + r,
                    fixed_prefix_len=1,   # lock root
                    device=device,
                )

                logit = float(best["logit"])
                prob = float(best["prob"])
                order = list(map(int, best["order"]))

                root_scores.append({"root": int(r), "best_logit": logit, "best_prob": prob})

                if best_root is None or logit > best_logit:
                    best_root = int(r)
                    best_logit = logit
                    best_prob = prob
                    best_order = order

            # sort scores for easier inspection
            root_scores_sorted = sorted(root_scores, key=lambda x: x["best_logit"], reverse=True)

            pred = int(best_root)
            ok = (pred == root_true)
            total += 1
            correct += int(ok)

            rec = {
                "graph_id": inst.graph_id,
                "root_true": root_true,
                "root_pred": pred,
                "correct": bool(ok),

                "root_pred_best_logit": float(best_logit),
                "root_pred_best_prob": float(best_prob),
                "order_mle_given_root_pred": best_order,

                "root_profile_scores": root_scores_sorted,
                "oracle_note": "root selected by profile MLE: max_r max_{tau:tau[0]=r} score(tau,F(tau))",
            }
            fout.write(json.dumps(rec) + "\n")

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
