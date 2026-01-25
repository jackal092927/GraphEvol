from __future__ import annotations
import os, json, argparse, pickle, math
import numpy as np
from scipy.stats import kendalltau

import torch

from ..models.mlp import MLPClassifier
from ..models.train import load_checkpoint, device_auto
from ..data.split_io import load_split, apply_split
from ..features.featurize import featurize_hatF

# reuse helpers from your annealing module
from ..search.annealing import proposal_swap, order_to_tau, temperature_schedule


def tau_to_order(tau: np.ndarray) -> np.ndarray:
    return np.argsort(tau).astype(int)


@torch.no_grad()
def score_tau_consistent(model: MLPClassifier, G, tau: np.ndarray, pi_cfg: dict, device) -> tuple[float, float]:
    """
    ORACLE inference scoring: score([tau, F(tau)]) where F(tau) is computed from the given tau.
    """
    F = featurize_hatF(
        G, tau,
        bandwidth=float(pi_cfg["bandwidth"]),
        resolution=(int(pi_cfg["resolution"][0]), int(pi_cfg["resolution"][1])),
        im_range=(float(pi_cfg["im_range"][0]), float(pi_cfg["im_range"][1]), float(pi_cfg["im_range"][2]), float(pi_cfg["im_range"][3])),
        dims=tuple(int(x) for x in pi_cfg["dims"]),
    ).astype(np.float32)

    x = np.concatenate([tau.astype(np.float32), F], axis=0)[None, :]  # (1, d)
    Xt = torch.from_numpy(x).to(device)
    logit = float(model(Xt).detach().cpu().numpy().reshape(-1)[0])
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return logit, prob


def simulated_annealing_ordering_oracle(
    model: MLPClassifier,
    G,
    init_order: np.ndarray,
    *,
    pi_cfg: dict,
    steps: int = 3000,
    T0: float = 1.0,
    Tend: float = 1e-3,
    seed: int = 0,
    fixed_prefix_len: int = 0,
    device=None,
):
    rng = np.random.default_rng(seed)
    cur_order = init_order.astype(int).copy()

    n = int(cur_order.shape[0])
    k = int(fixed_prefix_len)
    if k < 0: k = 0
    if k > n: k = n

    cur_tau = order_to_tau(cur_order)
    cur_logit, cur_prob = score_tau_consistent(model, G, cur_tau, pi_cfg, device)

    best_order = cur_order.copy()
    best_logit, best_prob = cur_logit, cur_prob
    hist = {"cur_logit": [], "best_logit": [], "T": []}

    for t in range(steps):
        T = temperature_schedule(T0, Tend, t, steps)

        prop_order = proposal_swap(cur_order, rng, fixed_prefix_len=k)
        prop_tau = order_to_tau(prop_order)
        prop_logit, prop_prob = score_tau_consistent(model, G, prop_tau, pi_cfg, device)

        delta = prop_logit - cur_logit
        accept = False
        if delta >= 0:
            accept = True
        elif T > 0:
            accept = (rng.random() < math.exp(delta / T))

        if accept:
            cur_order = prop_order
            cur_logit, cur_prob = prop_logit, prop_prob

        if cur_logit > best_logit:
            best_order = cur_order.copy()
            best_logit, best_prob = cur_logit, cur_prob

        hist["cur_logit"].append(float(cur_logit))
        hist["best_logit"].append(float(best_logit))
        hist["T"].append(float(T))

    best = {
        "order": best_order.astype(int).tolist(),
        "tau": order_to_tau(best_order).astype(float).tolist(),
        "logit": float(best_logit),
        "prob": float(best_prob),
    }
    return best, hist


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
    target = (train_inst + val_inst) if args.which == "trainval" else test_inst

    # load pi config from meta.json colocated with instances.pkl
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

    kendalls = []
    out_lines = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for idx, inst in enumerate(target):
            n = inst.graph.number_of_nodes()
            G = inst.graph

            init_order = tau_to_order(np.array(inst.tau_hat, dtype=np.float64))
            best, _hist = simulated_annealing_ordering_oracle(
                model, G, init_order,
                pi_cfg=pi_cfg,
                steps=args.steps, T0=args.T0, Tend=args.Tend, seed=args.seed + idx,
                fixed_prefix_len=0,
                device=device,
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
                "oracle_note": "scored with (tau, F(tau))",
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
