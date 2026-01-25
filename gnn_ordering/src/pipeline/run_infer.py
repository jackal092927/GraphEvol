from __future__ import annotations
import os, json, argparse, pickle
import numpy as np
from ..models.mlp import MLPClassifier
from ..models.train import load_checkpoint, device_auto
from ..search.random_search import topk_orderings_random_search
from ..search.annealing import simulated_annealing_ordering
from ..viz.plots import plot_tau_ranks, plot_annealing_history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph_index", type=int, default=0)

    ap.add_argument("--do_random_search", type=int, default=1)
    ap.add_argument("--M", type=int, default=200000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)

    ap.add_argument("--do_anneal", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T0", type=float, default=1.0)
    ap.add_argument("--Tend", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = device_auto()

    with open(os.path.join(args.data_dir, "instances.pkl"), "rb") as f:
        instances = pickle.load(f)
    inst = instances[args.graph_index]

    obj = load_checkpoint(args.ckpt, device=device)
    model = MLPClassifier(in_dim=int(obj["meta"]["in_dim"]))
    model.load_state_dict(obj["state_dict"])
    model.to(device); model.eval()

    plot_tau_ranks(np.array(inst.tau_hat, dtype=np.float64),
                   os.path.join(args.out_dir, f"{inst.graph_id}_tauhat_ranks.png"),
                   title="tau_hat ranks")

    results = {"graph_id": inst.graph_id, "n": int(inst.graph.number_of_nodes())}

    if args.do_random_search:
        results["random_search_topk"] = topk_orderings_random_search(
            model, np.array(inst.F_hat), n_nodes=results["n"],
            M=args.M, K=args.K, batch_size=args.batch_size, seed=args.seed
        )

    if args.do_anneal:
        init_order = np.argsort(np.array(inst.tau_hat, dtype=np.float64))
        best, hist = simulated_annealing_ordering(
            model, np.array(inst.F_hat), init_order,
            steps=args.steps, T0=args.T0, Tend=args.Tend, seed=args.seed
        )
        results["anneal_best"] = best
        plot_annealing_history(hist, os.path.join(args.out_dir, f"{inst.graph_id}_anneal_history.png"))
        plot_tau_ranks(np.array(best["tau"], dtype=np.float64),
                       os.path.join(args.out_dir, f"{inst.graph_id}_anneal_best_ranks.png"),
                       title="anneal best ranks")

    with open(os.path.join(args.out_dir, f"{inst.graph_id}_infer_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote inference outputs under {args.out_dir}")

if __name__ == "__main__":
    main()
