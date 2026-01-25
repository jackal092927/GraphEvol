from __future__ import annotations
import os, json, argparse, pickle
import numpy as np

from ..utils_seed import seed_all
from ..data.graph_gen import generate_pa_graph, ordering_from_pa
from ..data.dataset import GraphInstance, make_pos_neg_samples, samples_to_arrays
from ..data.split import split_by_graph
from ..features.ordering import DegreeOrderingProvider, FileOrderingProvider
from ..features.featurize import featurize_hatF


def build_graph_id(i: int, *, seed: int, n_nodes: int, m: int, er_prob: float) -> str:
    return f"pa_seed{seed+i}_n{n_nodes}_m{m}_er{er_prob}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_nodes", type=int, default=30)
    ap.add_argument("--num_graphs", type=int, default=120)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--er_prob", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    # keep this for init_order; F_hat will NOT use tau_hat in oracle mode
    ap.add_argument("--tau_hat_source", choices=["degree", "file"], default="degree")
    ap.add_argument("--tau_hat_path", type=str, default="")

    ap.add_argument("--pi_bandwidth", type=float, default=0.05)
    ap.add_argument("--pi_res", type=int, nargs=2, default=[12, 12])
    ap.add_argument("--pi_range", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0])
    ap.add_argument("--pi_dims", type=int, nargs="+", default=[0, 1])

    ap.add_argument("--split_path", type=str, default="")

    args = ap.parse_args()
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    deg_provider = DegreeOrderingProvider()
    if args.tau_hat_source == "degree":
        provider = deg_provider
    else:
        if not args.tau_hat_path:
            raise ValueError("--tau_hat_path required for tau_hat_source=file")
        provider = FileOrderingProvider(path=args.tau_hat_path, fallback=deg_provider)

    instances = []
    for i in range(args.num_graphs):
        gid = build_graph_id(i, seed=args.seed, n_nodes=args.n_nodes, m=args.m, er_prob=args.er_prob)
        G = generate_pa_graph(args.n_nodes, args.m, args.er_prob, seed=args.seed + i)
        tau_true = ordering_from_pa(args.n_nodes)

        # keep tau_hat ONLY for init in inference; does not affect F_hat in oracle dataset
        tau_hat = provider.get(G, gid).astype(np.float64)

        # ORACLE: F_hat is computed from tau_true (not tau_hat)
        F_hat = featurize_hatF(
            G, tau_true,
            bandwidth=args.pi_bandwidth,
            resolution=(args.pi_res[0], args.pi_res[1]),
            im_range=(args.pi_range[0], args.pi_range[1], args.pi_range[2], args.pi_range[3]),
            dims=tuple(args.pi_dims),
        )

        instances.append(GraphInstance(graph_id=gid, graph=G, tau_true=tau_true, tau_hat=tau_hat, F_hat=F_hat))

    from ..data.split_io import save_split, load_split, apply_split

    # fixed split: create once, reuse forever
    if args.split_path and os.path.exists(args.split_path):
        split = load_split(args.split_path)
        train_inst, val_inst, test_inst = apply_split(instances, split)
    else:
        train_inst, val_inst, test_inst = split_by_graph(instances, args.train_ratio, args.val_ratio, args.seed)
        if args.split_path:
            split = {
                "train_graph_ids": [x.graph_id for x in train_inst],
                "val_graph_ids": [x.graph_id for x in val_inst],
                "test_graph_ids": [x.graph_id for x in test_inst],
            }
            save_split(args.split_path, split)

    rng = np.random.default_rng(args.seed + 12345)

    def build_split(insts):
        ss = []
        for inst in insts:
            # IMPORTANT: make_pos_neg_samples will create:
            #   pos: (tau_true, F_hat) where F_hat = F(tau_true)
            #   neg: (perm_tau, F_hat) where F_hat is STILL F(tau_true)  <-- mismatch you want
            ss.extend(make_pos_neg_samples(inst, rng))
        return samples_to_arrays(ss)

    Xtr, ytr, gid_tr = build_split(train_inst)
    Xva, yva, gid_va = build_split(val_inst)
    Xte, yte, gid_te = build_split(test_inst)

    np.savez(os.path.join(args.out_dir, "oracle_train.npz"), X=Xtr, y=ytr, graph_id=np.array(gid_tr, dtype=object))
    np.savez(os.path.join(args.out_dir, "oracle_val.npz"), X=Xva, y=yva, graph_id=np.array(gid_va, dtype=object))
    np.savez(os.path.join(args.out_dir, "oracle_test.npz"), X=Xte, y=yte, graph_id=np.array(gid_te, dtype=object))

    with open(os.path.join(args.out_dir, "instances.pkl"), "wb") as f:
        pickle.dump(instances, f)

    meta = {
        "oracle_mode": {
            "train": "pos=(tau_true, F(tau_true)), neg=(perm_tau, F(tau_true))  [mismatched]",
            "infer": "score=(tau, F(tau))  [consistent]",
        },
        "data": {k: getattr(args, k) for k in ["n_nodes", "num_graphs", "m", "er_prob", "seed", "train_ratio", "val_ratio"]},
        "pi": {"bandwidth": args.pi_bandwidth, "resolution": args.pi_res, "im_range": args.pi_range, "dims": args.pi_dims},
        "tau_hat_source": args.tau_hat_source,
        "tau_hat_path": args.tau_hat_path,
        "counts": {"train_graphs": len(train_inst), "val_graphs": len(val_inst), "test_graphs": len(test_inst)},
        "dims": {"X_dim": int(Xtr.shape[1]), "F_dim": int(instances[0].F_hat.shape[0]), "n": int(args.n_nodes)},
        "split_path": args.split_path,
        "npz_names": {"train": "oracle_train.npz", "val": "oracle_val.npz", "test": "oracle_test.npz"},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved ORACLE dataset to {args.out_dir}")
    print(meta)


if __name__ == "__main__":
    main()
