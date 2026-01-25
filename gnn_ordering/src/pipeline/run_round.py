# src/pipeline/run_round.py
from __future__ import annotations

import os
import json
import argparse
import subprocess
from typing import Dict, Any


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _assert_not_overwrite(paths: list[str]) -> None:
    for p in paths:
        if os.path.exists(p):
            # if directory exists and not empty, treat as occupied
            if os.path.isdir(p) and os.listdir(p):
                raise RuntimeError(
                    f"[SAFETY] Refusing to overwrite existing directory: {p}\n"
                    f"Pass --overwrite to allow overwriting this round."
                )
            # if file exists, also refuse
            if os.path.isfile(p):
                raise RuntimeError(
                    f"[SAFETY] Refusing to overwrite existing file: {p}\n"
                    f"Pass --overwrite to allow overwriting this round."
                )


def _merge_jsonl_unique(out_path: str, in_paths: list[str]) -> None:
    """
    Merge multiple jsonl files (each line is a JSON object with key 'graph_id')
    into one, keeping the first occurrence of each graph_id.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen: Dict[str, Any] = {}

    for ip in in_paths:
        with open(ip, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                gid = obj["graph_id"]
                if gid not in seen:
                    seen[gid] = obj

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, obj in seen.items():
            f.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()

    # round control
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs for this round")

    # dataset params
    ap.add_argument("--n_nodes", type=int, default=30)
    ap.add_argument("--num_graphs", type=int, default=120)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--er_prob", type=float, default=0.05)

    # tau_hat source for THIS round's dataset featurization
    ap.add_argument("--tau_hat_source", choices=["degree", "file"], required=True)
    ap.add_argument("--tau_hat_path", type=str, default="")

    # fixed split path (constant across rounds)
    ap.add_argument("--split_path", type=str, required=True)

    # train params
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # annealing params
    ap.add_argument("--anneal_steps", type=int, default=3000)
    ap.add_argument("--anneal_T0", type=float, default=1.0)
    ap.add_argument("--anneal_Tend", type=float, default=1e-3)

    args = ap.parse_args()

    # output locations for this round (round-specific, so round1 won't touch round0)
    out_dataset = f"outputs/datasets/round{args.round}"
    out_ckpt = f"outputs/checkpoints/round{args.round}"
    out_order_dir = f"outputs/orderings/round{args.round}"
    out_eval_dir = f"outputs/evals/round{args.round}"

    trainval_jsonl = os.path.join(out_order_dir, "tau_hat_hat_trainval.jsonl")
    test_jsonl = os.path.join(out_order_dir, "tau_hat_hat_test.jsonl")
    all_jsonl = os.path.join(out_order_dir, "tau_hat_hat_all.jsonl")  # NEW (trainval+test)

    trainval_eval = os.path.join(out_eval_dir, "trainval_eval.json")
    test_eval = os.path.join(out_eval_dir, "test_eval.json")

    if not args.overwrite:
        _assert_not_overwrite([out_dataset, out_ckpt, out_order_dir, out_eval_dir])

    os.makedirs(out_order_dir, exist_ok=True)
    os.makedirs(out_eval_dir, exist_ok=True)

    # ---------- (1) Make dataset ----------
    cmd = [
        "python", "-m", "src.pipeline.make_dataset",
        "--out_dir", out_dataset,
        "--n_nodes", str(args.n_nodes),
        "--num_graphs", str(args.num_graphs),
        "--m", str(args.m),
        "--er_prob", str(args.er_prob),
        "--seed", str(args.seed),
        "--tau_hat_source", args.tau_hat_source,
        "--split_path", args.split_path,
    ]
    if args.tau_hat_source == "file":
        if not args.tau_hat_path:
            raise ValueError("--tau_hat_path required when --tau_hat_source=file")
        cmd += ["--tau_hat_path", args.tau_hat_path]
    run(cmd)

    # ---------- (2) Train MLP (train/val only) ----------
    run([
        "python", "-m", "src.pipeline.run_train",
        "--data_dir", out_dataset,
        "--out_dir", out_ckpt,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
    ])

    ckpt_path = os.path.join(out_ckpt, "model.pt")
    inst_pkl = os.path.join(out_dataset, "instances.pkl")

    # ---------- (3) Infer on TRAIN+VAL (for next-round updates) ----------
    run([
        "python", "-m", "src.pipeline.run_anneal_batch",
        "--ckpt", ckpt_path,
        "--instances_pkl", inst_pkl,
        "--split_path", args.split_path,
        "--which", "trainval",
        "--out_jsonl", trainval_jsonl,
        "--out_eval", trainval_eval,
        "--seed", str(args.seed),
        "--steps", str(args.anneal_steps),
        "--T0", str(args.anneal_T0),
        "--Tend", str(args.anneal_Tend),
    ])

    # ---------- (4) Infer on TEST (save predictions + eval) ----------
    run([
        "python", "-m", "src.pipeline.run_anneal_batch",
        "--ckpt", ckpt_path,
        "--instances_pkl", inst_pkl,
        "--split_path", args.split_path,
        "--which", "test",
        "--out_jsonl", test_jsonl,
        "--out_eval", test_eval,
        "--seed", str(args.seed),
        "--steps", str(args.anneal_steps),
        "--T0", str(args.anneal_T0),
        "--Tend", str(args.anneal_Tend),
    ])

    root_jsonl = os.path.join(out_order_dir, "root_pred_test.jsonl")
    root_eval  = os.path.join(out_eval_dir, "root_eval_test.json")

    run([
    "python","-m","src.pipeline.run_root_identify",
    "--ckpt", ckpt_path,
    "--instances_pkl", inst_pkl,
    "--split_path", args.split_path,
    "--out_jsonl", root_jsonl,
    "--out_eval", root_eval,
    "--seed", str(args.seed),
    "--steps", str(args.anneal_steps),
    "--T0", str(args.anneal_T0),
    "--Tend", str(args.anneal_Tend),
    ])


    # ---------- (5) NEW: merge trainval+test into ALL (for same-round featurization next round) ----------
    # Next round featurization can use this ALL file so test isn't stuck with degree.
    _merge_jsonl_unique(all_jsonl, [trainval_jsonl, test_jsonl])

    print("\n===== ROUND COMPLETE =====")
    print("dataset_dir:", out_dataset)
    print("checkpoint :", ckpt_path)
    print("trainval inferred (for analysis / debugging):", trainval_jsonl)
    print("test inferred (saved predictions):          ", test_jsonl)
    print("ALL inferred (NEXT ROUND FEATURIZE INPUT):  ", all_jsonl)
    print("test eval:", test_eval)
    print("==========================\n")


if __name__ == "__main__":
    main()

