#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis for annealing outputs (JSONL) — field names aligned to current pipeline.

Expected JSONL format (one line per test graph):
{
  "gid": int,
  "n_nodes": int,
  "pred_order": [int,...],
  "true_order": [int,...],
  "best_score": float,
  "accept_rate": float,
  "kendall_distance": int,
  "kendall_tau": float,                 # Kendall tau similarity in [-1,1]
  "pred_root": int,
  "true_root": int,
  "pred_root_true_rank": int,           # 0 means correct
  "root_rank_error_norm": float         # in [0,1]
}

What this script does:
1) For ONLY the first 20 graphs (sorted by gid): rank scatter plot true_rank vs pred_rank (y=x reference).
2) Summary stats: Kendall tau similarity (mean/median/min/max), root accuracy, root error norm, accept/score.
3) Histogram of Kendall tau similarity.

Usage:
  python analysis.py --jsonl outputs_exact_v0/anneal_results.jsonl --outdir analysis_out
  python analysis.py --jsonl ... --outdir ... --only_gids 1 5 12
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt


REQUIRED_KEYS = [
    "gid", "n_nodes", "pred_order", "true_order",
    "best_score", "accept_rate", "kendall_distance", "kendall_tau",
    "pred_root", "true_root", "pred_root_true_rank", "root_rank_error_norm"
]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no} in {path}: {e}") from e
    if not rows:
        raise RuntimeError(f"No records found in {path}")
    return rows


def validate_schema(records: List[Dict[str, Any]]) -> None:
    # Fail fast if keys missing; this avoids silent NaNs/empty plots.
    for i, r in enumerate(records[:5]):  # check first few
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise RuntimeError(
                f"Record {i} (gid={r.get('gid')}) missing keys: {missing}\n"
                f"Available keys: {sorted(list(r.keys()))}"
            )


def order_to_rank(order: List[int]) -> np.ndarray:
    """
    order: list of nodes in arrival order (earliest -> latest)
    returns rank[node] = position in order
    """
    n = len(order)
    rank = np.empty(n, dtype=int)
    for pos, node in enumerate(order):
        rank[int(node)] = pos
    return rank


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_rank_scatter(true_order: List[int], pred_order: List[int], gid: int, outpath: str, title_extra: str = "") -> None:
    true_rank = order_to_rank(true_order)
    pred_rank = order_to_rank(pred_order)

    x = true_rank
    y = pred_rank
    n = len(x)

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=18)
    plt.plot([0, n - 1], [0, n - 1], linewidth=1.0)  # 45-degree

    plt.xlim(-1, n)
    plt.ylim(-1, n)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.xlabel("True rank (0 = earliest)")
    plt.ylabel("Predicted rank (0 = earliest)")

    title = f"Rank scatter (gid={gid})"
    if title_extra:
        title += f" | {title_extra}"
    plt.title(title)

    ensure_dir(os.path.dirname(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ktau = np.array([r["kendall_tau"] for r in records], dtype=float)
    kdist = np.array([r["kendall_distance"] for r in records], dtype=float)
    accept = np.array([r["accept_rate"] for r in records], dtype=float)
    score = np.array([r["best_score"] for r in records], dtype=float)

    root_ok = np.array([(r["pred_root"] == r["true_root"]) for r in records], dtype=bool)
    root_err_norm = np.array([r["root_rank_error_norm"] for r in records], dtype=float)
    root_rank = np.array([r["pred_root_true_rank"] for r in records], dtype=float)

    out = {
        "num_graphs": int(len(records)),

        "kendall_tau_mean": float(np.mean(ktau)),
        "kendall_tau_median": float(np.median(ktau)),
        "kendall_tau_min": float(np.min(ktau)),
        "kendall_tau_max": float(np.max(ktau)),

        "kendall_distance_mean": float(np.mean(kdist)),
        "kendall_distance_median": float(np.median(kdist)),

        "root_accuracy": float(root_ok.mean()),
        "root_rank_mean": float(np.mean(root_rank)),
        "root_rank_median": float(np.median(root_rank)),
        "root_err_norm_mean": float(np.mean(root_err_norm)),
        "root_err_norm_median": float(np.median(root_err_norm)),

        "accept_mean": float(np.mean(accept)),
        "accept_median": float(np.median(accept)),

        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
    }
    return out


def plot_kendall_tau_hist(ktau: np.ndarray, outpath: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(ktau, bins=20, alpha=0.8)
    plt.xlabel("Kendall tau similarity to true order ([-1, 1])")
    plt.ylabel("Count")
    plt.title("Distribution of Kendall tau similarity on test graphs")
    plt.grid(alpha=0.3)
    ensure_dir(os.path.dirname(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def write_summary_txt(summary: Dict[str, Any], outpath: str) -> None:
    lines = [
        f"num_graphs: {summary['num_graphs']}",
        "",
        f"kendall_tau_mean:    {summary['kendall_tau_mean']:.4f}",
        f"kendall_tau_median:  {summary['kendall_tau_median']:.4f}",
        f"kendall_tau_min:     {summary['kendall_tau_min']:.4f}",
        f"kendall_tau_max:     {summary['kendall_tau_max']:.4f}",
        "",
        f"kendall_dist_mean:   {summary['kendall_distance_mean']:.2f}",
        f"kendall_dist_median: {summary['kendall_distance_median']:.2f}",
        "",
        f"root_accuracy:       {summary['root_accuracy']:.4f}",
        f"root_rank_mean:      {summary['root_rank_mean']:.2f}",
        f"root_rank_median:    {summary['root_rank_median']:.2f}",
        f"root_err_norm_mean:  {summary['root_err_norm_mean']:.4f}",
        f"root_err_norm_median:{summary['root_err_norm_median']:.4f}",
        "",
        f"accept_mean:         {summary['accept_mean']:.4f}",
        f"accept_median:       {summary['accept_median']:.4f}",
        "",
        f"score_mean:          {summary['score_mean']:.4f}",
        f"score_median:        {summary['score_median']:.4f}",
        "",
    ]
    ensure_dir(os.path.dirname(outpath))
    with open(outpath, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to anneal_results.jsonl")
    parser.add_argument("--outdir", default="analysis_out", help="Output directory for plots")
    parser.add_argument("--only_gids", nargs="*", type=int, default=None, help="Optional list of gids to analyze")
    parser.add_argument("--max_scatters", type=int, default=20, help="Max number of rank scatter plots to generate")
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)

    # Optional filter by gid
    if args.only_gids is not None and len(args.only_gids) > 0:
        keep = set(args.only_gids)
        records = [r for r in records if int(r.get("gid", -1)) in keep]
        if not records:
            raise RuntimeError("No records left after filtering by --only_gids")

    # Validate schema early (avoid silent empty plots)
    validate_schema(records)

    # Deterministic order
    records_sorted = sorted(records, key=lambda r: int(r["gid"]))

    # Only first N for scatters
    scatter_records = records_sorted[: max(args.max_scatters, 0)]

    ensure_dir(args.outdir)
    scatter_dir = os.path.join(args.outdir, "rank_scatters")
    ensure_dir(scatter_dir)

    # Rank scatter plots
    for r in scatter_records:
        gid = int(r["gid"])
        true_order = r["true_order"]
        pred_order = r["pred_order"]

        extra = f"tau={r['kendall_tau']:+.2f}, acc_rate={r['accept_rate']:.2f}, score={r['best_score']:.2f}"
        outpath = os.path.join(scatter_dir, f"gid_{gid:04d}_rank_scatter.png")
        plot_rank_scatter(true_order, pred_order, gid, outpath, title_extra=extra)

    # Summary + kendall tau histogram
    summary = summarize(records_sorted)
    write_summary_txt(summary, os.path.join(args.outdir, "summary.txt"))

    ktau_all = np.array([r["kendall_tau"] for r in records_sorted], dtype=float)
    plot_kendall_tau_hist(ktau_all, os.path.join(args.outdir, "kendall_tau_hist.png"))

    print(f"[OK] Wrote per-graph scatters (N={len(scatter_records)}) to: {scatter_dir}/")
    print(f"[OK] Wrote Kendall tau similarity histogram + summary to: {args.outdir}/")
    print("[Summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


