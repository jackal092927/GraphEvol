#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0-revisit + annealing + diagnostics

Exact energy model:
    score(tau; G) = MLP( F(G, tau) )

Training data construction (graph-split, no leakage):
  For each graph G_i:
    - random relabeling to destroy global monotone tau pattern
    - positive: (G_i, tau_true) -> y=1
    - negative: (G_i, permute(tau_true)) -> y=0

Inference (test graphs only):
  - simulated annealing over permutations to maximize score(tau; G)
  - predicted order -> root identify (earliest arrival)

Saved outputs:
  1) classifier_metrics.json  (train/val/test accuracy)
  2) anneal_results.jsonl     (per test graph results, incl. Kendall tau similarity + root error)
  3) annealing_history/*.png  (optional; first N test graphs)
  4) diagnostics_score_vs_kendall/*.png (optional; first N test graphs)
     For each selected test graph: scatter of score(tau;G) vs Kendall-tau similarity to the true order,
     using many sampled taus (random + local perturbations). This diagnoses whether the learned score
     correlates with true proximity.

Dependencies:
  pip install networkx numpy torch gudhi matplotlib
"""

from __future__ import annotations
import os
import math
import json
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import networkx as nx
import torch
import torch.nn as nn


# -------------------------
# Repro / device
# -------------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# (A) Graph generator (PA + optional ER noise)
# -------------------------
def generate_pa_graph(n_nodes: int, m: int, seed: int) -> nx.Graph:
    return nx.barabasi_albert_graph(n_nodes, m, seed=seed)


def sprinkle_er_edges(G: nx.Graph, er_prob: float, seed: int) -> nx.Graph:
    if er_prob <= 0:
        return G
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    for u in range(n):
        for v in range(u + 1, n):
            if not G.has_edge(u, v) and rng.random() < er_prob:
                G.add_edge(u, v)
    return G


def generate_graph(n_nodes: int, m: int, er_prob: float, seed: int) -> nx.Graph:
    G = generate_pa_graph(n_nodes=n_nodes, m=m, seed=seed)
    return sprinkle_er_edges(G, er_prob=er_prob, seed=seed + 10000)


# -------------------------
# (B) Random relabeling (kill global monotone tau pattern)
# -------------------------
def relabel_graph_and_tau(G: nx.Graph, rng: np.random.Generator) -> Tuple[nx.Graph, np.ndarray]:
    """
    Original PA arrival time ~= old node id (0..n-1).
    Randomly relabel nodes: old -> new. Return:
      G_new: nodes still 0..n-1
      tau_true[new_id] = old_id/(n-1)
    """
    n = G.number_of_nodes()
    perm = rng.permutation(n)  # perm[old] = new
    mapping = {old: int(perm[old]) for old in range(n)}
    G2 = nx.relabel_nodes(G, mapping, copy=True)

    denom = max(n - 1, 1)
    tau_true = np.zeros(n, dtype=np.float32)
    for old in range(n):
        new = mapping[old]
        tau_true[new] = float(old) / denom
    return G2, tau_true


def permute_tau_values(tau: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Negative: keep same graph, scramble tau values across nodes."""
    return tau[rng.permutation(tau.shape[0])].astype(np.float32)


def tau_to_order(tau: np.ndarray) -> np.ndarray:
    """Convert tau (time per node) to ordering permutation (earliest -> latest)."""
    return np.argsort(tau).astype(np.int64)


def order_to_tau(order: np.ndarray) -> np.ndarray:
    """Convert ordering permutation to tau (rank normalized to [0,1])."""
    n = order.shape[0]
    tau = np.empty(n, dtype=np.float32)
    denom = max(n - 1, 1)
    for rank, node in enumerate(order):
        tau[int(node)] = float(rank) / denom
    return tau


# -------------------------
# (C) TDA feature extractor: F(G, tau)
# -------------------------
def featurize_tda(G: nx.Graph, tau: np.ndarray) -> np.ndarray:
    """
    Extended persistence -> persistence image -> flat vector + graph stats.
    """
    import gudhi as gd
    from gudhi.representations import PersistenceImage

    # lower-star filtration
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([int(v)], filtration=float(tau[int(v)]))
    for u, v in G.edges():
        st.insert([int(u), int(v)], filtration=float(max(tau[int(u)], tau[int(v)])))
    st.initialize_filtration()
    st.extend_filtration()

    ord_dgm, rel_dgm, extp_dgm, extm_dgm = st.extended_persistence()

    # persistence image (fixed dim)
    dims = (0, 1)
    resolution = (12, 12)
    bandwidth = 0.05
    im_range = [0.0, 1.0, 0.0, 1.0]
    pi = PersistenceImage(
        bandwidth=bandwidth,
        resolution=list(resolution),
        im_range=im_range,
        weight=(lambda x: 1.0),
    )

    def safe_diag(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        return arr.reshape(-1, 2)

    chunks = []
    for part in (ord_dgm, rel_dgm, extp_dgm, extm_dgm):
        for d in dims:
            dgm = np.array([pair for (dim, pair) in part if dim == d], dtype=float)
            dgm = safe_diag(dgm)
            vec = pi.fit_transform([dgm])[0].astype(np.float32)
            chunks.append(vec)

    tda_vec = np.concatenate(chunks, axis=0).astype(np.float32)

    # graph stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2.0 * m / max(n, 1)
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    stats = np.array([n, m, avg_deg, density, clustering], dtype=np.float32)

    return np.concatenate([tda_vec, stats], axis=0).astype(np.float32)


# -------------------------
# (D) Dataset
# -------------------------
@dataclass
class GraphInstance:
    gid: int
    G: nx.Graph
    tau_true: np.ndarray  # (n,)


@dataclass
class Sample:
    x: np.ndarray         # F(G,tau)
    y: float
    gid: int
    tau: np.ndarray       # keep for debug


def make_instances(num_graphs: int, n_nodes: int, m: int, er_prob: float, seed: int = SEED) -> List[GraphInstance]:
    rng = np.random.default_rng(seed)
    insts: List[GraphInstance] = []
    for gid in range(num_graphs):
        G0 = generate_graph(n_nodes=n_nodes, m=m, er_prob=er_prob, seed=seed + gid)
        G, tau_true = relabel_graph_and_tau(G0, rng=rng)
        insts.append(GraphInstance(gid=gid, G=G, tau_true=tau_true))
    return insts


def split_by_graph(insts: List[GraphInstance], train_ratio=0.7, val_ratio=0.15, seed: int = SEED):
    rng = np.random.default_rng(seed + 2026)
    ids = np.array([inst.gid for inst in insts], dtype=np.int64)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(ids[:n_train].tolist())
    val_ids = set(ids[n_train:n_train + n_val].tolist())
    test_ids = set(ids[n_train + n_val:].tolist())
    return train_ids, val_ids, test_ids


def build_samples(insts: List[GraphInstance], seed: int = SEED) -> List[Sample]:
    """
    For each graph:
      pos: F(G, tau_true)
      neg: F(G, permute(tau_true))
    """
    rng = np.random.default_rng(seed + 999)
    samples: List[Sample] = []
    for inst in insts:
        tau_pos = inst.tau_true
        tau_neg = permute_tau_values(inst.tau_true, rng)

        x_pos = featurize_tda(inst.G, tau_pos)
        x_neg = featurize_tda(inst.G, tau_neg)

        samples.append(Sample(x=x_pos, y=1.0, gid=inst.gid, tau=tau_pos))
        samples.append(Sample(x=x_neg, y=0.0, gid=inst.gid, tau=tau_neg))
    return samples


# -------------------------
# (E) Model + training
# -------------------------
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def to_tensors(samples: List[Sample]):
    X = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    y = np.array([s.y for s in samples], dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(X.to(DEVICE)).detach().cpu()
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    return float((pred == y).float().mean().item())


def train_mlp(train_samples: List[Sample], val_samples: List[Sample],
              epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-4) -> MLPClassifier:
    X_train, y_train = to_tensors(train_samples)
    X_val, y_val = to_tensors(val_samples)

    model = MLPClassifier(in_dim=X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train.to(DEVICE))
        loss = loss_fn(logits, y_train.to(DEVICE))
        loss.backward()
        opt.step()

        val_acc = accuracy(model, X_val, y_val)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            tr_acc = accuracy(model, X_train, y_train)
            print(f"[ep {ep:03d}] loss={loss.item():.4f}  train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}  best_val={best_val:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------
# (F) Exact scoring + Annealing
# -------------------------
@torch.no_grad()
def score_tau(model: MLPClassifier, G: nx.Graph, tau: np.ndarray) -> float:
    """Exact-A score: logit = MLP( F(G,tau) )."""
    x = featurize_tda(G, tau)
    xt = torch.from_numpy(x).float().to(DEVICE).unsqueeze(0)
    return float(model(xt).item())


def proposal_swap(order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = order.shape[0]
    i, j = rng.integers(0, n, size=2)
    while j == i:
        j = int(rng.integers(0, n))
    new = order.copy()
    new[i], new[j] = new[j], new[i]
    return new


def anneal_order(
    model: MLPClassifier,
    G: nx.Graph,
    init_order: np.ndarray,
    n_steps: int = 1500,
    t0: float = 1.0,
    t_end: float = 0.05,
    seed: int = SEED,
    return_history: bool = False,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed + 4242)

    def temp(step: int) -> float:
        if n_steps <= 1:
            return t_end
        r = step / (n_steps - 1)
        return t0 * (t_end / t0) ** r

    cur_order = init_order.copy()
    cur_tau = order_to_tau(cur_order)
    cur_score = score_tau(model, G, cur_tau)

    best_order = cur_order.copy()
    best_score = cur_score
    accept = 0

    hist_cur: Optional[List[float]] = [] if return_history else None
    hist_best: Optional[List[float]] = [] if return_history else None

    for step in range(n_steps):
        T = temp(step)
        prop_order = proposal_swap(cur_order, rng)
        prop_tau = order_to_tau(prop_order)
        prop_score = score_tau(model, G, prop_tau)

        delta = prop_score - cur_score
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-8)):
            cur_order, cur_tau, cur_score = prop_order, prop_tau, prop_score
            accept += 1
            if cur_score > best_score:
                best_score = cur_score
                best_order = cur_order.copy()

        if return_history:
            hist_cur.append(cur_score)
            hist_best.append(best_score)

    out: Dict[str, object] = {
        "best_order": best_order,
        "best_tau": order_to_tau(best_order),
        "best_score": best_score,
        "accept_rate": accept / max(n_steps, 1),
    }
    if return_history:
        out["history_current"] = hist_cur
        out["history_best"] = hist_best
    return out


def identify_root_from_tau(tau: np.ndarray) -> int:
    return int(np.argmin(tau))


def kendall_tau_distance(order_a: np.ndarray, order_b: np.ndarray) -> int:
    """Kendall tau distance between two permutations (number of discordant pairs). O(n^2) fine for n=30."""
    n = order_a.shape[0]
    pos_a = np.empty(n, dtype=np.int64)
    pos_b = np.empty(n, dtype=np.int64)
    for i, node in enumerate(order_a):
        pos_a[int(node)] = i
    for i, node in enumerate(order_b):
        pos_b[int(node)] = i
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += int((pos_a[i] < pos_a[j]) != (pos_b[i] < pos_b[j]))
    return dist


def kendall_tau_similarity_from_distance(K: int, n: int) -> float:
    """
    Kendall tau similarity in [-1,1]:
      tau = 1 - 2*K / C(n,2)
    where K is Kendall distance (#discordant pairs).
    """
    kmax = n * (n - 1) // 2
    if kmax == 0:
        return 1.0
    return float(1.0 - 2.0 * (K / kmax))


# -------------------------
# (G) Plot helpers (annealing history + diagnostic)
# -------------------------
def save_anneal_history_plot(history_current: List[float], history_best: List[float], outpath: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history_current, label="current logit")
    plt.plot(history_best, label="best logit")
    plt.title("Simulated annealing progress")
    plt.xlabel("step")
    plt.ylabel("logit score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _order_to_rank(order: np.ndarray) -> np.ndarray:
    n = order.shape[0]
    rank = np.empty(n, dtype=np.int64)
    for i, node in enumerate(order):
        rank[int(node)] = i
    return rank


def diagnostic_score_vs_kendall(
    model: MLPClassifier,
    G: nx.Graph,
    true_order: np.ndarray,
    n_random: int = 250,
    n_local: int = 250,
    local_k_swaps: int = 3,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample many taus and compute:
      x = Kendall-tau similarity between sampled order and true_order (in [-1,1])
      y = score(tau;G) (logit)
    Sampling mix:
      - n_random: completely random permutations
      - n_local: local perturbations around true_order (k random swaps)
    """
    rng = np.random.default_rng(seed)

    n = true_order.shape[0]
    scores = []
    ktau = []

    # random perms
    for _ in range(n_random):
        order = rng.permutation(n).astype(np.int64)
        K = kendall_tau_distance(true_order, order)
        kt = kendall_tau_similarity_from_distance(K, n)
        tau = order_to_tau(order)
        sc = score_tau(model, G, tau)
        ktau.append(kt)
        scores.append(sc)

    # local around true order
    base = true_order.copy()
    for _ in range(n_local):
        order = base.copy()
        for __ in range(local_k_swaps):
            i, j = rng.integers(0, n, size=2)
            while j == i:
                j = int(rng.integers(0, n))
            order[i], order[j] = order[j], order[i]

        K = kendall_tau_distance(true_order, order)
        kt = kendall_tau_similarity_from_distance(K, n)
        tau = order_to_tau(order)
        sc = score_tau(model, G, tau)
        ktau.append(kt)
        scores.append(sc)

    return np.asarray(ktau, dtype=float), np.asarray(scores, dtype=float)


def save_diag_plot_score_vs_kendall(ktau: np.ndarray, score: np.ndarray, outpath: str, gid: int) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(ktau, score, s=18)  # no explicit colors
    plt.grid(alpha=0.3)
    plt.xlabel("Kendall tau similarity to true order ([-1,1])")
    plt.ylabel("score(tau; G) = model logit")
    plt.title(f"Score vs proximity diagnostic (gid={gid})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# -------------------------
# (H) End-to-end
# -------------------------
def main(
    out_dir: str = "outputs_exact_v0",
    num_graphs: int = 500,
    n_nodes: int = 30,
    m: int = 2,
    er_prob: float = 0.03,
    epochs: int = 50,
    anneal_steps: int = 2400,
    # requested add-ons:
    save_first_n_test_plots: int = 20,
    diag_random: int = 250,
    diag_local: int = 250,
    diag_local_k: int = 3,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) make graphs and split by graph id (no leakage)
    insts = make_instances(num_graphs=num_graphs, n_nodes=n_nodes, m=m, er_prob=er_prob)
    train_ids, val_ids, test_ids = split_by_graph(insts)

    train_insts = [x for x in insts if x.gid in train_ids]
    val_insts   = [x for x in insts if x.gid in val_ids]
    test_insts  = [x for x in insts if x.gid in test_ids]

    print(f"Graphs: train={len(train_insts)} val={len(val_insts)} test={len(test_insts)}")

    # 2) build samples (computes TDA twice per graph: pos+neg)
    print("Building samples (computes TDA twice per graph: pos+neg) ...")
    train_samples = build_samples(train_insts)
    val_samples   = build_samples(val_insts)
    test_samples  = build_samples(test_insts)

    # 3) train classifier (best val state)
    model = train_mlp(train_samples, val_samples, epochs=epochs)

    # 4) classifier accuracies (train/val/test)
    X_train, y_train = to_tensors(train_samples)
    X_val, y_val     = to_tensors(val_samples)
    X_test, y_test   = to_tensors(test_samples)

    train_acc = accuracy(model, X_train, y_train)
    val_acc   = accuracy(model, X_val, y_val)
    test_acc  = accuracy(model, X_test, y_test)

    print(f"[Classifier] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}")

    with open(os.path.join(out_dir, "classifier_metrics.json"), "w") as f:
        json.dump(
            {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc},
            f,
            indent=2
        )

    # directories for plots
    hist_dir = os.path.join(out_dir, "annealing_history")
    diag_dir = os.path.join(out_dir, "diagnostics_score_vs_kendall")
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # 5) annealing on held-out test graphs + (optional) plots
    results: List[Dict[str, object]] = []

    for idx, inst in enumerate(test_insts):
        # init: degree order (just a start point)
        deg = np.array([inst.G.degree(i) for i in range(n_nodes)], dtype=np.float32)
        init_order = np.argsort(-deg, kind="stable").astype(np.int64)

        do_plots = (idx < save_first_n_test_plots)

        out = anneal_order(
            model=model,
            G=inst.G,
            init_order=init_order,
            n_steps=anneal_steps,
            t0=1.0,
            t_end=0.05,
            seed=SEED + inst.gid,
            return_history=do_plots,
        )

        # save annealing history plot
        if do_plots:
            fig_path = os.path.join(hist_dir, f"gid_{inst.gid:04d}_anneal.png")
            save_anneal_history_plot(out["history_current"], out["history_best"], fig_path)

        pred_order = out["best_order"].astype(np.int64)
        pred_tau = out["best_tau"].astype(np.float32)
        true_order = tau_to_order(inst.tau_true).astype(np.int64)

        # Kendall tau similarity for recovered ordering
        K = kendall_tau_distance(true_order, pred_order)
        ktau = kendall_tau_similarity_from_distance(K, n_nodes)

        # root metrics
        pred_root = identify_root_from_tau(pred_tau)
        true_root = identify_root_from_tau(inst.tau_true)

        # pred_root rank in TRUE ordering (how far off is the root guess?)
        true_rank = np.empty(n_nodes, dtype=np.int64)
        for pos, node in enumerate(true_order):
            true_rank[int(node)] = pos

        pred_root_true_rank = int(true_rank[pred_root])          # 0 means correct
        root_rank_error_norm = pred_root_true_rank / max(n_nodes - 1, 1)

        # diagnostic D: score vs proximity to true order
        if do_plots:
            ktau_s, score_s = diagnostic_score_vs_kendall(
                model=model,
                G=inst.G,
                true_order=true_order,
                n_random=diag_random,
                n_local=diag_local,
                local_k_swaps=diag_local_k,
                seed=SEED + 100000 + inst.gid,
            )
            diag_path = os.path.join(diag_dir, f"gid_{inst.gid:04d}_score_vs_kendall.png")
            save_diag_plot_score_vs_kendall(ktau_s, score_s, diag_path, gid=inst.gid)

        results.append({
            "gid": inst.gid,
            "n_nodes": n_nodes,

            # inference outputs
            "pred_order": pred_order.tolist(),
            "true_order": true_order.tolist(),
            "best_score": float(out["best_score"]),
            "accept_rate": float(out["accept_rate"]),

            # ordering accuracy
            "kendall_distance": int(K),
            "kendall_tau": float(ktau),

            # root accuracy / severity
            "pred_root": int(pred_root),
            "true_root": int(true_root),
            "pred_root_true_rank": int(pred_root_true_rank),
            "root_rank_error_norm": float(root_rank_error_norm),
        })

        print(
            f"[gid {inst.gid:03d}] "
            f"kendall_tau={ktau:+.3f}  "
            f"root(pred/true)={pred_root}/{true_root}  "
            f"root_rank_error_norm={root_rank_error_norm:.3f}  "
            f"acc_rate={out['accept_rate']:.2f}"
        )

    # 6) save per-graph test inference results
    out_jsonl = os.path.join(out_dir, "anneal_results.jsonl")
    with open(out_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved -> {out_jsonl}")
    print(f"Saved -> {os.path.join(out_dir, 'classifier_metrics.json')}")
    print(f"Saved plots -> {hist_dir} (first {save_first_n_test_plots} test graphs)")
    print(f"Saved diagnostics -> {diag_dir} (first {save_first_n_test_plots} test graphs)")


if __name__ == "__main__":
    main()

