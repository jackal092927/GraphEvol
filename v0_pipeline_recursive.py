#!/usr/bin/env python3
"""
v0 pipeline: PA graph + vertex ordering -> TDA features -> binary classification.

Positive samples: PA graphs with their true arrival ordering.
Negative samples: same graphs, but vertex ordering is randomly permuted.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

try:
    import gudhi as gd
    from gudhi.representations import PersistenceImage
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gudhi. Install it (e.g., `pip install gudhi`) and rerun."
    ) from exc


SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Sample:
    graph: nx.Graph
    ordering: np.ndarray
    label: int
    features: np.ndarray


def generate_pa_graph(n_nodes: int, m: int, er_prob: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if not G.has_edge(u, v) and rng.random() < er_prob:
                G.add_edge(u, v)
    return G


def generate_recursive_er_graph(
    n_nodes: int = 20, 
    m: int = 1,
    alpha: float = 0.0, 
    beta: float = 1.0,
    er_prob: float = 0.03, 
    seed: int | None = None
):
    """
    Create a graph with preferential-attachment backbone and sprinkled ER shortcut edges.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes.
    m : int
        Number of edges to attach per new node. Default is 1.
    alpha : float
        Preferential attachment offset. 
        Default is 0.0.
    beta : float
        Preferential attachment degree multiplier.
        The weight of a node is: beta * degree + alpha.
        Default is 1.0.
    er_prob : float
        Base probability for adding random noise edges.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    G : nx.Graph
        The generated NetworkX graph.
    """

    rng = np.random.default_rng(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Track degrees
    degrees = np.zeros(n_nodes, dtype=float)
    
    # Build PA tree with m edges per new node
    for t in range(1, n_nodes):
        # Calculate PA weights: beta * Degree + alpha
        weights = (beta * degrees[:t]) + alpha
        
        # Ensure all weights are non-negative
        weights = np.maximum(weights, 0.0)
        weights_sum = weights.sum()
        
        # Handle cases where weights sum to 0
        if weights_sum > 0:
            probs = weights / weights_sum
        else:
            # Fallback to uniform if weights are invalid or sum to 0
            probs = np.ones(t) / t
        
        # Attach m edges from new node t to existing nodes
        num_edges = min(m, t)  # Can't attach more edges than existing nodes
        parents = rng.choice(t, size=num_edges, replace=False, p=probs)
        
        for parent in parents:
            G.add_edge(t, parent)
            degrees[t] += 1
            degrees[parent] += 1

    # Sprinkle ER Noise Edges
    if er_prob > 0:
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if not G.has_edge(u, v):
                    if rng.random() < er_prob:
                        G.add_edge(u, v)

    return G

def ordering_from_pa(n_nodes: int) -> np.ndarray:
    # PA growth is indexed by node id in NetworkX BA generator (0..n-1).
    times = np.arange(n_nodes, dtype=np.float32)
    return times / max(n_nodes - 1, 1)


def random_ordering(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    # Negative samples keep the same graph but scramble the arrival order.
    times = np.arange(n_nodes, dtype=np.float32)
    times = times[rng.permutation(n_nodes)]
    return times / max(n_nodes - 1, 1)


def simplex_tree_lower_star(G: nx.Graph, f: np.ndarray) -> gd.SimplexTree:
    # Lower-star filtration on vertices using the ordering as a scalar field.
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([v], filtration=float(f[v]))
    for u, v in G.edges():
        st.insert([u, v], filtration=float(max(f[u], f[v])))
    st.initialize_filtration()
    return st


def _safe_diag(dgm: np.ndarray | list | None) -> np.ndarray:
    if dgm is None:
        return np.empty((0, 2), dtype=float)
    dgm = np.asarray(dgm, dtype=float)
    if dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    return dgm.reshape(-1, 2)


def vectorize_extended_persistence(
    ord_dgm,
    rel_dgm,
    extp_dgm,
    extm_dgm,
    dims: Tuple[int, ...] = (0, 1),
    resolution: Tuple[int, int] = (12, 12),
    bandwidth: float = 0.05,
    im_range: List[float] | None = None,
) -> np.ndarray:
    # Fixed range is required for consistent feature dimensions across samples.
    if im_range is None:
        im_range = [0.0, 1.0, 0.0, 1.0]
    pi = PersistenceImage(
        bandwidth=bandwidth,
        resolution=list(resolution),
        im_range=im_range,
        weight=(lambda x: 1.0),
    )
    parts = [ord_dgm, rel_dgm, extp_dgm, extm_dgm]
    chunks = []
    for part in parts:
        for d in dims:
            dgm = np.array([pair for (dim, pair) in part if dim == d], dtype=float)
            dgm = _safe_diag(dgm)
            vec = pi.fit_transform([dgm])[0]
            chunks.append(vec)
    return np.concatenate(chunks, axis=0)


def graph_stats(G: nx.Graph) -> np.ndarray:
    # Small graph-level descriptors to complement TDA features.
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2.0 * m / max(n, 1)
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    return np.array([n, m, avg_deg, density, clustering], dtype=np.float32)


def featurize_sample(G: nx.Graph, ordering: np.ndarray) -> np.ndarray:
    # Extended persistence -> persistence images -> flat vector + graph stats.
    st = simplex_tree_lower_star(G, ordering)
    st.extend_filtration()
    ord_dgm, rel_dgm, extp_dgm, extm_dgm = st.extended_persistence()
    tda_vec = vectorize_extended_persistence(ord_dgm, rel_dgm, extp_dgm, extm_dgm)
    return np.concatenate([tda_vec, graph_stats(G)], axis=0)


def make_dataset(
    num_graphs: int = 120,
    n_nodes: int = 30,
    m: int = 2,
    alpha: float = 0.0,
    beta: float = 1.0,
    er_prob: float = 0.03,
) -> List[Sample]:
    # Each graph yields one positive and one negative sample.
    rng = np.random.default_rng(SEED)
    samples: List[Sample] = []
    for i in range(num_graphs):
        G = generate_recursive_er_graph(n_nodes=n_nodes, m=m, alpha=alpha, beta=beta, er_prob=er_prob, seed=SEED + i)
        pos_order = ordering_from_pa(n_nodes)
        neg_order = random_ordering(n_nodes, rng)
        pos_feat = featurize_sample(G, pos_order)
        neg_feat = featurize_sample(G, neg_order)
        samples.append(Sample(G, pos_order, 1, pos_feat))
        samples.append(Sample(G, neg_order, 0, neg_feat))
    return samples


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


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_weight = nn.Parameter(torch.empty(heads, 2 * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Full attention over the adjacency mask (dense for small graphs).
        n = x.size(0)
        adj = adj.to(x.device)
        h = self.lin(x).view(n, self.heads, self.out_dim)
        h_i = h.unsqueeze(1).repeat(1, n, 1, 1)
        h_j = h.unsqueeze(0).repeat(n, 1, 1, 1)
        attn_input = torch.cat([h_i, h_j], dim=-1)
        scores = torch.einsum("ijhd,hd->ijh", attn_input, self.att_weight)
        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)
        mask = (adj > 0).unsqueeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        alpha = self.dropout(alpha)
        alpha = alpha.permute(0, 2, 1)
        h_prime = torch.einsum("ihj,jhd->ihd", alpha, h)
        if self.concat:
            h_prime = h_prime.reshape(n, self.heads * self.out_dim)
        else:
            h_prime = h_prime.mean(dim=1)
        return h_prime


class GATGraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GraphAttentionLayer(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GraphAttentionLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Mean pool node embeddings to a graph embedding.
        h = F.elu(self.gat1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.gat2(h, adj))
        graph_emb = h.mean(dim=0)
        return self.out(graph_emb).squeeze(-1)


def split_dataset(samples: List[Sample], train_ratio: float = 0.7, val_ratio: float = 0.15):
    random.shuffle(samples)
    n = len(samples)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    return samples[:train_cut], samples[train_cut:val_cut], samples[val_cut:]


def batch_tensors(samples: List[Sample]):
    X = np.stack([s.features for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.float32)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    labels = labels.float()
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_node_features(G: nx.Graph, ordering: np.ndarray) -> torch.Tensor:
    # Node features for GAT include the ordering plus standard structural stats.
    n = G.number_of_nodes()
    degrees = np.array([G.degree(i) for i in range(n)], dtype=np.float32)
    clustering = np.array([nx.clustering(G, i) for i in range(n)], dtype=np.float32)
    pagerank_dict = nx.pagerank(G)
    pagerank = np.array([pagerank_dict[i] for i in range(n)], dtype=np.float32)
    core_dict = nx.core_number(G)
    core = np.array([core_dict[i] for i in range(n)], dtype=np.float32)
    neighbor_mean = []
    for node in range(n):
        neigh = [G.degree(nb) for nb in G.neighbors(node)]
        neighbor_mean.append(np.mean(neigh) if neigh else 0.0)
    neighbor_mean = np.array(neighbor_mean, dtype=np.float32)
    features = np.stack([ordering, degrees, clustering, pagerank, core, neighbor_mean], axis=1)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - mean) / std
    return torch.from_numpy(features).float()


def build_adjacency(G: nx.Graph) -> torch.Tensor:
    # Binary adjacency with self-loops for attention.
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)
    adj = adj + np.eye(n, dtype=np.float32)
    adj[adj > 0] = 1.0
    return torch.from_numpy(adj).float()


def train_model(
    train_samples: List[Sample],
    val_samples: List[Sample],
    epochs: int = 120,
    lr: float = 1e-3,
) -> Tuple[MLPClassifier, dict]:
    X_train, y_train = batch_tensors(train_samples)
    X_val, y_val = batch_tensors(val_samples)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)

    model = MLPClassifier(in_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            metrics = compute_metrics(val_logits, y_val)

        history["train_loss"].append(float(loss.item()))
        history["val_acc"].append(float(metrics["acc"]))
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={metrics['acc']:.3f}")
    return model, history


def train_gat_model(
    train_samples: List[Sample],
    val_samples: List[Sample],
    epochs: int = 120,
    lr: float = 1e-3,
) -> Tuple[GATGraphClassifier, dict]:
    model = GATGraphClassifier(in_dim=6, hidden_dim=32, heads=4, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        for sample in train_samples:
            optimizer.zero_grad()
            x = build_node_features(sample.graph, sample.ordering).to(DEVICE)
            adj = build_adjacency(sample.graph).to(DEVICE)
            logit = model(x, adj)
            label = torch.tensor(float(sample.label), device=DEVICE)
            loss = loss_fn(logit, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = []
            val_labels = []
            for sample in val_samples:
                x = build_node_features(sample.graph, sample.ordering).to(DEVICE)
                adj = build_adjacency(sample.graph).to(DEVICE)
                val_logits.append(model(x, adj).cpu())
                val_labels.append(float(sample.label))
            val_logits = torch.stack(val_logits)
            val_labels = torch.tensor(val_labels)
            metrics = compute_metrics(val_logits, val_labels)

        history["train_loss"].append(total_loss / max(len(train_samples), 1))
        history["val_acc"].append(float(metrics["acc"]))
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={history['train_loss'][-1]:.4f} | val_acc={metrics['acc']:.3f}")

    return model, history


def save_fig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def visualize_samples(samples: List[Sample], out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pos = next(s for s in samples if s.label == 1)
    neg = next(s for s in samples if s.label == 0)
    pos_layout = nx.spring_layout(pos.graph, seed=SEED)
    neg_layout = nx.spring_layout(neg.graph, seed=SEED + 1)

    nx.draw_networkx(
        pos.graph,
        pos=pos_layout,
        node_color=pos.ordering,
        cmap="viridis",
        node_size=120,
        ax=axes[0],
        with_labels=False,
    )
    axes[0].set_title("Positive: PA ordering")
    axes[0].axis("off")

    nx.draw_networkx(
        neg.graph,
        pos=neg_layout,
        node_color=neg.ordering,
        cmap="viridis",
        node_size=120,
        ax=axes[1],
        with_labels=False,
    )
    axes[1].set_title("Negative: permuted ordering")
    axes[1].axis("off")
    save_fig(os.path.join(out_dir, "sample_graphs.png"))


def visualize_persistence_images(samples: List[Sample], out_dir: str):
    pos = next(s for s in samples if s.label == 1)
    neg = next(s for s in samples if s.label == 0)
    res = (12, 12)
    chunk = res[0] * res[1]
    def to_grid(vec, idx):
        return vec[idx * chunk : (idx + 1) * chunk].reshape(res)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes[0, 0].imshow(to_grid(pos.features, 0))
    axes[0, 0].set_title("Pos: ord H0")
    axes[0, 1].imshow(to_grid(pos.features, 1))
    axes[0, 1].set_title("Pos: ord H1")
    axes[1, 0].imshow(to_grid(neg.features, 0))
    axes[1, 0].set_title("Neg: ord H0")
    axes[1, 1].imshow(to_grid(neg.features, 1))
    axes[1, 1].set_title("Neg: ord H1")
    for ax in axes.ravel():
        ax.axis("off")
    save_fig(os.path.join(out_dir, "persistence_images.png"))


def visualize_tda_class_averages(samples: List[Sample], out_dir: str, resolution: Tuple[int, int] = (12, 12)):
    # Compare mean persistence images between positive and negative classes.
    pos_feats = np.stack([s.features for s in samples if s.label == 1], axis=0)
    neg_feats = np.stack([s.features for s in samples if s.label == 0], axis=0)
    chunk = resolution[0] * resolution[1]

    def mean_grid(features, idx):
        vecs = features[:, idx * chunk : (idx + 1) * chunk]
        return vecs.mean(axis=0).reshape(resolution)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes[0, 0].imshow(mean_grid(pos_feats, 0))
    axes[0, 0].set_title("Pos mean: ord H0")
    axes[0, 1].imshow(mean_grid(neg_feats, 0))
    axes[0, 1].set_title("Neg mean: ord H0")
    axes[0, 2].imshow(mean_grid(pos_feats, 0) - mean_grid(neg_feats, 0), cmap="coolwarm")
    axes[0, 2].set_title("Diff: ord H0")

    axes[1, 0].imshow(mean_grid(pos_feats, 1))
    axes[1, 0].set_title("Pos mean: ord H1")
    axes[1, 1].imshow(mean_grid(neg_feats, 1))
    axes[1, 1].set_title("Neg mean: ord H1")
    axes[1, 2].imshow(mean_grid(pos_feats, 1) - mean_grid(neg_feats, 1), cmap="coolwarm")
    axes[1, 2].set_title("Diff: ord H1")

    for ax in axes.ravel():
        ax.axis("off")
    save_fig(os.path.join(out_dir, "tda_class_means.png"))


def visualize_feature_norms(samples: List[Sample], out_dir: str):
    # Quick separation check: norm distribution of TDA+stats vectors per class.
    pos_norms = [float(np.linalg.norm(s.features)) for s in samples if s.label == 1]
    neg_norms = [float(np.linalg.norm(s.features)) for s in samples if s.label == 0]
    plt.figure(figsize=(6, 4))
    plt.hist(pos_norms, bins=20, alpha=0.7, label="Positive")
    plt.hist(neg_norms, bins=20, alpha=0.7, label="Negative")
    plt.xlabel("Feature vector norm")
    plt.ylabel("Count")
    plt.title("TDA feature norms by class")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(os.path.join(out_dir, "feature_norms.png"))


def visualize_training(history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_acc"])
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    save_fig(os.path.join(out_dir, "training_curves.png"))


def visualize_confusion(metrics: dict, out_dir: str):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(os.path.join(out_dir, "confusion_matrix.png"))


def main():
    parser = argparse.ArgumentParser(description="v0 graph ordering classifier")
    parser.add_argument("--model", choices=["mlp", "gat"], default="mlp")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    # Dataset generation parameters (easy to tweak)
    m = 1
    alpha = 0
    beta = 1.0
    er_prob = 0.1

    samples = make_dataset(num_graphs=120, n_nodes=30, m=m, alpha=alpha, beta=beta, er_prob=er_prob)
    train_samples, val_samples, test_samples = split_dataset(samples)

    if args.model == "gat":
        model, history = train_gat_model(train_samples, val_samples, epochs=120, lr=1e-3)
        model.eval()
        with torch.no_grad():
            test_logits = []
            test_labels = []
            for sample in test_samples:
                x = build_node_features(sample.graph, sample.ordering).to(DEVICE)
                adj = build_adjacency(sample.graph).to(DEVICE)
                test_logits.append(model(x, adj).cpu())
                test_labels.append(float(sample.label))
            test_logits = torch.stack(test_logits)
            test_labels = torch.tensor(test_labels)
        metrics = compute_metrics(test_logits, test_labels)
    else:
        model, history = train_model(train_samples, val_samples, epochs=120, lr=1e-3)
        X_test, y_test = batch_tensors(test_samples)
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(DEVICE)).cpu()
        metrics = compute_metrics(logits, y_test)

    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items() if k not in {"tp", "tn", "fp", "fn"}})

    out_dir_base = "outputs_tda" if args.model == "mlp" else "outputs_gat"
    out_dir = f"{out_dir_base}_m{m}_alpha{alpha}_er{er_prob}"
    visualize_samples(samples, out_dir)
    visualize_persistence_images(samples, out_dir)
    visualize_tda_class_averages(samples, out_dir)
    visualize_feature_norms(samples, out_dir)
    visualize_training(history, out_dir)
    visualize_confusion(metrics, out_dir)
    print(f"Saved visualizations to `{out_dir}/`")


if __name__ == "__main__":
    main()
