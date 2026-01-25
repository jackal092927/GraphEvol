from __future__ import annotations
import numpy as np
import networkx as nx

def generate_pa_graph(n_nodes: int, m: int, er_prob: float, seed: int) -> nx.Graph:
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if (not G.has_edge(u, v)) and (rng.random() < er_prob):
                G.add_edge(u, v)
    return G

def ordering_from_pa(n_nodes: int) -> np.ndarray:
    times = np.arange(n_nodes, dtype=np.float64)
    if n_nodes <= 1:
        return np.zeros((n_nodes,), dtype=np.float64)
    return times / (n_nodes - 1)
