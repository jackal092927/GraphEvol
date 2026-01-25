from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import networkx as nx
import networkx as nx

from .tda_features import simplex_tree_lower_star, extended_persistence_diagrams, diagrams_to_pi_vector

def graph_stats(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 2.0 * m / max(n, 1)
    dens = nx.density(G) if n > 1 else 0.0
    clust = nx.average_clustering(G) if n > 2 else 0.0
    return np.array([n, m, avg_deg, dens, clust], dtype=np.float64)

def featurize_hatF(
    G: nx.Graph,
    tau_hat: np.ndarray,
    *,
    bandwidth: float = 0.05,
    resolution: Tuple[int, int] = (12, 12),
    im_range: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    dims: Tuple[int, ...] = (0, 1),
    return_meta: bool = False,
):
    st = simplex_tree_lower_star(G, tau_hat)
    ord_dgm, rel_dgm, extp_dgm, extm_dgm = extended_persistence_diagrams(st)
    tda_vec = diagrams_to_pi_vector(
        diagrams_parts=[ord_dgm, rel_dgm, extp_dgm, extm_dgm],
        dims=dims,
        bandwidth=bandwidth,
        resolution=resolution,
        im_range=im_range,
    )
    stats = graph_stats(G)
    F = np.concatenate([tda_vec, stats], axis=0).astype(np.float32)

    if not return_meta:
        return F
    meta: Dict[str, Any] = {
        "tda_dim": int(tda_vec.shape[0]),
        "stats_dim": int(stats.shape[0]),
        "F_dim": int(F.shape[0]),
        "resolution": tuple(resolution),
        "dims": tuple(dims),
        "parts": ["ord", "rel", "ext+", "ext-"],
    }
    return F, meta
