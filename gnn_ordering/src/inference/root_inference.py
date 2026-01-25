"""Root inference placeholder (you will redesign).

Only includes a utility sampler for "fixed root then random completion".
"""
from __future__ import annotations
import numpy as np

def sample_tau_with_fixed_root(n: int, root: int, rng: np.random.Generator) -> np.ndarray:
    nodes = np.arange(n)
    others = nodes[nodes != root]
    perm = rng.permutation(others)
    order = np.concatenate([[root], perm], axis=0)
    tau = np.empty((n,), dtype=np.float64)
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    for r, node in enumerate(order.tolist()):
        tau[int(node)] = r / (n - 1)
    return tau
