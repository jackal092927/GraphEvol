from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import math
from ..inference.scoring import score_taus_batch

def order_to_tau(order: np.ndarray) -> np.ndarray:
    n = int(order.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    tau = np.empty((n,), dtype=np.float64)
    for r, node in enumerate(order.tolist()):
        tau[int(node)] = r / (n - 1)
    return tau

def proposal_swap(order: np.ndarray, rng: np.random.Generator, fixed_prefix_len: int = 0) -> np.ndarray:
    """
    Swap two positions in the order. If fixed_prefix_len=k, keep order[:k] unchanged.
    """
    prop = order.copy()
    n = int(prop.shape[0])

    k = int(fixed_prefix_len)
    if k < 0:
        k = 0
    if k >= n - 1:
        # nothing meaningful to swap; return copy
        return prop

    i = int(rng.integers(k, n))
    j = int(rng.integers(k, n))
    while j == i:
        j = int(rng.integers(k, n))

    prop[i], prop[j] = prop[j], prop[i]
    return prop


def temperature_schedule(T0: float, Tend: float, t: int, steps: int) -> float:
    if steps <= 1:
        return Tend
    frac = t / (steps - 1)
    return T0 * (Tend / T0) ** frac

def simulated_annealing_ordering(
    model,
    F_obs: np.ndarray,
    init_order: np.ndarray,
    *,
    steps: int = 3000,
    T0: float = 1.0,
    Tend: float = 1e-3,
    seed: int = 0,
    fixed_prefix_len: int = 0,   # NEW: lock first k positions
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    cur_order = init_order.astype(int).copy()
    cur_tau = order_to_tau(cur_order)
    cur_logit, cur_prob = score_taus_batch(model, cur_tau[None, :], F_obs)
    cur_logit = float(cur_logit[0]); cur_prob = float(cur_prob[0])

    best_order = cur_order.copy()
    best_logit, best_prob = cur_logit, cur_prob
    hist = {"cur_logit": [], "best_logit": [], "T": []}

    # normalize fixed_prefix_len
    n = int(cur_order.shape[0])
    k = int(fixed_prefix_len)
    if k < 0:
        k = 0
    if k > n:
        k = n

    for t in range(steps):
        T = temperature_schedule(T0, Tend, t, steps)

        # propose swap only in suffix [k, n)
        prop_order = proposal_swap(cur_order, rng, fixed_prefix_len=k)

        # safety: ensure prefix is unchanged
        if k > 0:
            # if your proposal function is correct, this is always true
            assert np.array_equal(prop_order[:k], cur_order[:k])

        prop_tau = order_to_tau(prop_order)
        prop_logit, prop_prob = score_taus_batch(model, prop_tau[None, :], F_obs)
        prop_logit = float(prop_logit[0]); prop_prob = float(prop_prob[0])

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

        hist["cur_logit"].append(cur_logit)
        hist["best_logit"].append(best_logit)
        hist["T"].append(T)

    best = {
        "order": best_order.astype(int).tolist(),
        "tau": order_to_tau(best_order).astype(float).tolist(),
        "logit": float(best_logit),
        "prob": float(best_prob),
    }
    return best, hist

