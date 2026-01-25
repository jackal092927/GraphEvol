from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from ..inference.scoring import score_taus_batch

def random_tau_batch(rng: np.random.Generator, B: int, n: int) -> np.ndarray:
    taus = np.zeros((B, n), dtype=np.float64)
    for b in range(B):
        perm = rng.permutation(n)
        if n <= 1:
            taus[b] = 0.0
        else:
            for r, node in enumerate(perm.tolist()):
                taus[b, node] = r / (n - 1)
    return taus

def topk_orderings_random_search(model, F_obs: np.ndarray, n_nodes: int, *,
                                 M: int = 200_000, K: int = 10, batch_size: int = 4096, seed: int = 0
                                 ) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    best: List[Dict[str, Any]] = []
    seen = 0
    while seen < M:
        B = min(batch_size, M - seen)
        taus = random_tau_batch(rng, B, n_nodes)
        logits, probs = score_taus_batch(model, taus, F_obs)
        idx = np.argsort(-probs)[: min(K, B)]
        cand = []
        for i in idx.tolist():
            tau = taus[i]
            order = np.argsort(tau).astype(int).tolist()
            cand.append({"prob": float(probs[i]), "logit": float(logits[i]), "order": order, "tau": tau.astype(float).tolist()})
        best.extend(cand)
        best = sorted(best, key=lambda x: x["prob"], reverse=True)[:K]
        seen += B
    return best
