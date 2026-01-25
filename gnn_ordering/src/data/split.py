from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .dataset import GraphInstance

def split_by_graph(instances: List[GraphInstance], train_ratio: float, val_ratio: float, seed: int
                   ) -> Tuple[List[GraphInstance], List[GraphInstance], List[GraphInstance]]:
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1
    rng = np.random.default_rng(seed)
    idx = np.arange(len(instances))
    rng.shuffle(idx)
    n_train = int(len(instances) * train_ratio)
    n_val = int(len(instances) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    def sel(idxs): return [instances[i] for i in idxs]
    return sel(train_idx), sel(val_idx), sel(test_idx)
