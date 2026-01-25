from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import networkx as nx

@dataclass
class GraphInstance:
    graph_id: str
    graph: nx.Graph
    tau_true: np.ndarray  # (n,)
    tau_hat: np.ndarray   # (n,)
    F_hat: np.ndarray     # (dF,)

@dataclass
class Sample:
    graph_id: str
    tau_input: np.ndarray  # (n,)
    label: int             # 0/1
    x: np.ndarray          # (n+dF,)

def make_pos_neg_samples(inst: GraphInstance, rng: np.random.Generator) -> List[Sample]:
    n = inst.tau_true.shape[0]

    tau_pos = inst.tau_true.astype(np.float32)
    x_pos = np.concatenate([tau_pos, inst.F_hat.astype(np.float32)], axis=0)
    s_pos = Sample(graph_id=inst.graph_id, tau_input=tau_pos, label=1, x=x_pos)

    perm = rng.permutation(n)
    tau_neg = inst.tau_true[perm].astype(np.float32)
    x_neg = np.concatenate([tau_neg, inst.F_hat.astype(np.float32)], axis=0)
    s_neg = Sample(graph_id=inst.graph_id, tau_input=tau_neg, label=0, x=x_neg)

    return [s_pos, s_neg]

def samples_to_arrays(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    y = np.array([s.label for s in samples], dtype=np.float32)
    gids = [s.graph_id for s in samples]
    return X, y, gids
