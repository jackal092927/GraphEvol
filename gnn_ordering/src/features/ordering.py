from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any
import json
import numpy as np
import networkx as nx

def degree_rank_ordering(G: nx.Graph) -> np.ndarray:
    nodes = np.array(list(G.nodes()), dtype=int)
    deg = np.array([G.degree(int(v)) for v in nodes], dtype=np.int64)
    order = np.lexsort((nodes, -deg))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(nodes))
    if len(nodes) <= 1:
        tau = np.zeros((len(nodes),), dtype=np.float64)
    else:
        tau = rank.astype(np.float64) / (len(nodes) - 1)
    if not np.array_equal(nodes, np.arange(len(nodes))):
        tau_by_node = np.zeros((len(nodes),), dtype=np.float64)
        for node, t in zip(nodes, tau):
            tau_by_node[int(node)] = float(t)
        return tau_by_node
    return tau

def order_to_tau(order: np.ndarray) -> np.ndarray:
    n = int(order.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    tau = np.empty((n,), dtype=np.float64)
    for r, node in enumerate(order.tolist()):
        tau[int(node)] = r / (n - 1)
    return tau

class OrderingProvider(Protocol):
    def get(self, G: nx.Graph, graph_id: str) -> np.ndarray: ...

@dataclass
class DegreeOrderingProvider:
    def get(self, G: nx.Graph, graph_id: str) -> np.ndarray:
        return degree_rank_ordering(G)

@dataclass
class FileOrderingProvider:
    path: str
    fallback: Optional[OrderingProvider] = None
    _cache: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        cache: Dict[str, Any] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cache[obj["graph_id"]] = obj
        self._cache = cache

    def get(self, G: nx.Graph, graph_id: str) -> np.ndarray:
        if self._cache is None:
            self._load()
        assert self._cache is not None
        if graph_id not in self._cache:
            if self.fallback is not None:
                return self.fallback.get(G, graph_id)
            raise KeyError(f"graph_id={graph_id} not found in {self.path}")
        obj = self._cache[graph_id]
        if "tau_hat" in obj:
            return np.array(obj["tau_hat"], dtype=np.float64)
        if "order" in obj:
            return order_to_tau(np.array(obj["order"], dtype=np.int64))
        raise ValueError(f"Entry for {graph_id} missing 'tau_hat' or 'order'")
