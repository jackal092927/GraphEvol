from __future__ import annotations
from typing import Tuple, List
import numpy as np
import networkx as nx
import gudhi as gd
from persim import PersistenceImager

def simplex_tree_lower_star(G: nx.Graph, f: np.ndarray) -> gd.SimplexTree:
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([int(v)], filtration=float(f[int(v)]))
    for u, v in G.edges():
        u, v = int(u), int(v)
        st.insert([u, v], filtration=float(max(f[u], f[v])))
    st.make_filtration_non_decreasing()
    return st

def extended_persistence_diagrams(st: gd.SimplexTree):
    st_ext = gd.SimplexTree(st)
    st_ext.extend_filtration()
    ord_dgm, rel_dgm, extp_dgm, extm_dgm = st_ext.extended_persistence()
    return ord_dgm, rel_dgm, extp_dgm, extm_dgm

def diagrams_to_pi_vector(
    diagrams_parts,
    dims: Tuple[int, ...],
    bandwidth: float,
    resolution: Tuple[int, int],
    im_range: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Match v0_pipeline.ipynb exactly:
      - use gudhi.representations.PersistenceImage (NOT persim)
      - weight = lambda x: 1.0
      - fit_transform per diagram
    """
    from gudhi.representations import PersistenceImage
    import numpy as np

    def _safe_diag(dgm: np.ndarray) -> np.ndarray:
        if dgm is None:
            return np.empty((0, 2), dtype=float)
        dgm = np.asarray(dgm, dtype=float)
        if dgm.size == 0:
            return np.empty((0, 2), dtype=float)
        return dgm.reshape(-1, 2)

    x0, x1, y0, y1 = im_range
    pi = PersistenceImage(
        bandwidth=float(bandwidth),
        resolution=[int(resolution[0]), int(resolution[1])],
        im_range=[float(x0), float(x1), float(y0), float(y1)],
        weight=(lambda x: 1.0),
    )

    chunks = []
    for part in diagrams_parts:
        for d in dims:
            dgm = np.array([pair for (dim, pair) in part if int(dim) == int(d)], dtype=float)
            dgm = _safe_diag(dgm)
            vec = pi.fit_transform([dgm])[0]   # (H*W,)
            chunks.append(vec)

    return np.concatenate(chunks, axis=0).astype(np.float32)


