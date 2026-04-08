# model/s1_compare_experiment.py
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore
from model.query_split import build_kw_lr_bitset, split_queries_by_kw_lr_bitset


def csr_to_dense_topd(
    X_csr,
    max_dim: int,
    *,
    # --- graph smoothing (optional) ---
    edge_index=None,          # shape [2, E], torch tensor or np.ndarray
    edge_weight=None,         # shape [E], torch tensor or np.ndarray
    smooth_steps: int = 0,    # t in scheme A; 0 means "no smoothing"
    alpha: float = 0.5,       # keep self info ratio
    sym: bool = True,         # W = W + W^T
    eps: float = 1e-12,
):
    """
    Convert CSR sparse matrix (N x M) into dense (N x D) using top-D columns by global frequency,
    and optionally apply weighted-graph smoothing using (edge_index, edge_weight).

    - If smooth_steps == 0 or edge_index/edge_weight is None: returns densified X only.
    - Otherwise: returns smoothed dense X.

    Requirements:
      - X_csr: scipy.sparse.csr_matrix
      - edge_index/edge_weight: (optional) graph over N nodes (rows of X_csr)
    """
    # ----------------------------
    # 1) densify by selecting top-D columns
    # ----------------------------
    if max_dim is None or max_dim <= 0:
        # return 0-dim dense
        N = X_csr.shape[0]
        X_dense = np.zeros((N, 0), dtype=np.float32)
    else:
        # pick top columns by global sum (fast, typical)
        # X_csr.sum(axis=0) -> 1 x M
        col_sum = np.asarray(X_csr.sum(axis=0)).ravel()
        if col_sum.size <= max_dim:
            cols = np.arange(col_sum.size, dtype=np.int64)
        else:
            # argpartition for top max_dim
            cols = np.argpartition(-col_sum, kth=max_dim - 1)[:max_dim]
            # optional: sort by descending sum for determinism
            cols = cols[np.argsort(-col_sum[cols])]

        X_dense = X_csr[:, cols].toarray().astype(np.float32, copy=False)

    # ----------------------------
    # 2) optional graph smoothing / propagation
    # ----------------------------
    if smooth_steps <= 0 or edge_index is None or edge_weight is None:
        return X_dense

    # Convert edge_index/edge_weight to numpy
    try:
        import torch
        is_torch = isinstance(edge_index, torch.Tensor)
    except Exception:
        is_torch = False

    if is_torch:
        row = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
        col = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
        w = edge_weight.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        edge_index_np = np.asarray(edge_index)
        row = edge_index_np[0].astype(np.int64, copy=False)
        col = edge_index_np[1].astype(np.int64, copy=False)
        w = np.asarray(edge_weight).astype(np.float32, copy=False)

    N = X_dense.shape[0]
    if N == 0 or X_dense.shape[1] == 0:
        return X_dense

    # IMPORTANT: graph must be over N nodes (rows of X_csr)
    # If not, raise early to avoid silent wrong smoothing.
    if row.max(initial=-1) >= N or col.max(initial=-1) >= N or row.min(initial=0) < 0 or col.min(initial=0) < 0:
        raise ValueError(
            f"edge_index out of range: node count N={N}, "
            f"row in [{row.min()},{row.max()}], col in [{col.min()},{col.max()}]"
        )

    # Build row-normalized adjacency operator P implicitly (no scipy dependency)
    # deg[i] = sum_j W[i,j]
    deg = np.zeros((N,), dtype=np.float32)

    # accumulate degrees (directed first)
    np.add.at(deg, row, w)
    if sym:
        np.add.at(deg, col, w)  # because W + W^T adds weight to opposite direction

    deg = deg + eps
    invdeg = 1.0 / deg

    Xg = X_dense.copy()

    # Helper: Y = P @ X, where P = D^{-1} W (or D^{-1}(W+W^T))
    def apply_P(X):
        Y = np.zeros_like(X, dtype=np.float32)

        # directed contribution: row <- col with weight w
        # Y[row] += w * X[col]
        np.add.at(Y, row, (w[:, None] * X[col]))

        if sym:
            # symmetric extra: Y[col] += w * X[row]  (from W^T)
            np.add.at(Y, col, (w[:, None] * X[row]))

        # row normalize
        Y *= invdeg[:, None]
        return Y

    # propagation
    a = float(alpha)
    for _ in range(int(smooth_steps)):
        Xg = a * Xg + (1.0 - a) * apply_P(Xg)

    return Xg


def aow_row_normalized_dense(Aow: csr_matrix, max_dim: int = 256) -> np.ndarray:
    X = csr_to_dense_topd(Aow, max_dim=max_dim)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def route_queries_by_keywords(
    Aow_sample: csr_matrix,         # (n_s, d)
    Aqw_node: csr_matrix,           # (m, d)
    cluster_id: np.ndarray,         # (n_s,)
    query_idx_global: np.ndarray,   # (m,) global ids
    drop_if_neither: bool = True,
):
    left_local = np.where(cluster_id == 0)[0].astype(np.int64)
    right_local = np.where(cluster_id == 1)[0].astype(np.int64)

    left_kw = np.unique(Aow_sample[left_local].indices).astype(np.int64) if left_local.size else np.array([], dtype=np.int64)
    right_kw = np.unique(Aow_sample[right_local].indices).astype(np.int64) if right_local.size else np.array([], dtype=np.int64)

    kw_ids, kw_lr_bits = build_kw_lr_bitset(left_kw, right_kw)
    q_left_g, q_right_g = split_queries_by_kw_lr_bitset(
        Aqw_node=Aqw_node,
        kw_ids=kw_ids,
        kw_lr_bits=kw_lr_bits,
        query_idx_global=query_idx_global,
        drop_if_neither=drop_if_neither
    )
    return q_left_g, q_right_g


def cost_curve_for_two_means(
    X_dense: np.ndarray,
    Aow_sample: csr_matrix,               # (n_s, d)
    Aqw_node: csr_matrix,                 # (m, d)
    obj_idx_global_sample: np.ndarray,    # (n_s,) global obj ids
    query_idx_global: np.ndarray,         # (m,) global query ids
    node_cost_fn,
    A_ow_all: csr_matrix,
    A_qw_all: csr_matrix,
    iters: int = 300,
    seed: int = 0,
):
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    n = X_dense.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64), (time.perf_counter() - t0)

    i1, i2 = rng.choice(n, size=2, replace=False)
    mu1 = X_dense[i1].copy()
    mu2 = X_dense[i2].copy()

    cid = np.zeros(n, dtype=np.int64)
    curve = []

    for _ in range(iters):
        d1 = np.sum((X_dense - mu1) ** 2, axis=1)
        d2 = np.sum((X_dense - mu2) ** 2, axis=1)
        new_cid = (d2 < d1).astype(np.int64)

        # compute cost_children
        if not (np.all(new_cid == 0) or np.all(new_cid == 1)):
            qL, qR = route_queries_by_keywords(Aow_sample, Aqw_node, new_cid, query_idx_global, drop_if_neither=True)
            objL = obj_idx_global_sample[new_cid == 0]
            objR = obj_idx_global_sample[new_cid == 1]
            c = float(node_cost_fn(A_ow_all, A_qw_all, objL, qL) + node_cost_fn(A_ow_all, A_qw_all, objR, qR))
            curve.append(c)
        else:
            curve.append(np.nan)

        if np.all(new_cid == cid):
            cid = new_cid
            # break
        cid = new_cid

        if np.any(cid == 0):
            mu1 = X_dense[cid == 0].mean(axis=0)
        if np.any(cid == 1):
            mu2 = X_dense[cid == 1].mean(axis=0)

        if not np.any(cid == 0) or not np.any(cid == 1):
            i1, i2 = rng.choice(n, size=2, replace=False)
            mu1 = X_dense[i1].copy()
            mu2 = X_dense[i2].copy()
            cid[:] = 0

    t1 = time.perf_counter()
    return np.array(curve, dtype=np.float64), (t1 - t0)
