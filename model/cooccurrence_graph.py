# model/cooccurrence_graph.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import parameter as param

def _row_nonzero_cols(A, i: int) -> np.ndarray:
    """Return nonzero column indices of row i for csr matrix or dense list."""
    if hasattr(A, "getrow"):  # scipy csr
        return A.getrow(i).indices
    # dense
    return np.array([j for j, v in enumerate(A[i]) if v == 1], dtype=np.int32)

def _ensure_csc(A):
    """Convert to CSC for fast column postings, if scipy is available."""
    if not hasattr(A, "tocsc"):
        return None
    return A.tocsc()

def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersect two sorted int arrays."""
    return np.intersect1d(a, b, assume_unique=False)

def _build_hit_matrix_H(A_ow, A_qw) -> Any:
    """
    Build H in shape [n_objects, m_queries], where H[i, q]=1 iff query q hits object i.
    Hit condition: Ω_q ⊆ Ω_i  (conjunctive keyword query)
    """
    if A_qw is None:
        return None

    # sizes
    if hasattr(A_ow, "shape"):
        n, d = A_ow.shape
    else:
        n = len(A_ow)
        d = len(A_ow[0]) if n > 0 else 0

    if hasattr(A_qw, "shape"):
        m = A_qw.shape[0]
    else:
        m = len(A_qw)

    # Use CSC to get postings list of each keyword quickly
    A_ow_csc = _ensure_csc(A_ow)
    if A_ow_csc is None:
        raise RuntimeError("pip install scipy")

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []

    for q in range(m):
        terms = _row_nonzero_cols(A_qw, q)
        if terms.size == 0:
            continue

        # postings lists: objects that contain term t
        postings_list = []
        for t in terms:
            start = A_ow_csc.indptr[t]
            end = A_ow_csc.indptr[t + 1]
            postings_list.append(A_ow_csc.indices[start:end])  # sorted by construction

        # intersect postings (start from shortest)
        postings_list.sort(key=lambda x: x.size)
        cand = postings_list[0]
        for pl in postings_list[1:]:
            if cand.size == 0:
                break
            cand = _intersect_sorted(cand, pl)

        # cand are object ids hit by query q
        for obj_id in cand:
            rows.append(int(obj_id))
            cols.append(int(q))
            data.append(1)

    # Build sparse CSR matrix H
    from scipy.sparse import csr_matrix  # type: ignore
    H = csr_matrix((data, (rows, cols)), shape=(n, m), dtype=np.int8)
    return H

def _topk_per_row_from_sparse(S, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given sparse similarity matrix S (CSR), pick top-K neighbors per row (excluding self).
    Return edge_index (2,E) and edge_weight (E,).
    """
    S = S.tocsr()
    n = S.shape[0]
    src_list = []
    dst_list = []
    w_list = []

    for i in range(n):
        row_start = S.indptr[i]
        row_end = S.indptr[i + 1]
        cols = S.indices[row_start:row_end]
        vals = S.data[row_start:row_end]

        # remove self-loop if exists
        mask = cols != i
        cols = cols[mask]
        vals = vals[mask]

        if cols.size == 0:
            continue

        if cols.size > K:
            top_idx = np.argpartition(vals, -K)[-K:]
            cols = cols[top_idx]
            vals = vals[top_idx]

            # sort descending for readability
            order = np.argsort(-vals)
            cols = cols[order]
            vals = vals[order]

        for c, v in zip(cols, vals):
            src_list.append(i)
            dst_list.append(int(c))
            w_list.append(float(v))

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(w_list, dtype=np.float32)
    return edge_index, edge_weight

def build_cooccurrence_graph(
    A_ow,
    A_qw,
    K: int,
    symmetrize: bool = True,
) -> Dict[str, Any]:
    """
    Build co-occurrence graph G=(V,E) for sampled objects.

    Inputs:
      A_ow: [n,d] object-keyword binary matrix
      A_qw: [m,d] query-keyword binary matrix
      K: keep top-K edges per node
      w_s, w_q: weights in Eq.(14)
      symmetrize: make graph undirected by max(S, S^T)

    Outputs (for GNN, e.g., PyG):
      X: node feature matrix (use A_ow)
      edge_index: shape [2,E]
      edge_weight: shape [E]
      (optional) H: object-query hit matrix
    """
    if K <= 0:
        raise ValueError("K must be positive")

    # sizes
    n = A_ow.shape[0] if hasattr(A_ow, "shape") else len(A_ow)

    # ---- (A) keyword Jaccard part ----
    # inter_kw = A_ow * A_ow^T  (shared keyword counts)
    # union_kw = |Ωi|+|Ωj|-inter
    from scipy.sparse import csr_matrix  # type: ignore
    A_ow_csr = A_ow.tocsr() if hasattr(A_ow, "tocsr") else csr_matrix(np.array(A_ow))
    kw_cnt = np.asarray(A_ow_csr.sum(axis=1)).reshape(-1)  # (n,)

    inter_kw = (A_ow_csr @ A_ow_csr.T).tocsr()

    # compute Jaccard on nonzeros only
    inter_kw = inter_kw.tocoo()
    union_kw = kw_cnt[inter_kw.row] + kw_cnt[inter_kw.col] - inter_kw.data
    j_kw_data = np.divide(
        inter_kw.data,
        union_kw,
        out=np.zeros_like(inter_kw.data, dtype=np.float32),
        where=union_kw != 0,
    )
    J_kw = csr_matrix((j_kw_data, (inter_kw.row, inter_kw.col)), shape=(n, n))

    # ---- (B) query-hit Jaccard part ----
    H = _build_hit_matrix_H(A_ow_csr, A_qw)  # (n,m) csr
    if H is None:
        J_q = csr_matrix((n, n), dtype=np.float32)
        q_cnt = np.zeros(n, dtype=np.float32)
    else:
        q_cnt = np.asarray(H.sum(axis=1)).reshape(-1)  # |Qi|
        inter_q = (H @ H.T).tocoo()
        union_q = q_cnt[inter_q.row] + q_cnt[inter_q.col] - inter_q.data
        j_q_data = np.divide(
            inter_q.data,
            union_q,
            out=np.zeros_like(inter_q.data, dtype=np.float32),
            where=union_q != 0,
        )
        J_q = csr_matrix((j_q_data, (inter_q.row, inter_q.col)), shape=(n, n))

    # ---- (C) combined similarity (Eq. 14) ----
    S = (param.w_s * param.W_S * J_kw) + (param.w_q * param.W_Q * J_q)

    # optional: symmetrize for undirected graph
    if symmetrize:
        S = S.maximum(S.T)

    # keep topK edges per node
    edge_index, edge_weight = _topk_per_row_from_sparse(S, K)

    return {
        "X": A_ow_csr,               # node features: keywords
        "edge_index": edge_index,    # [2,E]
        "edge_weight": edge_weight,  # [E]
        "H": H,                      # (optional) hits matrix
    }
