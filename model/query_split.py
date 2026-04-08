# model/query_split.py
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore

LEFT_BIT = 1   # 01
RIGHT_BIT = 2  # 10

def seed_queries_by_sample_clusters(
    Aow_sample_csr: csr_matrix,          # (n_s, d) sampled objects (local)
    Aqw_node: csr_matrix,                # (m_node, d) queries (local to node)
    cluster_id_sample: np.ndarray,        # (n_s,) 0/1 for sampled objects
    query_idx_global: np.ndarray | None = None,
    drop_if_neither: bool = True,
):
    cluster_id_sample = np.asarray(cluster_id_sample, dtype=np.int64)
    left_mask = (cluster_id_sample == 0)
    right_mask = ~left_mask

    left_kw = np.unique(Aow_sample_csr[left_mask].indices).astype(np.int64)
    right_kw = np.unique(Aow_sample_csr[right_mask].indices).astype(np.int64)

    kw_ids, kw_lr_bits = build_kw_lr_bitset(left_kw, right_kw)

    q_left, q_right = split_queries_by_kw_lr_bitset(
        Aqw_node=Aqw_node,
        kw_ids=kw_ids,
        kw_lr_bits=kw_lr_bits,
        query_idx_global=query_idx_global,
        drop_if_neither=drop_if_neither,
    )
    return q_left, q_right, kw_ids, kw_lr_bits



def build_kw_lr_bitset(
    left_kw_ids: np.ndarray,
    right_kw_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_kw_ids = np.asarray(left_kw_ids, dtype=np.int64)
    right_kw_ids = np.asarray(right_kw_ids, dtype=np.int64)

    kw_ids = np.union1d(left_kw_ids, right_kw_ids).astype(np.int64)

    left_set = set(left_kw_ids.tolist())
    right_set = set(right_kw_ids.tolist())

    bits = np.zeros(kw_ids.shape[0], dtype=np.uint8)
    for i, kw in enumerate(kw_ids):
        b = 0
        if int(kw) in left_set:
            b |= LEFT_BIT
        if int(kw) in right_set:
            b |= RIGHT_BIT
        bits[i] = b

    return kw_ids, bits


def split_queries_by_kw_lr_bitset(
    Aqw_node: csr_matrix,                
    kw_ids: np.ndarray,                
    kw_lr_bits: np.ndarray,              
    query_idx_global: np.ndarray | None = None, 
    drop_if_neither: bool = True,
):
    kw_ids = np.asarray(kw_ids, dtype=np.int64)
    kw_lr_bits = np.asarray(kw_lr_bits, dtype=np.uint8)
    assert kw_ids.shape[0] == kw_lr_bits.shape[0]

    m_node = Aqw_node.shape[0]
    if query_idx_global is None:
        query_idx_global = np.arange(m_node, dtype=np.int64)
    else:
        query_idx_global = np.asarray(query_idx_global, dtype=np.int64)
        assert query_idx_global.size == m_node

    # keyword_id -> position in kw_ids
    kw_pos = {int(k): i for i, k in enumerate(kw_ids.tolist())}

    q_left = []
    q_right = []

    for local_q in range(m_node):
        term_ids = Aqw_node[local_q].indices
        if term_ids.size == 0:
            continue

        left_ok = True
        right_ok = True

        for t in term_ids:
            p = kw_pos.get(int(t), None)
            if p is None:
                left_ok = False
                right_ok = False
                break
            b = int(kw_lr_bits[p])
            if (b & LEFT_BIT) == 0:
                left_ok = False
            if (b & RIGHT_BIT) == 0:
                right_ok = False
            if (not left_ok) and (not right_ok):
                break

        gid = int(query_idx_global[local_q])
        if left_ok:
            q_left.append(gid)
        if right_ok:
            q_right.append(gid)

        if (not left_ok) and (not right_ok) and (not drop_if_neither):
            q_left.append(gid)
            q_right.append(gid)

    return np.array(q_left, dtype=np.int64), np.array(q_right, dtype=np.int64)


