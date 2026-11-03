# model/query_split.py
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore

LEFT_BIT = 1   # 01
RIGHT_BIT = 2  # 10

# model/query_split.py 追加

def seed_queries_by_sample_clusters(
    Aow_sample_csr: csr_matrix,          # (n_s, d) sampled objects (local)
    Aqw_node: csr_matrix,                # (m_node, d) queries (local to node)
    cluster_id_sample: np.ndarray,        # (n_s,) 0/1 for sampled objects
    query_idx_global: np.ndarray | None = None,
    drop_if_neither: bool = True,
):
    """
    用 sampled objects 的聚类结果，构造左右关键词集合 -> 2-bit kw_lr_bits，
    然后用 split_queries_by_kw_lr_bitset 做 seed_q_left/right。

    这样 seed queries 的逻辑与后续递归的 query routing 完全一致，且比 majority-hit 更快。
    """
    cluster_id_sample = np.asarray(cluster_id_sample, dtype=np.int64)
    left_mask = (cluster_id_sample == 0)
    right_mask = ~left_mask

    # 左右 sampled 的 keyword ids
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
    """
    输入左右子簇的 keyword id 集合（升序/无序都行），输出：
      kw_ids: union keywords (sorted)
      kw_lr_bits: uint8 array, len=K
        bit0=1 => in_left
        bit1=1 => in_right
        so value in {0,1,2,3}
    """
    left_kw_ids = np.asarray(left_kw_ids, dtype=np.int64)
    right_kw_ids = np.asarray(right_kw_ids, dtype=np.int64)

    kw_ids = np.union1d(left_kw_ids, right_kw_ids).astype(np.int64)

    bits = np.zeros(kw_ids.shape[0], dtype=np.uint8)
    bits[np.searchsorted(kw_ids, left_kw_ids)] |= LEFT_BIT
    bits[np.searchsorted(kw_ids, right_kw_ids)] |= RIGHT_BIT

    return kw_ids, bits


def split_queries_by_kw_lr_bitset(
    Aqw_node: csr_matrix,                # (m_node, d) 当前节点的 query 子矩阵
    kw_ids: np.ndarray,                  # (K,) 当前节点 union keywords
    kw_lr_bits: np.ndarray,              # (K,) uint8, each in {0,1,2,3}
    query_idx_global: np.ndarray | None = None,  # (m_node,) local->global query id
    drop_if_neither: bool = True,
):
    """
    用 (kw_ids, kw_lr_bits) 做 query routing（overlap allowed）：
      left_ok  iff  query 的每个关键词都在 left（bit0=1）
      right_ok iff  query 的每个关键词都在 right（bit1=1）

    返回：
      q_left_global, q_right_global

    注意：这是 routing/necessary-condition，速度快，但不是精确命中判断。
    """
    kw_ids = np.asarray(kw_ids, dtype=np.int64)
    kw_lr_bits = np.asarray(kw_lr_bits, dtype=np.uint8)
    assert kw_ids.shape[0] == kw_lr_bits.shape[0]

    m_node = Aqw_node.shape[0]
    if query_idx_global is None:
        query_idx_global = np.arange(m_node, dtype=np.int64)
    else:
        query_idx_global = np.asarray(query_idx_global, dtype=np.int64)
        assert query_idx_global.size == m_node

    # A row-wise bitwise AND directly expresses keyword feasibility. Empty
    # queries stay unrouted, including consecutive empty CSR rows.
    lookup = np.zeros(Aqw_node.shape[1], dtype=np.uint8)
    valid = (kw_ids >= 0) & (kw_ids < lookup.size)
    lookup[kw_ids[valid]] = kw_lr_bits[valid]
    nonempty = np.diff(Aqw_node.indptr) > 0
    flags = np.zeros(m_node, dtype=np.uint8)
    if np.any(nonempty):
        flags[nonempty] = np.bitwise_and.reduceat(
            lookup[Aqw_node.indices], Aqw_node.indptr[:-1][nonempty])
    left_ok = (flags & LEFT_BIT) != 0
    right_ok = (flags & RIGHT_BIT) != 0

        # 如果两边都不满足，你可以选择丢弃或都下发
    if not drop_if_neither:
        neither = nonempty & ~left_ok & ~right_ok
        left_ok |= neither
        right_ok |= neither

    return query_idx_global[left_ok], query_idx_global[right_ok]


