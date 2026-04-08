# model/tree_cost.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore

import parameter as param


def _keywords_in_objects(A_ow_all: csr_matrix, obj_idx: np.ndarray) -> int:
    sub = A_ow_all[obj_idx]
    if sub.nnz == 0:
        return 0
    return int(np.unique(sub.indices).size)


def _sum_q_len_minus1(A_qw_all: csr_matrix, q_idx: np.ndarray) -> int:
    if q_idx.size == 0:
        return 0
    Aqw = A_qw_all[q_idx]
    q_lens = np.diff(Aqw.indptr).astype(np.int64)  # each row nnz = |q|
    return int(np.maximum(q_lens - 1, 0).sum())


def _calc_Rq_sum(A_ow_all: csr_matrix, A_qw_all: csr_matrix,
                 obj_idx: np.ndarray, q_idx: np.ndarray) -> int:
    """
    Sum over queries of |R_q|, computed exactly by postings intersection.
    (same logic as your node_cost)
    """
    O = int(obj_idx.size)
    if O == 0 or q_idx.size == 0:
        return 0

    Aow = A_ow_all[obj_idx].tocsc()
    indptr = Aow.indptr
    indices = Aow.indices

    total = 0
    for q in q_idx:
        term_ids = A_qw_all[q].indices
        if term_ids.size == 0:
            continue
        postings = []
        for t in term_ids:
            s = indptr[t]
            e = indptr[t + 1]
            postings.append(indices[s:e])
        postings.sort(key=lambda x: x.size)
        if postings[0].size == 0:
            continue
        hit = postings[0]
        for pl in postings[1:]:
            if hit.size == 0:
                break
            hit = np.intersect1d(hit, pl, assume_unique=False)
        total += int(hit.size)
    return int(total)


def node_cost_from_indices(A_ow_all: csr_matrix, A_qw_all: csr_matrix,
                           obj_idx: np.ndarray, q_idx: np.ndarray,
                           branches: int = 1,
                           sum_ws: int | None = None) -> float:
    """
    Cost for a node with 'branches' children (used for break-tree internal nodes).
    If branches==1, it's basically the original node cost shape (but still uses Eq.33/34 style).

    Storage: omega*branches + 32*omega + 32*branches
    Query: eps*|W| + alfa*sum(|q|-1)*branches + beta*branches + gama*(sum_ws)
      - sum_ws: sum of WS of its child subtrees (DP stores it)
        if None, we approximate with |W| (works for leaves / simple use)
    """
    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    q_idx = np.asarray(q_idx, dtype=np.int64)
    O = int(obj_idx.size)
    WN = int(q_idx.size)
    if O == 0 or WN == 0:
        return 0.0

    omega = _keywords_in_objects(A_ow_all, obj_idx)
    sum_qm1 = _sum_q_len_minus1(A_qw_all, q_idx)

    if sum_ws is None:
        sum_ws = WN

    storage = omega * branches + 32 * omega + 32 * branches
    query = (param.EPSILON * WN
             + param.ALFA * sum_qm1 * branches
             + param.BETA * branches
             + param.GAMA * int(sum_ws))

    return float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)


def tree_cost_from_builder_meta(builder,
                                root_id: int,
                                A_ow_all: csr_matrix,
                                A_qw_all: csr_matrix) -> float:
    """
    Total cost of the ORIGINAL tree (binary) using builder._meta and saved node files.
    We treat every internal node as retained (the tree you built).
    """
    meta = builder._meta
    out_dir = builder.out_dir

    def dfs(node_id: int) -> float:
        m = meta[node_id]
        if m.is_leaf:
            # leaf: objects are stored in file
            obj_path = os.path.join(out_dir, m.data["obj_ids"])
            obj_idx = np.load(obj_path).astype(np.int64)
            O = int(obj_idx.size)
            # queries for leaf are not stored; we use WN from meta if exists, else 0
            WN = int(m.data.get("WN", 0))
            # so for original-tree cost comparison, use the SAME formula as DP uses: rely on omega/WN/sum_qm1.
            omega = int(m.data.get("omega", 0))
            sum_qm1 = int(m.data.get("sum_q_len_minus1", 0))
            storage = omega * O + 32 * omega + 32 * O
            query = (param.EPSILON * WN
                     + param.ALFA * sum_qm1 * O
                     + param.BETA * O
                     + param.GAMA * WN)
            return float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)

        # internal
        ch_path = os.path.join(out_dir, m.data["children_ids"])
        ch = np.load(ch_path).astype(np.int64)
        child_cost = sum(dfs(int(c)) for c in ch.tolist())

        omega = int(m.data.get("omega", 0))
        WN = int(m.data.get("WN", 0))
        sum_qm1 = int(m.data.get("sum_q_len_minus1", 0))
        branches = int(ch.size)

        # for original tree retained, sum_ws = |W| (same as DP sets for retained)
        storage = omega * branches + 32 * omega + 32 * branches
        query = (param.EPSILON * WN
                 + param.ALFA * sum_qm1 * branches
                 + param.BETA * branches
                 + param.GAMA * WN)
        node_cost = float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)

        return node_cost + child_cost

    return float(dfs(root_id))


def tree_cost_from_treejson(tree_dir: str,
                            A_ow_all: csr_matrix,
                            A_qw_all: csr_matrix) -> float:
    """
    Total cost of the BREAK tree (multi-branch) from its tree.json + saved files.
    Uses the SAME scalar summary-form as DP (omega/WN/sum_qm1) if present in json.
    If not present, we still compute internal cost from children count & omega by reading kw_ids.npy sizes,
    but WN/sum_qm1 will be 0 -> recommend you keep them in json if you want accurate cost.
    """
    import json as _json
    with open(os.path.join(tree_dir, "tree.json"), "r", encoding="utf-8") as f:
        j = _json.load(f)

    nodes = {int(n["node_id"]): n for n in j["nodes"]}
    root_id = int(j["root"])

    def dfs(nid: int) -> float:
        nd = nodes[nid]
        is_leaf = bool(nd["is_leaf"])
        data = nd["data"]

        if is_leaf:
            obj_ids = np.load(os.path.join(tree_dir, data["obj_ids"])).astype(np.int64)
            O = int(obj_ids.size)
            # leaf cost uses omega/WN/sum_qm1 if available (else infer omega from kw file length)
            omega = int(data.get("omega", 0))
            if omega == 0 and os.path.exists(os.path.join(tree_dir, data["kw_ids"])):
                omega = int(np.load(os.path.join(tree_dir, data["kw_ids"])).size)
            WN = int(data.get("WN", 0))
            sum_qm1 = int(data.get("sum_q_len_minus1", 0))
            storage = omega * O + 32 * omega + 32 * O
            query = (param.EPSILON * WN
                     + param.ALFA * sum_qm1 * O
                     + param.BETA * O
                     + param.GAMA * WN)
            return float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)

        # internal
        ch = np.load(os.path.join(tree_dir, data["children_ids"])).astype(np.int64)
        branches = int(ch.size)
        child_cost = sum(dfs(int(c)) for c in ch.tolist())

        omega = int(data.get("omega", 0))
        if omega == 0 and os.path.exists(os.path.join(tree_dir, data["kw_ids"])):
            omega = int(np.load(os.path.join(tree_dir, data["kw_ids"])).size)
        WN = int(data.get("WN", 0))
        sum_qm1 = int(data.get("sum_q_len_minus1", 0))

        storage = omega * branches + 32 * omega + 32 * branches
        query = (param.EPSILON * WN
                 + param.ALFA * sum_qm1 * branches
                 + param.BETA * branches
                 + param.GAMA * WN)
        node_cost = float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)

        return node_cost + child_cost

    return float(dfs(root_id))
