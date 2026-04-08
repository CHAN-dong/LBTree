# model/streaming_assign.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import parameter as parm
import time

@dataclass
class StreamingAssignResult:
    obj_left: np.ndarray
    obj_right: np.ndarray
    q_left: np.ndarray
    q_right: np.ndarray
    Ew: torch.Tensor              # [V, emb_dim]
    z_o: torch.Tensor             # [n, emb_dim]
    z_q: torch.Tensor             # [m, emb_dim]


def _csr_row_indices(mat_csr, row: int) -> np.ndarray:
    """Return nonzero col indices of a CSR row."""
    start = mat_csr.indptr[row]
    end = mat_csr.indptr[row + 1]
    return mat_csr.indices[start:end]


def _mean_pool_rows(mat_csr, Ew: torch.Tensor, device: str) -> torch.Tensor:
    """
    Given a CSR binary matrix A (rows are sets of keyword ids),
    return z (row embeddings) by mean-pooling keyword embeddings (Eq.25).
    """
    n = mat_csr.shape[0]
    emb_dim = Ew.shape[1]
    z = torch.zeros((n, emb_dim), dtype=torch.float32, device=device)

    for i in range(n):
        cols = _csr_row_indices(mat_csr, i)
        if cols.size == 0:
            continue
        idx = torch.from_numpy(cols.astype(np.int64)).to(device)
        z[i] = Ew.index_select(0, idx).mean(dim=0)
    return z


def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Cosine similarity for a batch: a:[N,d], b:[d] -> [N]."""
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm() + eps)
    return (a_norm * b_norm).sum(dim=1)

from typing import Optional, Sequence
from collections import defaultdict
import numpy as np


def streaming_assign(
    A_ow_csr,                    # scipy.sparse.csr_matrix (n, V)
    A_qw_csr,                    # scipy.sparse.csr_matrix (m, V)
    seed_obj_left: Sequence[int],
    seed_obj_right: Sequence[int],
    seed_q_left: Sequence[int],
    seed_q_right: Sequence[int],
    w_s: float = 1.0,             # unused, kept for compatibility
    w_q: float = 1.0,             # unused, kept for compatibility
    emb_dim: int = 64,            # unused, kept for compatibility
    seed: int = 42,               # unused, kept for compatibility
    device: Optional[str] = None, # unused, kept for compatibility
    assign_all_queries_by_center: bool = False,  # unused

    # must match node_cost(...)
    epsilon: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,

    # optional
    stream_balance_lambda: float = 0.0,
) -> StreamingAssignResult:
    """
    Greedy cost-driven streaming assign.

    IMPORTANT:
    - initial query sets are EXACTLY seed_q_left / seed_q_right
    - when adding an object to one child, we consider:
        (1) storage delta
        (2) delta on queries already in that child
        (3) delta caused by newly-activated queries that become feasible in that child
    """

    import numpy as np
    from collections import defaultdict

    # ------------------------------------------------------------------
    # canonicalize CSR once: then each row indices are already unique/sorted
    # ------------------------------------------------------------------
    if not A_ow_csr.has_canonical_format:
        A_ow_csr = A_ow_csr.copy()
        A_ow_csr.sum_duplicates()
        A_ow_csr.sort_indices()

    if not A_qw_csr.has_canonical_format:
        A_qw_csr = A_qw_csr.copy()
        A_qw_csr.sum_duplicates()
        A_qw_csr.sort_indices()

    n, V = A_ow_csr.shape
    m = A_qw_csr.shape[0]

    seed_obj_left = np.asarray(seed_obj_left, dtype=np.int64)
    seed_obj_right = np.asarray(seed_obj_right, dtype=np.int64)
    seed_q_left = np.asarray(seed_q_left, dtype=np.int64)
    seed_q_right = np.asarray(seed_q_right, dtype=np.int64)

    # ------------------------------------------------------------------
    # helpers / cached locals
    # ------------------------------------------------------------------
    A_ow_indptr = A_ow_csr.indptr
    A_ow_indices = A_ow_csr.indices
    A_qw_indptr = A_qw_csr.indptr
    A_qw_indices = A_qw_csr.indices

    def _row_indices_unique_obj(row_id: int) -> np.ndarray:
        st, ed = A_ow_indptr[row_id], A_ow_indptr[row_id + 1]
        if ed <= st:
            return np.empty(0, dtype=np.int64)
        return A_ow_indices[st:ed]

    def _row_indices_unique_query(row_id: int) -> np.ndarray:
        st, ed = A_qw_indptr[row_id], A_qw_indptr[row_id + 1]
        if ed <= st:
            return np.empty(0, dtype=np.int64)
        return A_qw_indices[st:ed]

    q_len = np.diff(A_qw_indptr).astype(np.int32)
    q_qm1 = np.maximum(q_len - 1, 0).astype(np.float64)

    # keyword -> queries inverted lists
    kw_to_queries = defaultdict(list)
    for qid in range(m):
        qkws = _row_indices_unique_query(qid)
        for kw in qkws:
            kw_to_queries[int(kw)].append(int(qid))
    kw_to_queries_get = kw_to_queries.get

    # reusable temp arrays (avoid repeated defaultdict / boolean mask allocations)
    tmp_hit_cnt = np.zeros(m, dtype=np.int32)
    tmp_dec_cnt = np.zeros(m, dtype=np.int32)

    alpha_f = float(alpha)
    beta_f = float(beta)
    gamma_f = float(gamma)
    epsilon_f = float(epsilon)

    # keep original behavior as much as possible:
    # if parm exists outside, use it; otherwise fall back to neutral weights
    try:
        WS = float(parm.w_s) * float(parm.W_S)
        WQ = float(parm.w_q) * float(parm.W_Q)
    except Exception:
        WS = 1.0
        WQ = 1.0

    def _build_kw_set_from_objs(seed_objs: np.ndarray):
        kw_set = set()
        for oid in seed_objs:
            okws = _row_indices_unique_obj(int(oid))
            if okws.size > 0:
                kw_set.update(map(int, okws))
        return kw_set

    def _build_miss_cnt_from_kw_set(kw_set):
        """
        miss_cnt[q] = how many keywords of q are NOT covered by this child keyword-union
        """
        miss_cnt = q_len.copy()
        for kw in kw_set:
            for qid in kw_to_queries_get(int(kw), ()):
                if miss_cnt[qid] > 0:
                    miss_cnt[qid] -= 1
        return miss_cnt

    def _object_hit_queries(obj_kws: np.ndarray) -> np.ndarray:
        """
        Exact queries satisfied by THIS object:
            q subseteq obj
        """
        if obj_kws.size == 0:
            return np.empty(0, dtype=np.int64)

        obj_kw_num = int(obj_kws.size)
        touched = []

        for kw in obj_kws:
            for qid in kw_to_queries_get(int(kw), ()):
                if q_len[qid] <= obj_kw_num:
                    if tmp_hit_cnt[qid] == 0:
                        touched.append(qid)
                    tmp_hit_cnt[qid] += 1

        if not touched:
            return np.empty(0, dtype=np.int64)

        hit = [qid for qid in touched if tmp_hit_cnt[qid] == q_len[qid]]

        for qid in touched:
            tmp_hit_cnt[qid] = 0

        if not hit:
            return np.empty(0, dtype=np.int64)
        return np.asarray(hit, dtype=np.int64)

    def _build_state(seed_objs: np.ndarray, seed_qs: np.ndarray):
        """
        Child state.
        active queries are initialized from seed_qs.
        """
        kw_set = _build_kw_set_from_objs(seed_objs)
        miss_cnt = _build_miss_cnt_from_kw_set(kw_set)

        active = np.zeros(m, dtype=bool)
        if seed_qs.size > 0:
            active[seed_qs] = True

        obj_list = [int(x) for x in seed_objs.tolist()]
        active_count = int(active.sum())
        active_sum_qm1 = float(q_qm1[active].sum())

        return {
            "obj_list": obj_list,
            "kw_set": kw_set,
            "O": int(len(obj_list)),
            "omega": int(len(kw_set)),
            "miss_cnt": miss_cnt,
            "active": active,
            "active_count": active_count,
            "active_sum_qm1": active_sum_qm1,
        }

    def _eval_delta(state, obj_kws: np.ndarray, hit_q: np.ndarray):
        """
        Evaluate incremental cost if this object is added to this child.
        """
        O = state["O"]
        omega = state["omega"]
        kw_set = state["kw_set"]
        miss_cnt = state["miss_cnt"]
        active = state["active"]

        # -------- new keywords introduced to this child --------
        new_kws = [int(kw) for kw in obj_kws if int(kw) not in kw_set]
        dOmega = len(new_kws)

        # -------- storage delta --------
        # S = |Omega|*|O| + 32|Omega| + 32|O|
        # add one object + dOmega new keywords:
        # DeltaS = omega + dOmega*(O + 33) + 32
        delta_storage = float(omega) + float(dOmega) * (float(O) + 33.0) + 32.0

        # -------- existing queries already in this child --------
        # For every active q:
        #   Δ = alpha(|q|-1) + beta + gamma * 1[q subseteq obj]
        if state["active_count"] > 0:
            hit_existing = int(active[hit_q].sum()) if hit_q.size > 0 else 0
            delta_query_existing = (
                alpha_f * float(state["active_sum_qm1"])
                + beta_f * float(state["active_count"])
                + gamma_f * float(hit_existing)
            )
        else:
            hit_existing = 0
            delta_query_existing = 0.0

        # -------- newly activated queries in this child --------
        if dOmega == 0:
            new_qs = np.empty(0, dtype=np.int64)
            num_new = 0
            sum_qm1_new = 0.0
            hit_new = 0
        else:
            touched = []
            for kw in new_kws:
                for qid in kw_to_queries_get(int(kw), ()):
                    if (not active[qid]) and (miss_cnt[qid] > 0):
                        if tmp_dec_cnt[qid] == 0:
                            touched.append(qid)
                        tmp_dec_cnt[qid] += 1

            if touched:
                new_q_list = []
                sum_qm1_new = 0.0

                for qid in touched:
                    if miss_cnt[qid] == tmp_dec_cnt[qid]:
                        new_q_list.append(int(qid))
                        sum_qm1_new += float(q_qm1[qid])

                if new_q_list:
                    new_qs = np.asarray(new_q_list, dtype=np.int64)
                    num_new = int(new_qs.size)

                    hit_new = 0
                    if hit_q.size > 0:
                        for qid in hit_q:
                            if (not active[qid]) and (tmp_dec_cnt[qid] > 0) and (miss_cnt[qid] == tmp_dec_cnt[qid]):
                                hit_new += 1
                else:
                    new_qs = np.empty(0, dtype=np.int64)
                    num_new = 0
                    sum_qm1_new = 0.0
                    hit_new = 0

                for qid in touched:
                    tmp_dec_cnt[qid] = 0
            else:
                new_qs = np.empty(0, dtype=np.int64)
                num_new = 0
                sum_qm1_new = 0.0
                hit_new = 0

        # newly activated q:
        # old cost in this child = 0
        # new cost in this child =
        #   epsilon + alpha(|q|-1)(O+1) + beta(O+1) + gamma * 1[q subseteq obj]
        delta_query_new = (
            epsilon_f * float(num_new)
            + float(O + 1) * (alpha_f * float(sum_qm1_new) + beta_f * float(num_new))
            + gamma_f * float(hit_new)
        )

        delta_total = parm.w_s * parm.W_S * delta_storage + parm.w_q * parm.W_Q * delta_query_existing + delta_query_new

        return delta_total, new_kws, new_qs

    def _apply_update(state, oid: int, new_kws, new_qs: np.ndarray):
        state["obj_list"].append(int(oid))
        state["O"] += 1

        if new_kws:
            kw_set = state["kw_set"]
            miss_cnt = state["miss_cnt"]
            for kw in new_kws:
                k = int(kw)
                if k in kw_set:
                    continue
                kw_set.add(k)
                for qid in kw_to_queries_get(k, ()):
                    if miss_cnt[qid] > 0:
                        miss_cnt[qid] -= 1
            state["omega"] = int(len(kw_set))

        if new_qs.size > 0:
            state["active"][new_qs] = True
            state["active_count"] += int(new_qs.size)
            state["active_sum_qm1"] += float(q_qm1[new_qs].sum())

    # ------------------------------------------------------------------
    # corner cases
    # ------------------------------------------------------------------
    if seed_obj_left.size == 0 and seed_obj_right.size == 0:
        return StreamingAssignResult(
            obj_left=np.array([], dtype=np.int64),
            obj_right=np.array([], dtype=np.int64),
            q_left=seed_q_left.copy(),
            q_right=seed_q_right.copy(),
            Ew=None,
            z_o=None,
            z_q=None,
        )

    if seed_obj_left.size == 0 and seed_obj_right.size > 0:
        return StreamingAssignResult(
            obj_left=np.array([], dtype=np.int64),
            obj_right=np.arange(n, dtype=np.int64),
            q_left=seed_q_left.copy(),
            q_right=seed_q_right.copy(),
            Ew=None,
            z_o=None,
            z_q=None,
        )

    if seed_obj_right.size == 0 and seed_obj_left.size > 0:
        return StreamingAssignResult(
            obj_left=np.arange(n, dtype=np.int64),
            obj_right=np.array([], dtype=np.int64),
            q_left=seed_q_left.copy(),
            q_right=seed_q_right.copy(),
            Ew=None,
            z_o=None,
            z_q=None,
        )

    # ------------------------------------------------------------------
    # initialize states
    # ------------------------------------------------------------------
    left = _build_state(seed_obj_left, seed_q_left)
    right = _build_state(seed_obj_right, seed_q_right)

    in_seed = np.zeros(n, dtype=bool)
    in_seed[seed_obj_left] = True
    in_seed[seed_obj_right] = True
    remaining = np.where(~in_seed)[0].astype(np.int64)

    # ------------------------------------------------------------------
    # greedy assign
    # ------------------------------------------------------------------
    for oid in remaining:
        obj_kws = _row_indices_unique_obj(int(oid))
        hit_q = _object_hit_queries(obj_kws)

        delta_left, new_kws_left, new_qs_left = _eval_delta(left, obj_kws, hit_q)
        delta_right, new_kws_right, new_qs_right = _eval_delta(right, obj_kws, hit_q)

        if stream_balance_lambda > 0.0:
            cur_imb = float((left["O"] - right["O"]) ** 2)
            imb_left = float(((left["O"] + 1) - right["O"]) ** 2)
            imb_right = float((left["O"] - (right["O"] + 1)) ** 2)
            delta_left += stream_balance_lambda * (imb_left - cur_imb)
            delta_right += stream_balance_lambda * (imb_right - cur_imb)

        if delta_left <= delta_right:
            _apply_update(left, int(oid), new_kws_left, new_qs_left)
        else:
            _apply_update(right, int(oid), new_kws_right, new_qs_right)

    obj_left = np.asarray(left["obj_list"], dtype=np.int64)
    obj_right = np.asarray(right["obj_list"], dtype=np.int64)

    q_left = np.where(left["active"])[0].astype(np.int64)
    q_right = np.where(right["active"])[0].astype(np.int64)

    return StreamingAssignResult(
        obj_left=obj_left,
        obj_right=obj_right,
        q_left=q_left,
        q_right=q_right,
        Ew=None,
        z_o=None,
        z_q=None,
    )



# # model/streaming_assign.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Sequence, Tuple, Optional, Dict
# import numpy as np
# import torch
# import torch.nn.functional as F


# @dataclass
# class StreamingAssignResult:
#     obj_left: np.ndarray
#     obj_right: np.ndarray
#     q_left: np.ndarray
#     q_right: np.ndarray
#     Ew: torch.Tensor              # [V, emb_dim]
#     z_o: torch.Tensor             # [n, emb_dim]
#     z_q: torch.Tensor             # [m, emb_dim]


# def _csr_row_indices(mat_csr, row: int) -> np.ndarray:
#     """Return nonzero col indices of a CSR row."""
#     start = mat_csr.indptr[row]
#     end = mat_csr.indptr[row + 1]
#     return mat_csr.indices[start:end]


# def _mean_pool_rows(mat_csr, Ew: torch.Tensor, device: str) -> torch.Tensor:
#     """
#     Given a CSR binary matrix A (rows are sets of keyword ids),
#     return z (row embeddings) by mean-pooling keyword embeddings (Eq.25).
#     """
#     n = mat_csr.shape[0]
#     emb_dim = Ew.shape[1]
#     z = torch.zeros((n, emb_dim), dtype=torch.float32, device=device)

#     for i in range(n):
#         cols = _csr_row_indices(mat_csr, i)
#         if cols.size == 0:
#             continue
#         idx = torch.from_numpy(cols.astype(np.int64)).to(device)
#         z[i] = Ew.index_select(0, idx).mean(dim=0)
#     return z


# def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
#     """Cosine similarity for a batch: a:[N,d], b:[d] -> [N]."""
#     a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
#     b_norm = b / (b.norm() + eps)
#     return (a_norm * b_norm).sum(dim=1)


# def streaming_assign(
#     A_ow_csr,                    # scipy.sparse.csr_matrix (n, V)
#     A_qw_csr,                    # scipy.sparse.csr_matrix (m, V)
#     seed_obj_left: Sequence[int],
#     seed_obj_right: Sequence[int],
#     seed_q_left: Sequence[int],
#     seed_q_right: Sequence[int],
#     w_s: float = 1.0,
#     w_q: float = 1.0,
#     emb_dim: int = 64,
#     seed: int = 42,
#     device: Optional[str] = None,
#     assign_all_queries_by_center: bool = False,
# ) -> StreamingAssignResult:
#     """
#     Implements Streaming-assign as in the paper:
#       - init keyword embedding Ew randomly
#       - compute z_o, z_q by mean pooling (Eq.25)
#       - compute centers mu^o_L/R, mu^q_L/R (Eq.26, Eq.27)
#       - stream remaining objects to the closer child using Eq.(28)(29)

#     Note: 论文的 S2 主要是把剩余 objects 分配到左右簇。
#           queries 的左右簇通常来自 S1 的 BCM 输出；这里可选 assign_all_queries_by_center=True
#           用中心把所有 queries 也重新分一遍（工程上方便调试）。
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     n, V = A_ow_csr.shape
#     m = A_qw_csr.shape[0]

#     # 1) init keyword embeddings Ew (random)  e_w ∈ R^d  
#     g = torch.Generator(device=device)
#     g.manual_seed(seed)
#     Ew = torch.randn((V, emb_dim), generator=g, device=device, dtype=torch.float32)

#     # 2) compute z_o, z_q by mean pooling (Eq.25) 
#     z_o = _mean_pool_rows(A_ow_csr, Ew, device)
#     z_q = _mean_pool_rows(A_qw_csr, Ew, device)

#     # seed sets
#     seed_obj_left = np.array(list(seed_obj_left), dtype=np.int64)
#     seed_obj_right = np.array(list(seed_obj_right), dtype=np.int64)
#     seed_q_left = np.array(list(seed_q_left), dtype=np.int64)
#     seed_q_right = np.array(list(seed_q_right), dtype=np.int64)

#     # ====== seed empty handling (your requested behavior) ======

#     # If both object seeds are empty: no split info => skip streaming-assign
#     # (Return empty partitions to signal "do nothing" to caller.)
#     if seed_obj_left.size == 0 and seed_obj_right.size == 0:
#         # queries: follow the same rule
#         if seed_q_left.size == 0 and seed_q_right.size == 0:
#             q_left = np.array([], dtype=np.int64)
#             q_right = np.array([], dtype=np.int64)
#         elif seed_q_left.size == 0:
#             q_left = np.array([], dtype=np.int64)
#             q_right = np.arange(m, dtype=np.int64)
#         elif seed_q_right.size == 0:
#             q_left = np.arange(m, dtype=np.int64)
#             q_right = np.array([], dtype=np.int64)
#         else:
#             q_left = seed_q_left.copy()
#             q_right = seed_q_right.copy()

#         return StreamingAssignResult(
#             obj_left=np.array([], dtype=np.int64),
#             obj_right=np.array([], dtype=np.int64),
#             q_left=q_left,
#             q_right=q_right,
#             Ew=Ew,
#             z_o=z_o,
#             z_q=z_q,
#         )

#     # If only one side object seed exists => all objects go to that side
#     if seed_obj_left.size == 0 and seed_obj_right.size > 0:
#         obj_left = np.array([], dtype=np.int64)
#         obj_right = np.arange(n, dtype=np.int64)
#     elif seed_obj_right.size == 0 and seed_obj_left.size > 0:
#         obj_left = np.arange(n, dtype=np.int64)
#         obj_right = np.array([], dtype=np.int64)
#     else:
#         obj_left = None   # will be computed by streaming
#         obj_right = None

#     # Queries seeds: if one side empty => all queries go to the other side
#     if seed_q_left.size == 0 and seed_q_right.size == 0:
#         q_left = np.array([], dtype=np.int64)
#         q_right = np.array([], dtype=np.int64)
#     elif seed_q_left.size == 0:
#         q_left = np.array([], dtype=np.int64)
#         q_right = np.arange(m, dtype=np.int64)
#     elif seed_q_right.size == 0:
#         q_left = np.arange(m, dtype=np.int64)
#         q_right = np.array([], dtype=np.int64)
#     else:
#         q_left = seed_q_left.copy()
#         q_right = seed_q_right.copy()

#     # If objects already decided (one-side seed case), we can early return.
#     if obj_left is not None and obj_right is not None:
#         return StreamingAssignResult(
#             obj_left=obj_left,
#             obj_right=obj_right,
#             q_left=q_left,
#             q_right=q_right,
#             Ew=Ew,
#             z_o=z_o,
#             z_q=z_q,
#         )

#     # 3) centers (Eq.26, Eq.27) 
#     mu_o_L = z_o[torch.from_numpy(seed_obj_left).to(device)].mean(dim=0)
#     mu_o_R = z_o[torch.from_numpy(seed_obj_right).to(device)].mean(dim=0)
#     mu_q_L = z_q[torch.from_numpy(seed_q_left).to(device)].mean(dim=0)
#     mu_q_R = z_q[torch.from_numpy(seed_q_right).to(device)].mean(dim=0)

#     # 4) stream remaining objects by Eq.(28)(29) 
#     in_seed = np.zeros(n, dtype=bool)
#     in_seed[seed_obj_left] = True
#     in_seed[seed_obj_right] = True
#     remaining = np.where(~in_seed)[0]

#     rem_t = torch.from_numpy(remaining.astype(np.int64)).to(device)
#     z_rem = z_o.index_select(0, rem_t)  # [nr, d]

#     sim_L = w_s * _cos(z_rem, mu_o_L) + w_q * _cos(z_rem, mu_q_L)
#     sim_R = w_s * _cos(z_rem, mu_o_R) + w_q * _cos(z_rem, mu_q_R)

#     go_left = (sim_L > sim_R).detach().cpu().numpy()

#     obj_left = np.concatenate([seed_obj_left, remaining[go_left]])
#     obj_right = np.concatenate([seed_obj_right, remaining[~go_left]])

#     # queries: 默认保持 seed_q_left/right 不变（因为论文里 workload 的 WL/WR 来自 S1）
#     q_left = seed_q_left.copy()
#     q_right = seed_q_right.copy()

#     # 可选：把所有 queries 也按中心重新分一遍（调试方便）
#     if assign_all_queries_by_center:
#         # 用 Eq.(28) 的形式把 query embedding 跟 query centers 比（工程简化）
#         zq_all = z_q  # [m,d]
#         simq_L = _cos(zq_all, mu_q_L)
#         simq_R = _cos(zq_all, mu_q_R)
#         q_left = np.where((simq_L > simq_R).detach().cpu().numpy())[0].astype(np.int64)
#         q_right = np.where((simq_R >= simq_L).detach().cpu().numpy())[0].astype(np.int64)

#     return StreamingAssignResult(
#         obj_left=obj_left,
#         obj_right=obj_right,
#         q_left=q_left,
#         q_right=q_right,
#         Ew=Ew,
#         z_o=z_o,
#         z_q=z_q,
#     )
