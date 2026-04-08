# model/model_split.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix  # type: ignore
from model.viz import embed_2d_svd, plot_overlay_s1_s2
import parameter as param
from model.cooccurrence_graph import build_cooccurrence_graph
from model.gcn_encoder import WeightedGCNEncoder, build_normalized_adj, csr_to_torch_sparse
from model.cluster_mlp import ClusterMLP, BinaryClusterMLP
from model.loss import clustering_loss
from model.streaming_assign import streaming_assign
from model.query_split import (
    build_kw_lr_bitset,
    split_queries_by_kw_lr_bitset,
    seed_queries_by_sample_clusters,
)
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from model.s1_compare_experiment import (
    csr_to_dense_topd,
    aow_row_normalized_dense,
    cost_curve_for_two_means,
    route_queries_by_keywords,
)
from parameter import w_s, w_q
from model.split_cache import SplitModelCache

def grad_report(named_params, title="grad"):
    total = 0.0
    cnt = 0
    none_cnt = 0
    max_g = 0.0
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            none_cnt += 1
            continue
        g = p.grad.detach()
        gn = g.norm().item()
        total += gn
        cnt += 1
        if gn > max_g:
            max_g = gn
    print(f"[{title}] params_with_grad={cnt}, grad_none={none_cnt}, grad_sum={total:.6f}, grad_max={max_g:.6f}")


# ---------- Cost computation (node-level) ----------
def node_cost(
    A_ow_all: csr_matrix,
    A_qw_all: csr_matrix,
    obj_idx: np.ndarray,
    query_idx: np.ndarray,
) -> float:
    """
    Node-level bitmap cost model (a usable implementation):
      Storage(B): |Ω_O|*|O| + 32|Ω_O| + 32|O|
      Query(B,W): sum_q [ ε + α(|q|-1)|O| + β|O| + γ|R_q| ]

    |Ω_O|: number of keywords appearing in this node's objects
    |R_q|: result size for query q over this node's objects (exact via postings intersection)
    """
    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    O = int(obj_idx.size)
    if O == 0 or query_idx.size == 0:
        return 0.0

    Aow = A_ow_all[obj_idx]  # (O, d)
    omega = int(np.asarray(Aow.sum(axis=0)).reshape(-1).astype(np.int64).nonzero()[0].size)

    storage = (omega * O) + (32 * omega) + (32 * O)

    # For |R_q|: exact postings intersection on this node
    Aow_csc = Aow.tocsc()
    qcost = 0.0

    sum_r = 0
    sum_q = 0

    for q in query_idx:
        q_row = A_qw_all[q]
        term_ids = q_row.indices
        q_len = int(term_ids.size)
        if q_len == 0:
            continue

        postings = []
        for t in term_ids:
            s = Aow_csc.indptr[t]
            e = Aow_csc.indptr[t + 1]
            postings.append(Aow_csc.indices[s:e])

        postings.sort(key=lambda x: x.size)
        if postings[0].size == 0:
            r_size = 0
        else:
            cand = postings[0]
            for pl in postings[1:]:
                if cand.size == 0:
                    break
                cand = np.intersect1d(cand, pl, assume_unique=False)
            r_size = int(cand.size)
        
        # print(f"outside: r_size: {r_size}, q_len: {q_len}")
        sum_r += r_size
        sum_q += q_len

    qcost = (
        query_idx.size * param.EPSILON
        + param.ALFA * sum_q * O
        + param.BETA * query_idx.size * O
        + param.GAMA * sum_r
    )

    # print(f"outside: ------sum_q: {sum_q}, sum_r: {sum_r}, O: {O}")

    return float(w_s * storage + w_q * qcost)


# ---------- Output container ----------
@dataclass
class SplitResult:
    obj_left: np.ndarray
    obj_right: np.ndarray
    q_left: np.ndarray
    q_right: np.ndarray
    cost_parent: float
    cost_children: float


def _uniq_kw_count(Aow_csr: csr_matrix, obj_local_idx: np.ndarray) -> int:
    if obj_local_idx.size == 0:
        return 0
    return int(np.unique(Aow_csr[obj_local_idx].indices).size)


TAU_START = 5.0
TAU_END   = 0.5
TAU_EPOCHS = 50
TAU_MIN = 0.1
def get_tau(epoch: int) -> float:
    if epoch < 50:
        return 5.0
    if epoch < 150:
        return 2.0
    return 1.0

def compute_sampling_ratios(
    Aow_node: csr_matrix,
    sample_local: np.ndarray,
) -> tuple[float, float, int, int, int, int]:
    """
    Compute:
      r_o     = |O_all| / |O_sample|
      r_omega = |Omega_all| / |Omega_sample|

    Returns:
      (r_o, r_omega, O_all, O_sample, omega_all, omega_sample)
    """
    O_all = int(Aow_node.shape[0])
    O_sample = int(len(sample_local))
    if O_sample <= 0:
        raise ValueError("sample_local is empty")

    # total keywords in node (appear at least once)
    if Aow_node.nnz == 0:
        omega_all = 0
    else:
        omega_all = int(np.unique(Aow_node.indices).size)

    Aow_sample = Aow_node[sample_local]
    if Aow_sample.nnz == 0:
        omega_sample = 0
    else:
        omega_sample = int(np.unique(Aow_sample.indices).size)

    r_o = float(O_all) / float(O_sample)

    # avoid div-by-zero
    r_omega = float(omega_all) / float(omega_sample) if omega_sample > 0 else float("inf")

    return r_o, r_omega

















def _csr_row_l2_normalize_to_dense(A: csr_matrix) -> np.ndarray:
    X = A.astype(np.float32).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def _safe_l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def compute_sample_split_total_cost(
    Aow_sample: csr_matrix,
    Aqw_node: csr_matrix,
    cluster_id_sample: np.ndarray,
) -> float:
    cluster_id_sample = np.asarray(cluster_id_sample, dtype=np.int64)

    if cluster_id_sample.size != Aow_sample.shape[0]:
        raise ValueError("cluster_id_sample size mismatch with Aow_sample rows")

    if np.all(cluster_id_sample == 0) or np.all(cluster_id_sample == 1):
        return float("inf")

    q_left_local, q_right_local, _, _ = seed_queries_by_sample_clusters(
        Aow_sample_csr=Aow_sample,
        Aqw_node=Aqw_node,
        cluster_id_sample=cluster_id_sample,
        query_idx_global=None, 
        drop_if_neither=True,
    )

    obj_left_local = np.where(cluster_id_sample == 0)[0].astype(np.int64)
    obj_right_local = np.where(cluster_id_sample == 1)[0].astype(np.int64)

    cost_left = node_cost(Aow_sample, Aqw_node, obj_left_local, q_left_local.astype(np.int64))
    cost_right = node_cost(Aow_sample, Aqw_node, obj_right_local, q_right_local.astype(np.int64))

    return float(cost_left + cost_right)


def binary_spherical_kmeans_cosine_history(
    Aow_sample: csr_matrix,
    Aqw_node: csr_matrix,
    max_iter: int = 50,
    tol: float = 1e-6,
    random_state: int = 42,
):
    X = _csr_row_l2_normalize_to_dense(Aow_sample)   # [n_s, d]
    n_s = X.shape[0]

    if n_s < 2:
        raise ValueError("Need at least 2 sampled objects for k-means")

    rng = np.random.default_rng(random_state)

    i0 = int(rng.integers(0, n_s))
    sims0 = X @ X[i0]
    i1 = int(np.argmin(sims0))

    c0 = X[i0].copy()
    c1 = X[i1].copy()
    c0 = _safe_l2_normalize(c0)
    c1 = _safe_l2_normalize(c1)

    history = []
    prev_labels = None

    for it in range(max_iter):
        s0 = X @ c0
        s1 = X @ c1
        labels = (s1 > s0).astype(np.int64)   # 0 / 1

        if np.all(labels == 0):
            j = int(np.argmin(s0))
            labels[j] = 1
        elif np.all(labels == 1):
            j = int(np.argmin(s1))
            labels[j] = 0

        total_cost = compute_sample_split_total_cost(
            Aow_sample=Aow_sample,
            Aqw_node=Aqw_node,
            cluster_id_sample=labels,
        )
        history.append({
            "iter": it + 1,
            "cost": float(total_cost),
        })

        new_c0 = X[labels == 0].mean(axis=0)
        new_c1 = X[labels == 1].mean(axis=0)
        new_c0 = _safe_l2_normalize(new_c0)
        new_c1 = _safe_l2_normalize(new_c1)

        shift = float(np.linalg.norm(new_c0 - c0) + np.linalg.norm(new_c1 - c1))

        # if prev_labels is not None and np.array_equal(labels, prev_labels):
        #     c0, c1 = new_c0, new_c1
        #     break
        # if shift < tol:
        #     c0, c1 = new_c0, new_c1
        #     break

        prev_labels = labels.copy()
        c0, c1 = new_c0, new_c1

    s0 = X @ c0
    s1 = X @ c1
    final_labels = (s1 > s0).astype(np.int64)

    if np.all(final_labels == 0):
        final_labels[int(np.argmin(s0))] = 1
    elif np.all(final_labels == 1):
        final_labels[int(np.argmin(s1))] = 0

    return history, final_labels







def bisplit_once(
    A_ow_all: csr_matrix,
    A_qw_all: csr_matrix,
    obj_idx: np.ndarray,
    query_idx: np.ndarray,
    N_sample: int = 1000,
    K_graph: int = 20,
    epochs: int = 200,
    lr: float = 1e-2,
    balance_lambda: float = 1e-2,
    device: Optional[str] = None,

    # ===== NEW for speed =====
    cache: Optional[SplitModelCache] = None,
    is_root: bool = False,                
    freeze_gcn: bool = True,               
    mlp_epochs: int = 30,                  
    patience: int = 300,
    min_delta: float = 0.001,
    bugDe: bool = True,
    vis_root: bool = True,
    vis_dir: str = "vis_root"
) -> SplitResult:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    n_node = obj_idx.size
    if n_node < 2 or query_idx.size == 0:
        return SplitResult(
            obj_left=obj_idx,
            obj_right=np.array([], dtype=np.int64),
            q_left=query_idx,
            q_right=np.array([], dtype=np.int64),
            cost_parent=node_cost(A_ow_all, A_qw_all, obj_idx, query_idx),
            cost_children=float("inf"),
        )

    cost_parent = node_cost(A_ow_all, A_qw_all, obj_idx, query_idx)
    min_delta = cost_parent * min_delta
    # print(f"cost_parent: {cost_parent}")
    # print(f"min_delta: {min_delta}")

    t_sample_before = time.perf_counter()

    # ----- node submatrices -----
    Aow_node = A_ow_all[obj_idx]
    Aqw_node = A_qw_all[query_idx]
    m_node = Aqw_node.shape[0]

    # ----- sample -----
    n_s = min(N_sample, n_node)
    perm = np.random.permutation(n_node)
    sample_local = perm[:n_s]
    Aow_sample = Aow_node[sample_local]

    # ---- stats: compare sample vs total ----
    r_o, r_omega = compute_sampling_ratios(
        Aow_node=Aow_node,
        sample_local=sample_local,
    )

    print(f"r_o: {r_o}, r_omega: {r_omega}")


    t_sample_end = time.perf_counter()
    t_cooccu_before = time.perf_counter()
    # ----- build co-occurrence graph -----
    g = build_cooccurrence_graph(A_ow=Aow_sample, A_qw=Aqw_node, K=K_graph)
    X_csr = g["X"]
    edge_index = g["edge_index"]
    edge_weight = g["edge_weight"]
    H_csr = g["H"]
    n_s, d = X_csr.shape
    t_cooccu_end = time.perf_counter()

    # torch tensors
    X_sparse = csr_to_torch_sparse(X_csr).to(device)
    A_norm = build_normalized_adj(edge_index, edge_weight, num_nodes=n_s, add_self_loops=True).to(device)
    Aqw_sparse = csr_to_torch_sparse(Aqw_node).to(device)

    # r on sampled bitmap
    r_np = np.asarray(H_csr.sum(axis=0)).reshape(-1).astype(np.float32)
    r = torch.from_numpy(r_np).to(device)

    t_train_before = time.perf_counter()

    # ----- build models -----
    gcn = WeightedGCNEncoder(vocab_dim=d, hidden_dim=64, out_dim=64, num_layers=2, dropout=0.0).to(device)
    mlp = BinaryClusterMLP(in_dim=64, hidden_dim=128, num_layers=2, dropout=0.0).to(device)

    # warm start from cache
    if cache is not None:
        init_gcn, init_mlp = cache.get_init(use_root=is_root)
        if init_gcn is not None:
            gcn.load_state_dict(init_gcn, strict=True)
        if init_mlp is not None:
            mlp.load_state_dict(init_mlp, strict=True)

    # ============================
    # TRAIN
    # ============================
    if is_root:
        # root: train both (as you already do)
        optimizer = torch.optim.Adam(list(gcn.parameters()) + list(mlp.parameters()), lr=lr)
        pre = float("inf")
        bad = 0

        for ep in range(epochs):
            gcn.train(); mlp.train()
            optimizer.zero_grad()

            emb = gcn(X_sparse, A_norm)
            # probs = torch.softmax(mlp(emb)/get_tau(ep), dim=1)
            # pi = probs[:, 0]
            pi = torch.sigmoid(mlp(emb) / get_tau(ep))

            loss, dbg = clustering_loss(X_sparse, Aqw_sparse, pi, r, r_o = r_o, r_omega = r_omega)
            loss.backward()
            optimizer.step()

            wL = float(dbg.get("wL", 0.0))
            wR = float(dbg.get("wR", 0.0))

            if bugDe and ep % 20 == 0:
                print(
                    f"epoch={ep:03d} loss={loss.item():.6f} "
                    f"oL={float(dbg['oL']):.2f} oR={float(dbg['oR']):.2f} "
                    f"omegaL={float(dbg['omegaL']):.2f} omegaR={float(dbg['omegaR']):.2f} "
                    f"wL={wL:.2f} wR={wR:.2f}"
                )

            cur = float(loss.detach().item())
            # if min_delta > abs(pre - cur):
            #     bad += 1
            # else:
            #     bad = 0
            # pre = cur
            # if bad >= patience:
            #     break

        # cache root weights
        if cache is not None:
            cache.set_root(gcn, mlp)

    else:
        # non-root: freeze GCN and only train MLP (fast)
        if freeze_gcn:
            for p in gcn.parameters():
                p.requires_grad_(False)
            gcn.eval()
            with torch.no_grad():
                emb_fixed = gcn(X_sparse, A_norm)   # (n_s, 64) computed once
        else:
            emb_fixed = None

        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
        best = float("inf")
        bad = 0

        for ep in range(mlp_epochs):
            mlp.train()
            optimizer.zero_grad()

            if emb_fixed is None:
                # still allow gcn training if you disable freeze_gcn
                gcn.train()
                emb = gcn(X_sparse, A_norm)
            else:
                emb = emb_fixed

            # probs = torch.softmax(mlp(emb), dim=1)
            # pi = probs[:, 0]
            pi = torch.sigmoid(mlp(emb) / get_tau(ep))

            loss, dbg = clustering_loss(X_sparse, Aqw_sparse, pi, r, r_o = r_o, r_omega = r_omega)
            loss.backward()
            optimizer.step()

            wL = float(dbg.get("wL", 0.0))
            wR = float(dbg.get("wR", 0.0))
            
            if bugDe and ep % 10 == 0:
                print(
                    f"epoch={ep:03d} loss={loss.item():.6f} "
                    f"oL={float(dbg['oL']):.2f} oR={float(dbg['oR']):.2f} "
                    f"omegaL={float(dbg['omegaL']):.2f} omegaR={float(dbg['omegaR']):.2f} "
                    f"wL={wL:.2f} wR={wR:.2f}"
                )

            cur = float(loss.detach().item())
            if cur < best - min_delta:
                best = cur
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

        # update last weights for next node warm-start
        if cache is not None:
            cache.update_last(gcn, mlp)

    # ============================
    # infer sampled hard clusters
    # ============================
    gcn.eval(); mlp.eval()
    with torch.no_grad():
        # if frozen and emb_fixed exists, reuse it
        if (not is_root) and freeze_gcn:
            emb = emb_fixed
        else:
            emb = gcn(X_sparse, A_norm)

        logit = mlp(emb)                        # [N]
        pi = torch.sigmoid(logit)               # [N] (或 sigmoid(logit / tau))
        cluster_id_sample = (pi >= 0.5).long().cpu().numpy().astype(np.int64)

        emb_sample_np = emb.detach().cpu().numpy().copy()

    if np.all(cluster_id_sample == 0) or np.all(cluster_id_sample == 1):
        return SplitResult(
            obj_left=obj_idx,
            obj_right=np.array([], dtype=np.int64),
            q_left=query_idx,
            q_right=np.array([], dtype=np.int64),
            cost_parent=cost_parent,
            cost_children=float("inf"),
        )

    # ----- seed queries for S2 using same fast routing -----
    seed_q_left, seed_q_right, _, _ = seed_queries_by_sample_clusters(
        Aow_sample_csr=Aow_sample,
        Aqw_node=Aqw_node,
        cluster_id_sample=cluster_id_sample,
        query_idx_global=None,      # LOCAL ids for streaming_assign
        drop_if_neither=True,
    )
    if seed_q_left.size == 0:
        seed_q_left = np.array([0], dtype=np.int64)
    if seed_q_right.size == 0 and m_node > 1:
        seed_q_right = np.array([1], dtype=np.int64)

    t_train_end = time.perf_counter()
    t_S2_before = time.perf_counter()
    # ----- S2 streaming assign -----
    seed_obj_left = sample_local[cluster_id_sample == 0]
    seed_obj_right = sample_local[cluster_id_sample == 1]

    # print(f"query left: {seed_q_left.size}, object left: {seed_obj_left.size}, query right: {seed_q_right.size}, object right: {seed_obj_right.size}")

    res = streaming_assign(
        A_ow_csr=Aow_node,
        A_qw_csr=Aqw_node,
        seed_obj_left=seed_obj_left,
        seed_obj_right=seed_obj_right,
        seed_q_left=seed_q_left,
        seed_q_right=seed_q_right,
        epsilon=param.EPSILON,
        alpha=param.ALFA,
        beta=param.BETA,
        gamma=param.GAMA,
        stream_balance_lambda=0.0,
    )


    if is_root and vis_root:
        out_file = os.path.join(vis_dir, f"root_split_{n_node}.png")
        visualize_root_split(
            Aow_node=Aow_node,
            sample_local=sample_local,
            emb_sample=emb_sample_np,
            cluster_id_sample=cluster_id_sample,
            res=res,
            out_file=out_file,
            title_prefix=f"Root split (n={n_node}, sample={len(sample_local)})"
        )
        print(f"[VIS] saved root visualization to: {out_file}")

    obj_left_global = obj_idx[res.obj_left]
    obj_right_global = obj_idx[res.obj_right]

    # ----- recursion queries by kw feasibility (fast) -----
    left_kw = np.unique(A_ow_all[obj_left_global].indices).astype(np.int64)
    right_kw = np.unique(A_ow_all[obj_right_global].indices).astype(np.int64)
    kw_ids, kw_lr_bits = build_kw_lr_bitset(left_kw, right_kw)

    q_left_global, q_right_global = split_queries_by_kw_lr_bitset(
        Aqw_node=Aqw_node,
        kw_ids=kw_ids,
        kw_lr_bits=kw_lr_bits,
        query_idx_global=query_idx,
        drop_if_neither=True,
    )

    # print(f"query left: {q_left_global.size}, object left: {obj_left_global.size}, query right: {q_right_global.size}, object right: {obj_right_global.size}")

    t_S2_end = time.perf_counter()

    cost_left = node_cost(A_ow_all, A_qw_all, obj_left_global, q_left_global)
    cost_right = node_cost(A_ow_all, A_qw_all, obj_right_global, q_right_global)
    cost_children = cost_left + cost_right

    print(f"cost before split: {cost_parent}, cost after split: {cost_children}")

    # print(f"time of sampling: {(t_sample_end - t_sample_before) / 1e9: .6f} s"
    #       f"time of cooccurence construct: {(t_cooccu_end - t_cooccu_before) / 1e9: .6f} s"
    #       f"time of training: {(t_train_end - t_train_before) / 1e9: .6f} s, epoch: {ep}"
    #       f"time of S2: {(t_S2_end - t_S2_before) / 1e9: .6f} s"
    #       f""
    #     )

    return SplitResult(
        obj_left=obj_left_global,
        obj_right=obj_right_global,
        q_left=q_left_global,
        q_right=q_right_global,
        cost_parent=cost_parent,
        cost_children=cost_children,
    )




import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _pad_to_2d(Z):
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[1] == 0:
        Z = np.zeros((Z.shape[0], 2), dtype=np.float32)
    elif Z.shape[1] == 1:
        Z = np.concatenate([Z, np.zeros((Z.shape[0], 1), dtype=Z.dtype)], axis=1)
    return Z[:, :2]


def reduce_sparse_to_2d(X, random_state=42):
    if min(X.shape[0], X.shape[1]) <= 2:
        return _pad_to_2d(_to_dense(X))
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    return svd.fit_transform(X)


def reduce_dense_to_2d(X, random_state=42):
    X = np.asarray(X)
    if X.shape[1] <= 2:
        return _pad_to_2d(X)
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)


def visualize_root_split(
    Aow_node,
    sample_local,
    emb_sample,
    cluster_id_sample,
    res,
    out_file="root_split_vis.png",
    title_prefix="Root split"
):
    """
    Visualize the feature-space change during root-node bi-splitting.

    Parameters
    ----------
    Aow_node : sparse matrix or ndarray, shape [n_obj, d]
        Original feature matrix of all objects under the root node.
    sample_local : ndarray, shape [n_sample]
        Local indices of sampled objects within the root node.
    emb_sample : ndarray, shape [n_sample, d_emb]
        Learned embeddings of sampled objects.
    cluster_id_sample : ndarray, shape [n_sample]
        Predicted binary cluster labels of sampled objects (0/1).
    res : object
        Result returned by streaming_assign, containing:
            - res.obj_left
            - res.obj_right
    out_file : str
        Output figure path.
    title_prefix : str
        Figure title.
    """

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    # ---------- colors ----------
    blue_c = "#4C78A8"
    purple_c = "#8E63CE"
    gray_c = "#C9CDD3"

    # ---------- font sizes ----------
    title_fs = 15
    label_fs = 18
    tick_fs = 1
    legend_fs = 18
    suptitle_fs = 16

    # ---------- 2D projections ----------
    # raw/original feature space for all objects
    Z_all_raw = reduce_sparse_to_2d(Aow_node)
    Z_sample_raw = Z_all_raw[sample_local]

    # learned feature space for sampled objects
    Z_sample_emb = reduce_dense_to_2d(emb_sample)

    # ---------- final labels after streaming-assign ----------
    n_obj = Aow_node.shape[0]
    final_label = np.full(n_obj, -1, dtype=np.int64)
    final_label[np.asarray(res.obj_left, dtype=np.int64)] = 0
    final_label[np.asarray(res.obj_right, dtype=np.int64)] = 1

    is_sample = np.zeros(n_obj, dtype=bool)
    is_sample[np.asarray(sample_local, dtype=np.int64)] = True
    is_stream = ~is_sample

    # sample labels in final assignment
    idx_sample_left = is_sample & (final_label == 0)
    idx_sample_right = is_sample & (final_label == 1)

    # streaming-assigned labels
    idx_stream_left = is_stream & (final_label == 0)
    idx_stream_right = is_stream & (final_label == 1)

    # ---------- plotting ----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # ===== (a) Original feature space of sampled objects =====
    ax = axes[0]
    ax.scatter(
        Z_sample_raw[:, 0],
        Z_sample_raw[:, 1],
        s=18,
        alpha=0.80,
        c=blue_c,
        label="objects"
    )
    ax.set_title("Original feature space of sampled objects", fontsize=title_fs)
    ax.set_xlabel("dim-1", fontsize=label_fs)
    ax.set_ylabel("dim-2", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, frameon=True)
    ax.grid(alpha=0.2)

    # ===== (b) Learned feature space of sampled objects =====
    ax = axes[1]
    idx_left = (np.asarray(cluster_id_sample) == 0)
    idx_right = (np.asarray(cluster_id_sample) == 1)

    ax.scatter(
        Z_sample_emb[idx_left, 0],
        Z_sample_emb[idx_left, 1],
        s=18,
        alpha=0.82,
        c=blue_c,
        label="left cluster"
    )
    ax.scatter(
        Z_sample_emb[idx_right, 0],
        Z_sample_emb[idx_right, 1],
        s=18,
        alpha=0.82,
        c=purple_c,
        label="right cluster"
    )
    ax.set_title("Learned feature space of sampled objects", fontsize=title_fs)
    ax.set_xlabel("dim-1", fontsize=label_fs)
    ax.set_ylabel("dim-2", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, frameon=True)
    ax.grid(alpha=0.2)

    # ===== (c) Final clustering after streaming-assign =====
    ax = axes[2]

    # background: all objects
    ax.scatter(
        Z_all_raw[:, 0],
        Z_all_raw[:, 1],
        s=8,
        alpha=0.08,
        c=gray_c
    )

    # remaining objects assigned by streaming-assign
    ax.scatter(
        Z_all_raw[idx_stream_left, 0],
        Z_all_raw[idx_stream_left, 1],
        s=18,
        alpha=0.82,
        c=blue_c,
        label="left cluster"
    )
    ax.scatter(
        Z_all_raw[idx_stream_right, 0],
        Z_all_raw[idx_stream_right, 1],
        s=18,
        alpha=0.82,
        c=purple_c,
        label="right cluster"
    )

    # sample seeds highlighted by hollow black circles
    ax.scatter(
        Z_all_raw[is_sample, 0],
        Z_all_raw[is_sample, 1],
        s=46,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        label="sample seed"
    )

    ax.set_title("Final clustering after streaming-assign", fontsize=title_fs)
    ax.set_xlabel("dim-1", fontsize=label_fs)
    ax.set_ylabel("dim-2", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, frameon=True)
    ax.grid(alpha=0.2)

    fig.suptitle(title_prefix, fontsize=suptitle_fs)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)