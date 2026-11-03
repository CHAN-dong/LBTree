# model/model_split.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix  # type: ignore
import parameter as param
from model.cooccurrence_graph import build_cooccurrence_graph
from model.gcn_encoder import (WeightedGCNEncoder, build_normalized_adj, csr_to_torch_sparse,
                               compact_keyword_projection, restore_keyword_projection)
from model.cluster_mlp import ClusterMLP, BinaryClusterMLP
from model.loss import clustering_loss, prepare_loss_context
from model.streaming_assign import streaming_assign
from model.query_split import (
    build_kw_lr_bitset,
    split_queries_by_kw_lr_bitset,
    seed_queries_by_sample_clusters,
)
import time
import os
import copy


from model.split_cache import SplitModelCache, BestValidSplit

def _record_stage(cache, name, seconds):
    if cache is not None:
        cache.stage_seconds[name]=cache.stage_seconds.get(name,0.)+seconds

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
    context=None,
    keyword_count=None,
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
    if O == 0:
        return 0.0

    Aow = None
    if keyword_count is not None:
        omega = int(keyword_count)
    else:
        Aow = A_ow_all[obj_idx]
        omega = int(np.asarray(Aow.sum(axis=0)).reshape(-1).astype(np.int64).nonzero()[0].size)

    storage = (omega * O) + (32 * omega) + (32 * O)

    if context is not None and context.hits is not None:
        sum_r = int(context.slice_hits(obj_idx, query_idx).sum())
        sum_q = int(np.maximum(context.query_lengths[query_idx] - 1, 0).sum())
        qcost = (query_idx.size * param.EPSILON + param.ALFA * sum_q * O
                 + param.BETA * query_idx.size * O + param.GAMA * sum_r)
        return float(param.weighted_cost(storage, qcost))

    # For |R_q|: exact postings intersection on this node
    if Aow is None:
        Aow = A_ow_all[obj_idx]
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
        sum_q += max(q_len - 1, 0)

    qcost = (
        query_idx.size * param.EPSILON
        + param.ALFA * sum_q * O
        + param.BETA * query_idx.size * O
        + param.GAMA * sum_r
    )

    # print(f"outside: ------sum_q: {sum_q}, sum_r: {sum_r}, O: {O}")

    return float(param.weighted_cost(storage, qcost))


# ---------- Output container ----------
@dataclass
class SplitResult:
    obj_left: np.ndarray
    obj_right: np.ndarray
    q_left: np.ndarray
    q_right: np.ndarray
    cost_parent: float
    cost_children: float
    left_keywords: Optional[np.ndarray] = None
    right_keywords: Optional[np.ndarray] = None
    keyword_ids: Optional[np.ndarray] = None
    keyword_bits: Optional[np.ndarray] = None


def _complete_external_partition(Aow_node, Aqw_node, obj_idx, query_idx,
                                 sample_local, labels, cost_parent, context, node_hits,
                                 A_ow_all, A_qw_all):
    """Use exactly the learned path's S2/routing/cost rules for ablations."""
    if np.all(labels == 0) or np.all(labels == 1):
        return SplitResult(obj_idx,np.array([],dtype=np.int64),query_idx,
                           np.array([],dtype=np.int64),cost_parent,float('inf'))
    ql,qr,_,_=seed_queries_by_sample_clusters(Aow_node[sample_local],Aqw_node,labels,
                                             drop_if_neither=True)
    if ql.size==0:ql=np.array([0],dtype=np.int64)
    if qr.size==0 and Aqw_node.shape[0]>1:qr=np.array([1],dtype=np.int64)
    res=streaming_assign(Aow_node,Aqw_node,sample_local[labels==0],sample_local[labels==1],ql,qr,
        epsilon=param.EPSILON,alpha=param.ALFA,beta=param.BETA,gamma=param.GAMA,
        stream_balance_lambda=0.,object_query_hits=node_hits)
    ol,orr=obj_idx[res.obj_left],obj_idx[res.obj_right]
    left_kw=np.unique(A_ow_all[ol].indices).astype(np.int64)
    right_kw=np.unique(A_ow_all[orr].indices).astype(np.int64)
    kw,bits=build_kw_lr_bitset(left_kw,right_kw)
    ql,qr=split_queries_by_kw_lr_bitset(Aqw_node,kw,bits,query_idx_global=query_idx,drop_if_neither=True)
    reuse=context is not None and context.positive_objects
    left=node_cost(A_ow_all,A_qw_all,ol,ql,context=context,keyword_count=left_kw.size if reuse else None)
    right=node_cost(A_ow_all,A_qw_all,orr,qr,context=context,keyword_count=right_kw.size if reuse else None)
    storage=2*kw.size+32*kw.size+64
    qlens=np.diff(Aqw_node.indptr)
    query=(param.EPSILON*query_idx.size+param.ALFA*np.maximum(qlens-1,0).sum()*2
           +param.BETA*query_idx.size*2+param.GAMA*(ql.size+qr.size))
    return SplitResult(ol,orr,ql,qr,cost_parent,left+right+param.weighted_cost(storage,query),
                       left_kw,right_kw,kw,bits)


def _uniq_kw_count(Aow_csr: csr_matrix, obj_local_idx: np.ndarray) -> int:
    """
    统计某个对象子集里出现过的不同 keyword 数量（硬计数）
    Aow_csr: (n, d) node-local objects matrix
    obj_local_idx: local indices into Aow_csr
    """
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
    """
    把稀疏 0/1 或 count 关键词矩阵转成 dense，并按行做 L2 normalize。
    这样用点积就等价于 cosine similarity。
    """
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
    """
    给定 sample 对象的二分结果，先路由 query，再计算左右两个簇的总 cost。
    这里算的是 sample 上的 total cost，用来画曲线对比。
    """
    cluster_id_sample = np.asarray(cluster_id_sample, dtype=np.int64)

    if cluster_id_sample.size != Aow_sample.shape[0]:
        raise ValueError("cluster_id_sample size mismatch with Aow_sample rows")

    # 退化分簇，直接记为 inf
    if np.all(cluster_id_sample == 0) or np.all(cluster_id_sample == 1):
        return float("inf")

    q_left_local, q_right_local, _, _ = seed_queries_by_sample_clusters(
        Aow_sample_csr=Aow_sample,
        Aqw_node=Aqw_node,
        cluster_id_sample=cluster_id_sample,
        query_idx_global=None,   # 返回 local query ids
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
    """
    二类 spherical k-means（cosine 版）：
    - 输入：sample 对象-关键词矩阵 Aow_sample
    - 相似度：cosine
    - 每一轮记录当前分簇对应的 sample total cost

    返回：
    - history: [{"iter": 1, "cost": ...}, ...]
    - final_labels: shape [n_sample]
    """
    X = _csr_row_l2_normalize_to_dense(Aow_sample)   # [n_s, d]
    n_s = X.shape[0]

    if n_s < 2:
        raise ValueError("Need at least 2 sampled objects for k-means")

    rng = np.random.default_rng(random_state)

    # 初始化：随机挑一个点，再挑与它 cosine 最不相似的点
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
        # assignment step: 按 cosine 相似度分到最近中心
        s0 = X @ c0
        s1 = X @ c1
        labels = (s1 > s0).astype(np.int64)   # 0 / 1

        # 防止空簇
        if np.all(labels == 0):
            j = int(np.argmin(s0))
            labels[j] = 1
        elif np.all(labels == 1):
            j = int(np.argmin(s1))
            labels[j] = 0

        # 记录当前迭代对应的 total cost
        total_cost = compute_sample_split_total_cost(
            Aow_sample=Aow_sample,
            Aqw_node=Aqw_node,
            cluster_id_sample=labels,
        )
        history.append({
            "iter": it + 1,
            "cost": float(total_cost),
        })

        # update step: 重新计算两个中心
        new_c0 = X[labels == 0].mean(axis=0)
        new_c1 = X[labels == 1].mean(axis=0)
        new_c0 = _safe_l2_normalize(new_c0)
        new_c1 = _safe_l2_normalize(new_c1)

        shift = float(np.linalg.norm(new_c0 - c0) + np.linalg.norm(new_c1 - c1))

        # 收敛判定
        # if prev_labels is not None and np.array_equal(labels, prev_labels):
        #     c0, c1 = new_c0, new_c1
        #     break
        # if shift < tol:
        #     c0, c1 = new_c0, new_c1
        #     break

        prev_labels = labels.copy()
        c0, c1 = new_c0, new_c1

    # 最终 labels 再算一次
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
    epochs: int = 300,
    lr: float = 0.0005,
    balance_lambda: float = 1e-2,
    device: Optional[str] = None,

    # ===== NEW for speed =====
    cache: Optional[SplitModelCache] = None,
    is_root: bool = False,                 # 根节点 True，其他 False
    freeze_gcn: bool = True,               # 非根节点默认 True：冻结 GCN 只训 MLP
    mlp_epochs: int = 30,                  # 非根节点s只训 MLP 的 epochc
    patience: int = 300,
    min_delta: float = 0.001,
    bugDe: bool = True,
    vis_root: bool = False,
    vis_dir: str = "vis_root",
    context=None,
    parent_keyword_count=None,
    partition_method='learned',
    graph_workers=1,
    graph_policy='exact',
    root_early_stopping=False,
    root_min_relative_improvement=.001,
    root_restore_best_valid=False,
    root_min_epochs=0,
    model_width=64,
    skip_training=False,
    node_keyword_ids=None,
) -> SplitResult:

    guarded_threshold=partition_method=='threshold_guarded'
    threshold_mode=partition_method in ('threshold_search','threshold_guarded')
    direct_mode=partition_method in ('direct_search','threshold_search','threshold_guarded')
    progressive_mode=partition_method=='progressive'
    if direct_mode or progressive_mode:partition_method='learned'
    cached_mode=partition_method=='cached_pointwise'
    if cached_mode:partition_method='learned'
    active_loss=partition_method in ('active','active_refine')
    if partition_method=='adaptive_active':
        active_loss=len(query_idx)>=4096
        partition_method='active' if active_loss else 'learned'
    if active_loss:partition_method='learned_refine' if partition_method=='active_refine' else 'learned'
    freeze_root_after=64 if partition_method=='frozen_refine' else None
    if partition_method=='frozen_refine':partition_method='learned_refine'
    if partition_method in ('budget_refine','budget_group_refine'):
        partition_method='learned_group_refine' if partition_method=='budget_group_refine' else 'learned_refine'
        if is_root:
            root_early_stopping=True;root_restore_best_valid=True;root_min_epochs=min(64,epochs)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    n_node = obj_idx.size
    if n_node < 2 or query_idx.size == 0 or skip_training:
        if skip_training and cache is not None:cache.prepruned_nodes+=1
        return SplitResult(
            obj_left=obj_idx,
            obj_right=np.array([], dtype=np.int64),
            q_left=query_idx,
            q_right=np.array([], dtype=np.int64),
            cost_parent=node_cost(A_ow_all, A_qw_all, obj_idx, query_idx, context=context, keyword_count=parent_keyword_count),
            cost_children=float("inf"),
        )

    cost_parent = node_cost(A_ow_all, A_qw_all, obj_idx, query_idx, context=context, keyword_count=parent_keyword_count)
    min_delta = cost_parent * min_delta
    # print(f"cost_parent: {cost_parent}")
    # print(f"min_delta: {min_delta}")

    t_sample_before = time.perf_counter()

    # ----- node submatrices -----
    Aow_node = A_ow_all[obj_idx]
    Aqw_node = A_qw_all[query_idx]
    node_hits = context.slice_hits(obj_idx, query_idx) if context is not None else None
    m_node = Aqw_node.shape[0]

    if partition_method in ('multilevel','multilevel_refine'):
        from model.multilevel_bbc import coarse_partition,refine_partition,result_from_labels
        labels,coarse_info=coarse_partition(Aow_node)
        if partition_method=='multilevel_refine':
            from model.cooccurrence_graph import _build_hit_matrix_H
            hits=node_hits if node_hits is not None else _build_hit_matrix_H(Aow_node,Aqw_node)
            labels,_=refine_partition(Aow_node,Aqw_node,labels,int(hits.sum()),groups=coarse_info.get('groups'))
        result=result_from_labels(Aow_node,Aqw_node,obj_idx,query_idx,labels,cost_parent,context,A_ow_all,A_qw_all)
        _record_stage(cache,'multilevel_partition_seconds',time.perf_counter()-t_sample_before)
        return result

    # ----- sample -----
    n_s = min(N_sample, n_node)
    perm = np.random.permutation(n_node)
    sample_local = perm[:n_s]
    Aow_sample = Aow_node[sample_local]

    if partition_method in ('fi_stream','fi_linear','fi_fm'):
        from model.factorized_bbc import split as factorized_split
        _record_stage(cache,'sampling_and_node_data_seconds',time.perf_counter()-t_sample_before)
        return factorized_split(Aow_node,Aqw_node,obj_idx,query_idx,sample_local,cost_parent,context,node_hits,
            A_ow_all,A_qw_all,cache,partition_method,epochs if is_root else mlp_epochs,lr,patience,min_delta,
            is_root,model_width,device,node_keywords=node_keyword_ids)
    if partition_method == 'greedy_cost':
        from model.greedy_cost import cluster
        from model.cooccurrence_graph import _build_hit_matrix_H
        sample_hits=node_hits[sample_local] if node_hits is not None else _build_hit_matrix_H(Aow_sample,Aqw_node)
        labels,info=cluster(Aow_sample,Aqw_node,int(sample_hits.sum()),
            seed=getattr(cache,'greedy_seed',42),max_rounds=getattr(cache,'greedy_max_rounds',20))
        _record_stage(cache,'greedy_clustering_seconds',time.perf_counter()-t_sample_before)
        if cache is not None:
            if not hasattr(cache,'greedy_audit'):cache.greedy_audit=[]
            cache.greedy_audit.append(dict(info,node_objects=n_node,sampled_objects=n_s,
                sample_object_ids=obj_idx[sample_local].tolist(),queries=int(query_idx.size),
                s_ref=param.S_REF,q_ref=param.Q_REF,lambda_value=param.LAMBDA))
        t_complete=time.perf_counter()
        result=_complete_external_partition(Aow_node,Aqw_node,obj_idx,query_idx,sample_local,
            labels,cost_parent,context,node_hits,A_ow_all,A_qw_all)
        _record_stage(cache,'greedy_streaming_routing_validation_seconds',time.perf_counter()-t_complete)
        return result
    if partition_method == 'kmeans':
        from sklearn.cluster import KMeans
        keywords=np.unique(Aow_sample.indices)
        labels=(KMeans(n_clusters=2,random_state=42,n_init=10,max_iter=300)
                .fit_predict(Aow_sample[:,keywords]) if keywords.size else np.zeros(n_s,dtype=np.int64))
        result = _complete_external_partition(Aow_node,Aqw_node,obj_idx,query_idx,sample_local,
            labels,cost_parent,context,node_hits,A_ow_all,A_qw_all)
        _record_stage(cache,'kmeans_partition_seconds',time.perf_counter()-t_sample_before)
        return result
    if partition_method not in ('learned','learned_refine','learned_group_refine'):raise ValueError(f'Unknown partition method {partition_method}')

    # ---- stats: compare sample vs total ----
    r_o, r_omega = compute_sampling_ratios(
        Aow_node=Aow_node,
        sample_local=sample_local,
    )

    print(f"r_o: {r_o}, r_omega: {r_omega}")


    t_sample_end = time.perf_counter()
    _record_stage(cache,'sampling_and_node_data_seconds',t_sample_end-t_sample_before)
    t_cooccu_before = time.perf_counter()
    # ----- build co-occurrence graph -----
    use_embedding_cache=(cached_mode and not is_root and freeze_gcn and cache is not None and cache.cached_embeddings is not None)
    if use_embedding_cache:
        from model.cooccurrence_graph import _build_hit_matrix_H
        sample_hits=node_hits[sample_local] if node_hits is not None else _build_hit_matrix_H(Aow_sample,Aqw_node)
        g=dict(X=Aow_sample,H=sample_hits,edge_index=None,edge_weight=None)
    elif graph_policy in ('fused_exact','fused_bitset'):
        from model.fused_graph import build as build_fused_graph
        g=build_fused_graph(Aow_sample,Aqw_node,K_graph,
            node_hits[sample_local] if node_hits is not None else None,
            engine='bitset' if graph_policy=='fused_bitset' else 'sort')
    elif graph_policy!='exact':
        from model.light_graph import build as build_light_graph
        g,graph_stats=build_light_graph(Aow_sample,Aqw_node,K_graph,
            node_hits[sample_local] if node_hits is not None else None,policy=graph_policy)
    else:
        g = build_cooccurrence_graph(A_ow=Aow_sample, A_qw=Aqw_node, K=K_graph,
                                object_query_hits=node_hits[sample_local] if node_hits is not None else None,
                                workers=graph_workers)
    X_csr = g["X"]
    edge_index = g["edge_index"]
    edge_weight = g["edge_weight"]
    H_csr = g["H"]
    n_s, d = X_csr.shape
    t_cooccu_end = time.perf_counter()
    _record_stage(cache,'graph_seconds',t_cooccu_end-t_cooccu_before)

    # torch tensors
    X_sparse = csr_to_torch_sparse(X_csr).to(device)
    A_norm = None if use_embedding_cache else build_normalized_adj(edge_index, edge_weight, num_nodes=n_s, add_self_loops=True).to(device)
    training_queries=Aqw_node
    feasible=None
    if active_loss:
        present=np.zeros(d,dtype=np.float32);present[np.unique(X_csr.indices)]=1
        lengths=np.diff(Aqw_node.indptr)
        feasible=(np.asarray(Aqw_node@present).ravel()==lengths)&(lengths>0)
        training_queries=Aqw_node[feasible]
        print(f'[active-bbc] feasible_queries={int(feasible.sum())}/{m_node}',flush=True)
    Aqw_sparse = csr_to_torch_sparse(training_queries).to(device)
    # Loss needs only sample/query columns; include query-only terms so unknown
    # keyword gates keep their original zero-column semantics.
    cost_keywords = np.union1d(X_csr.indices, training_queries.indices)
    if cost_keywords.size < d:
        loss_objects = csr_to_torch_sparse(X_csr[:, cost_keywords]).to(device)
        loss_queries = csr_to_torch_sparse(training_queries[:, cost_keywords]).to(device)
    else:
        loss_objects, loss_queries = X_sparse, Aqw_sparse
    loss_context = prepare_loss_context(loss_objects, loss_queries)
    if active_loss:
        # Screening changes neither parent constants nor global normalizers.
        # Only queries impossible under every partition of this sample vanish.
        loss_context['parent_query_count']=torch.tensor(int((lengths>0).sum()),device=device)
        loss_context['parent_query_length_sum']=torch.tensor(int(lengths.sum()),device=device)

    # r on sampled bitmap
    r_np = np.asarray(H_csr.sum(axis=0)).reshape(-1).astype(np.float32)
    if feasible is not None:r_np=r_np[feasible]
    r = torch.from_numpy(r_np).to(device)

    t_train_before = time.perf_counter()
    _record_stage(cache,'tensor_preparation_seconds',t_train_before-t_cooccu_end)

    # ----- build models -----
    reuse_frozen = (not is_root and freeze_gcn and cache is not None
                    and cache.frozen_encoder is not None)
    gcn = cache.frozen_encoder if reuse_frozen else WeightedGCNEncoder(
        vocab_dim=d, hidden_dim=model_width, out_dim=model_width, num_layers=2, dropout=0.0).to(device)
    mlp = BinaryClusterMLP(in_dim=model_width, hidden_dim=2*model_width, num_layers=2, dropout=0.0).to(device)

    # warm start from cache
    if cache is not None:
        init_gcn, init_mlp = cache.get_init(use_root=is_root)
        if init_gcn is not None and not reuse_frozen:
            gcn.load_state_dict(init_gcn, strict=True)
        if init_mlp is not None:
            mlp.load_state_dict(init_mlp, strict=True)

    if direct_mode and not is_root and freeze_gcn and cache is not None:
        from model.direct_bbc import search
        t_direct=time.perf_counter();gcn.eval();mlp.eval()
        with torch.no_grad():
            initial_emb=gcn(X_sparse,A_norm)
            initial_labels=(mlp(initial_emb).reshape(-1)>=0).cpu().numpy().astype(np.int64)
        if threshold_mode:
            from model.threshold_bbc import search as threshold_search
            with torch.no_grad():logits=mlp(initial_emb).reshape(-1).cpu().numpy()
            labels=threshold_search(Aow_sample,Aqw_node,initial_emb.cpu().numpy(),logits)
        else:labels,moves=search(Aow_sample,Aqw_node,initial_emb.cpu().numpy(),initial_labels,int(H_csr.sum()))
        candidate=_complete_external_partition(Aow_node,Aqw_node,obj_idx,query_idx,sample_local,labels,
            cost_parent,context,node_hits,A_ow_all,A_qw_all)
        _record_stage(cache,'direct_search_seconds',time.perf_counter()-t_direct)
        if candidate.cost_children<(.8 if guarded_threshold else .95)*cost_parent:
            cache.direct_accepted+=1
            return candidate
        cache.direct_fallback+=1
        t_train_before=time.perf_counter()

    if cache is not None:cache.training_calls+=1
    # ============================
    # TRAIN
    # ============================
    if is_root:
        training_features, projection_state = compact_keyword_projection(gcn, X_sparse)
        # root: train both (as you already do)
        optimizer = torch.optim.Adam(list(gcn.parameters()) + list(mlp.parameters()), lr=lr)
        pre = float("inf")
        bad = 0
        root_significant=float('inf');root_best_loss=float('inf');root_checkpoint=None
        root_fixed=None
        stage_features=training_features;stage_adj=A_norm
        stage_objects=loss_objects;stage_context=loss_context;stage_r=r
        stage_rows=np.arange(n_s);root_keywords=(projection_state[1].cpu().numpy() if projection_state is not None else np.arange(d))
        ep=-1

        for ep in range(epochs):
            gcn.train(); mlp.train()
            optimizer.zero_grad()

            if progressive_mode and ep in (0,50,100):
                if ep==0:stage_rows=np.arange(max(2,int(np.ceil(n_s/4))))[:n_s]
                elif ep==50:
                    with torch.no_grad():
                        confidence=torch.abs(mlp(gcn(training_features,A_norm)).reshape(-1)).cpu().numpy()
                    remaining=np.setdiff1d(np.arange(n_s),stage_rows)
                    extra=remaining[np.argsort(confidence[remaining])[:max(0,int(np.ceil(n_s/2))-len(stage_rows))]]
                    stage_rows=np.sort(np.r_[stage_rows,extra])
                else:stage_rows=np.arange(n_s)
                inverse=np.full(n_s,-1,dtype=np.int64);inverse[stage_rows]=np.arange(len(stage_rows))
                keep=(inverse[edge_index[0]]>=0)&(inverse[edge_index[1]]>=0)
                stage_adj=build_normalized_adj(inverse[edge_index[:,keep]],edge_weight[keep],len(stage_rows),add_self_loops=True).to(device)
                stage_features=csr_to_torch_sparse(X_csr[stage_rows][:,root_keywords]).to(device)
                stage_objects=csr_to_torch_sparse(X_csr[stage_rows][:,cost_keywords]).to(device)
                stage_context=prepare_loss_context(stage_objects,loss_queries)
                stage_r=torch.tensor(np.asarray(H_csr[stage_rows].sum(axis=0)).ravel().astype(np.float32),device=device)
            if cache is not None:cache.sample_object_epochs+=len(stage_rows)

            if freeze_root_after is not None and ep==freeze_root_after:
                for parameter in gcn.parameters():parameter.requires_grad_(False)
                gcn.eval()
                with torch.no_grad():root_fixed=gcn(training_features,A_norm).detach()
            emb = root_fixed if root_fixed is not None else gcn(stage_features, stage_adj)
            # probs = torch.softmax(mlp(emb)/get_tau(ep), dim=1)
            # pi = probs[:, 0]
            pi = torch.sigmoid(mlp(emb) / get_tau(ep))

            loss, dbg = clustering_loss(stage_objects, loss_queries, pi, stage_r, r_o = r_o, r_omega = r_omega, context=stage_context)
            if root_early_stopping and root_restore_best_valid:
                score=float(loss.detach());labels=pi.detach()>=.5
                if score<root_best_loss and bool(labels.any()) and not bool(labels.all()):
                    root_best_loss=score
                    root_checkpoint=(copy.deepcopy(gcn.state_dict()),copy.deepcopy(mlp.state_dict()))
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
            if root_early_stopping:
                if root_significant==float('inf') or cur<root_significant-max(abs(root_significant)*root_min_relative_improvement,1e-12):
                    root_significant=cur;bad=0
                else:bad+=1
                if (bad>=patience and ep+1>=root_min_epochs
                        and (not root_restore_best_valid or root_checkpoint is not None)):break
            # if min_delta > abs(pre - cur):
            #     bad += 1
            # else:
            #     bad = 0
            # pre = cur
            # if bad >= patience:
            #     break

        if cache is not None:cache.root_epochs+=ep+1
        if root_checkpoint is not None:
            gcn.load_state_dict(root_checkpoint[0]);mlp.load_state_dict(root_checkpoint[1])
        if root_early_stopping:
            policy=f'best_valid_loss={root_best_loss:.9g}' if root_restore_best_valid else 'checkpoint=last'
            print(f'[root-early-stop] epochs={ep+1}/{epochs} min_epochs={root_min_epochs} {policy}',flush=True)
        restore_keyword_projection(gcn, projection_state)
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
                emb_fixed = (cache.cached_embeddings[obj_idx[sample_local]].to(device) if use_embedding_cache
                             else gcn(X_sparse, A_norm))
        else:
            emb_fixed = None

        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
        best = float("inf")
        bad = 0
        best_valid = BestValidSplit()

        ep=-1
        for ep in range(mlp_epochs):
            if cache is not None:cache.sample_object_epochs+=n_s
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

            loss, dbg = clustering_loss(loss_objects, loss_queries, pi, r, r_o = r_o, r_omega = r_omega, context=loss_context)
            best_valid.consider(mlp, loss, pi)
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

        if cache is not None:cache.nonroot_epochs+=ep+1
        restored = best_valid.restore(mlp)
        if restored and bugDe:
            print(f'[checkpoint] restored best nonempty split loss={best_valid.loss:.9g}', flush=True)
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

    t_train_end = time.perf_counter()
    _record_stage(cache,'training_and_inference_seconds',t_train_end-t_train_before)
    if np.all(cluster_id_sample == 0) or np.all(cluster_id_sample == 1):
        return SplitResult(
            obj_left=obj_idx,
            obj_right=np.array([], dtype=np.int64),
            q_left=query_idx,
            q_right=np.array([], dtype=np.int64),
            cost_parent=cost_parent,
            cost_children=float("inf"),
        )

    if cached_mode and is_root and cache is not None:
        # Approximation: root encoder applied with self edges to all objects.
        # Reuse these fixed pointwise vectors instead of new subgraph messages.
        t_cache=time.perf_counter();parts=[]
        with torch.no_grad():
            for start in range(0,A_ow_all.shape[0],4096):
                features=csr_to_torch_sparse(A_ow_all[start:start+4096]).to(device)
                count=features.shape[0];ids=torch.arange(count,device=device)
                identity=torch.sparse_coo_tensor(torch.stack((ids,ids)),torch.ones(count,device=device),(count,count)).coalesce()
                parts.append(gcn(features,identity).detach().cpu())
        cache.cached_embeddings=torch.cat(parts)
        _record_stage(cache,'embedding_cache_seconds',time.perf_counter()-t_cache)

    t_seed_start=time.perf_counter()
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

    _record_stage(cache,'seed_routing_seconds',time.perf_counter()-t_seed_start)
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
        object_query_hits=node_hits,
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
    _record_stage(cache,'streaming_and_query_routing_seconds',t_S2_end-t_S2_before)

    reuse_counts = context is not None and context.positive_objects
    cost_left = node_cost(A_ow_all, A_qw_all, obj_left_global, q_left_global, context=context,
                          keyword_count=left_kw.size if reuse_counts else None)
    cost_right = node_cost(A_ow_all, A_qw_all, obj_right_global, q_right_global, context=context,
                           keyword_count=right_kw.size if reuse_counts else None)
    # Eq. (5): splitting must also pay for the new two-branch parent bitmap.
    storage_parent = 2 * kw_ids.size + 32 * kw_ids.size + 64
    q_lens = np.diff(Aqw_node.indptr)
    query_parent = (param.EPSILON * query_idx.size
                    + param.ALFA * np.maximum(q_lens - 1, 0).sum() * 2
                    + param.BETA * query_idx.size * 2
                    + param.GAMA * (q_left_global.size + q_right_global.size))
    cost_children = cost_left + cost_right + param.weighted_cost(storage_parent, query_parent)
    _record_stage(cache,'split_cost_validation_seconds',time.perf_counter()-t_S2_end)

    print(f"cost before split: {cost_parent}, cost after split: {cost_children}")

    # print(f"time of sampling: {(t_sample_end - t_sample_before) / 1e9: .6f} s"
    #       f"time of cooccurence construct: {(t_cooccu_end - t_cooccu_before) / 1e9: .6f} s"
    #       f"time of training: {(t_train_end - t_train_before) / 1e9: .6f} s, epoch: {ep}"
    #       f"time of S2: {(t_S2_end - t_S2_before) / 1e9: .6f} s"
    #       f""
    #     )

    if partition_method in ('learned_refine','learned_group_refine'):
        from model.multilevel_bbc import refine_partition,result_from_labels,refinement_groups
        t_refine=time.perf_counter();labels=np.ones(n_node,dtype=np.int64);labels[res.obj_left]=0
        if node_hits is None:
            from model.cooccurrence_graph import _build_hit_matrix_H
            node_hits=_build_hit_matrix_H(Aow_node,Aqw_node)
        groups=refinement_groups(Aow_node,labels) if partition_method=='learned_group_refine' else None
        labels,_=refine_partition(Aow_node,Aqw_node,labels,int(node_hits.sum()),groups=groups)
        refined=result_from_labels(Aow_node,Aqw_node,obj_idx,query_idx,labels,cost_parent,context,A_ow_all,A_qw_all)
        assert refined.cost_children<=cost_children+1e-10
        _record_stage(cache,'exact_refinement_seconds',time.perf_counter()-t_refine)
        return refined

    return SplitResult(
        obj_left=obj_left_global,
        obj_right=obj_right_global,
        q_left=q_left_global,
        q_right=q_right_global,
        cost_parent=cost_parent,
        cost_children=cost_children,
        left_keywords=left_kw,
        right_keywords=right_kw,
        keyword_ids=kw_ids,
        keyword_bits=kw_lr_bits,
    )





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
    from sklearn.decomposition import TruncatedSVD
    if min(X.shape[0], X.shape[1]) <= 2:
        return _pad_to_2d(_to_dense(X))
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    return svd.fit_transform(X)


def reduce_dense_to_2d(X, random_state=42):
    from sklearn.decomposition import PCA
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

    import matplotlib.pyplot as plt
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
