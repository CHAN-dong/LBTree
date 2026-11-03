# model/cooccurrence_graph.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import parameter as param

def _row_nonzero_cols(A, i: int) -> np.ndarray:
    """Return nonzero column indices of row i for csr matrix or dense list."""
    if getattr(A, 'format', None) == 'csr':
        return A.indices[A.indptr[i]:A.indptr[i + 1]]
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
    if a.size>b.size: a,b=b,a
    if not a.size or not b.size: return np.empty(0,dtype=np.result_type(a,b))
    positions=np.searchsorted(b,a)
    valid=positions<b.size
    values=a[valid];positions=positions[valid]
    values=values[b[positions]==values]
    if values.size>1: values=values[np.r_[True,values[1:]!=values[:-1]]]
    return values

def _build_hit_matrix_H(A_ow, A_qw, max_hits=None) -> Any:
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
        raise RuntimeError("建议安装 scipy，用稀疏矩阵构建共现图更快：pip install scipy")

    # Queries are columns of H; avoid boxing every hit in three Python lists.
    hit_parts = []
    hit_count = 0
    hit_indptr = [0]

    for q in range(m):
        if n>1_000_000 and q%2000==0:
            print(f'[hits] query={q}/{m} hits={hit_count}',flush=True)
        terms = _row_nonzero_cols(A_qw, q)
        if terms.size == 0:
            hit_indptr.append(hit_count)
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

        # Optional memory budget for the streaming-assignment cache.
        if max_hits is not None and hit_count + cand.size > max_hits:
            return None
        if cand.size:
            hit_parts.append(cand)
        hit_count += cand.size
        hit_indptr.append(hit_count)

    # Build sparse CSR matrix H
    from scipy.sparse import csc_matrix  # type: ignore
    indices = np.concatenate(hit_parts) if hit_parts else np.empty(0, dtype=np.int32)
    H = csc_matrix((np.ones(hit_count, dtype=np.int8), indices, np.asarray(hit_indptr)),
                   shape=(n, m)).tocsr()
    H.sum_duplicates()
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
    object_query_hits=None,
    workers: int = 1,
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
    if workers<1:raise ValueError('graph workers must be positive')

    if A_ow.shape[0] > 4096:
        return _build_blocked_graph(A_ow, A_qw, K, object_query_hits,workers=workers)

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
    H = object_query_hits if object_query_hits is not None else _build_hit_matrix_H(A_ow_csr, A_qw)
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
    # Eq. (6) combines dimensionless similarities, not normalized costs.
    S = param.w_s * J_kw + param.w_q * J_q

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


def _build_blocked_graph(objects, queries, K, hits=None,workers=1):
    """Exact row-block Jaccard/top-K: bound memory without changing sampling.

    Both Jaccards are symmetric on these binary matrices, so symmetrization
    before top-K is redundant. Preserve CSR column ordering and tie behavior.
    """
    from scipy.sparse import csr_matrix
    import time
    objects=objects.tocsr();n=objects.shape[0]
    if hits is None: hits=_build_hit_matrix_H(objects,queries)
    kw_count=np.asarray(objects.sum(axis=1)).ravel()
    q_count=np.asarray(hits.sum(axis=1)).ravel() if hits is not None else np.zeros(n)
    # scipy's CSR sparse product coerces its right operand to CSR. Cache that
    # conversion once instead of repeating it for every row block.
    ot=objects.T.tocsr();ht=hits.T.tocsr() if hits is not None else None
    edges=[];weights=[];started=time.perf_counter()
    def process(start):
        edges=[];weights=[]
        stop=min(start+128,n)
        kw=(objects[start:stop]@ot).tocsr();kw.sort_indices()
        query=(hits[start:stop]@ht).tocsr() if ht is not None and ht.nnz else None
        score=None
        if n<=8192:
            # For medium graphs, vectorized normalization amortizes Python
            # dispatch better than the cache-sized row path used for big ones.
            def normalize(inter,counts):
                row_counts=np.repeat(counts[start:stop],np.diff(inter.indptr))
                union=row_counts+counts[inter.indices]-inter.data
                values=np.divide(inter.data,union,out=np.zeros_like(inter.data,dtype=np.float32),where=union!=0)
                return csr_matrix((values,inter.indices,inter.indptr),shape=inter.shape)
            jk=normalize(kw,kw_count)
            if query is not None:
                query.sort_indices();score=param.w_s*jk+param.w_q*normalize(query,q_count)
            else:score=param.w_s*jk
            score=score.tocsr();score.sort_indices()
        # Normalize and combine one row at a time, keeping temporary vectors
        # cache-sized rather than materializing two large Jaccard matrices and
        # a third combined CSR. Column order stays identical to the old score.
        for local in range(stop-start):
            row=start+local
            if score is not None:
                lo,hi=score.indptr[local:local+2];cols=score.indices[lo:hi];values=score.data[lo:hi]
                valid=(cols!=row)&(values!=0);cols=cols[valid];values=values[valid]
                if len(cols)>K:
                    chosen=np.argpartition(values,-K)[-K:];cols=cols[chosen];values=values[chosen]
                    order=np.argsort(-values);cols=cols[order];values=values[order]
                if len(cols):
                    edges.append(np.vstack((np.full(len(cols),row,dtype=np.int64),cols)))
                    weights.append(values.astype(np.float32,copy=False))
                continue
            lo,hi=kw.indptr[local:local+2];cols=kw.indices[lo:hi];inter=kw.data[lo:hi]
            union=kw_count[row]+kw_count[cols]-inter
            values=np.divide(inter,union,out=np.zeros_like(inter,dtype=np.float32),where=union!=0)*param.w_s
            if query is not None:
                qlo,qhi=query.indptr[local:local+2]
                if qhi>qlo:
                    qcols=query.indices[qlo:qhi];qinter=query.data[qlo:qhi]
                    qunion=q_count[row]+q_count[qcols]-qinter
                    qvalues=np.divide(qinter,qunion,out=np.zeros_like(qinter,dtype=np.float32),where=qunion!=0)*param.w_q
                    positions=np.searchsorted(cols,qcols)
                    if np.all(positions<cols.size) and np.array_equal(cols[positions],qcols):
                        values[positions]+=qvalues
                    else:
                        # Also support externally supplied hit matrices whose
                        # nonzero similarities are not a subset of keyword ones.
                        merged=np.union1d(cols,qcols);combined=np.zeros(merged.size,dtype=np.float32)
                        combined[np.searchsorted(merged,cols)]=values
                        combined[np.searchsorted(merged,qcols)]+=qvalues
                        cols,values=merged,combined
            valid=(cols!=row)&(values!=0);cols=cols[valid];values=values[valid]
            if len(cols)>K:
                chosen=np.argpartition(values,-K)[-K:];cols=cols[chosen];values=values[chosen]
                order=np.argsort(-values);cols=cols[order];values=values[order]
            if len(cols):
                edges.append(np.vstack((np.full(len(cols),start+local,dtype=np.int64),cols)))
                weights.append(values.astype(np.float32,copy=False))
        return (np.concatenate(edges,axis=1) if edges else np.empty((2,0),dtype=np.int64),
                np.concatenate(weights) if weights else np.empty(0,dtype=np.float32))
    starts=range(0,n,128);pool=None
    if workers>1:
        from concurrent.futures import ThreadPoolExecutor
        pool=ThreadPoolExecutor(max_workers=min(workers,max(1,len(starts))))
        results=pool.map(process,starts)
    else:results=map(process,starts)
    try:
        # map yields in input order even when workers finish out of order.
        for start,(block_edges,block_weights) in zip(starts,results):
            if block_weights.size:edges.append(block_edges);weights.append(block_weights)
            stop=min(start+128,n)
            if start%4096==0 or stop==n:
                print(f'[graph] rows={stop}/{n} elapsed={time.perf_counter()-started:.1f}s workers={workers}',flush=True)
    finally:
        if pool is not None:pool.shutdown(wait=True)
    return {'X':objects,'H':hits,
            'edge_index':np.concatenate(edges,axis=1) if edges else np.empty((2,0),dtype=np.int64),
            'edge_weight':np.concatenate(weights) if weights else np.empty(0,dtype=np.float32)}
