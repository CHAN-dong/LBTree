"""Default BBC for LBTree_icde (3), Sections IV-B and IV-C.

The graph is represented by sparse object-keyword and query-keyword incidence
matrices. It never constructs object-object edges. Historical repetitions are
represented by repeated query rows, equivalently unit frequency per row.
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch import nn
import parameter as param
from model.cluster_mlp import BinaryClusterMLP
from model.cooccurrence_graph import _build_hit_matrix_H


def sparse_tensor(matrix, device="cpu", dtype=torch.float32):
    matrix = matrix.tocoo()
    indices = torch.as_tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.long, device=device)
    values = torch.as_tensor(matrix.data.copy(), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device, check_invariants=True).coalesce()


def row_normalize(matrix):
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    return matrix.multiply((1.0 / np.maximum(lengths, 1))[:, None]).tocsr()


def keyword_features(objects, queries, keyword_ids, hash_width=16):
    """Document frequency, workload frequency, and deterministic hashed code."""
    df = np.asarray(objects.sum(axis=0)).ravel()
    qf = np.asarray(queries.sum(axis=0)).ravel()
    # Fixed integer arithmetic makes the code independent of Python hash seeds.
    ids = np.asarray(keyword_ids, dtype=np.uint64)
    hashed = ids * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    code = np.column_stack([((hashed >> np.uint64(i)) & np.uint64(1)).astype(np.float32)
                            for i in range(hash_width)]) * 2 - 1
    frequency = np.column_stack((np.log1p(df) / np.log1p(max(1, objects.shape[0])),
                                 np.log1p(qf) / np.log1p(max(1, queries.shape[0]))))
    return np.column_stack((frequency, code)).astype(np.float32)


class RelationEncoder(nn.Module):
    """Eq. (6): shared keyword encoder and query-to-keyword propagation."""
    def __init__(self, width=64):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(18, width), nn.ReLU(), nn.Linear(width, width))
        self.self_projection = nn.Linear(width, width, bias=False)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, features, query_mean, keyword_from_queries):
        base = self.features(features)
        queries = torch.sparse.mm(query_mean, base)
        message = torch.sparse.mm(keyword_from_queries, queries)
        return torch.relu(self.self_projection(base) + self.query_projection(message) + self.bias)


def prepare_paper_loss(objects, queries, hits=None, device="cpu", dtype=torch.float32):
    """Fixed sparse incidences for Eqs. (10)-(19); H contains exact conjunction hits."""
    if hits is None:
        hits = _build_hit_matrix_H(objects, queries)
    ao = sparse_tensor(objects, device, dtype)
    aq = sparse_tensor(queries, device, dtype)
    return {
        "objects": ao, "queries": aq, "hits_t": sparse_tensor(hits.T, device, dtype),
        "lengths": torch.as_tensor(np.diff(queries.indptr), device=device, dtype=dtype),
        "parent_keywords": int(np.unique(objects.indices).size),
    }


def product_gate(matrix, probabilities):
    """Product of keyword probabilities per query, including exact zero gates."""
    rows, cols = matrix.indices()
    factors = probabilities[cols]
    result = probabilities.new_ones(matrix.shape[0])
    return result.scatter_reduce(0, rows, factors, reduce="prod", include_self=True)


def keyword_presence(objects, probabilities):
    """Eq. (11), exactly 1 - product_i(1 - A[i,t] * pi[i])."""
    rows, cols = objects.indices()
    absence = probabilities.new_ones(objects.shape[1])
    absence = absence.scatter_reduce(0, cols, 1 - probabilities[rows], reduce="prod", include_self=True)
    return 1 - absence


def paper_loss(pi, context):
    """The three-bitmap objective in Eq. (19), without STE or regularizers."""
    lengths = context["lengths"]
    nonempty = (lengths > 0).to(pi.dtype)
    stats = []
    for assignment in (pi, 1 - pi):
        presence = keyword_presence(context["objects"], assignment)
        gates = product_gate(context["queries"], presence) * nonempty
        results = torch.sparse.mm(context["hits_t"], assignment[:, None]).squeeze(1)
        count = assignment.sum()
        keywords = presence.sum()
        storage = keywords * count + 32 * keywords + 32 * count
        query = (gates * (param.predict_query(count, lengths, results))).sum()
        stats.append(dict(presence=presence, gates=gates, results=results,
                          storage=storage, query=query))
    d = context["parent_keywords"]
    parent_storage = pi.new_tensor(2 * d + 32 * d + 64)
    parent_results = stats[0]["gates"] + stats[1]["gates"]
    parent_query = (nonempty * (param.predict_query(2, lengths, parent_results))).sum()
    loss = param.weighted_cost(parent_storage, parent_query)
    for child in stats:
        loss = loss + param.weighted_cost(child["storage"], child["query"])
    return loss, {"children": stats, "parent_storage": parent_storage, "parent_query": parent_query}


def split(objects, queries, obj_idx, query_idx, *, sample_count, epochs, lr,
          device, width, cache, context=None, parent_keyword_count=None, skip_training=False):
    """Learn sampled partitions, then run cost-driven streaming assignment."""
    from model.model_split import SplitResult, node_cost, _complete_external_partition, _record_stage
    start = time.perf_counter()
    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)
    parent_cost = node_cost(objects, queries, obj_idx, query_idx, context=context,
                            keyword_count=parent_keyword_count)
    if len(obj_idx) < 2 or not len(query_idx) or skip_training:
        return SplitResult(obj_idx, np.array([], dtype=np.int64), query_idx,
                           np.array([], dtype=np.int64), parent_cost, float("inf"))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ao, aq = objects[obj_idx], queries[query_idx]
    sampled = np.random.permutation(len(obj_idx))[:min(sample_count, len(obj_idx))]
    active = np.union1d(ao.indices, aq.indices)
    sample = ao[sampled][:, active]
    local_queries = aq[:, active]
    features = torch.as_tensor(keyword_features(ao[:, active], local_queries, active), device=device)
    query_mean = sparse_tensor(row_normalize(local_queries), device)
    # Each historical query occurrence has frequency one. Repeated rows count
    # repeatedly, implementing the frequency-weighted mean in Eq. (6).
    keyword_from_queries = sparse_tensor(row_normalize(local_queries.T.tocsr()), device)
    df = np.asarray(ao[:, active].sum(axis=0)).ravel()
    alpha = 1 + np.log((len(obj_idx) + 1) / (df + 1))
    weighted = row_normalize(sample.multiply(alpha).tocsr())
    object_aggregation = sparse_tensor(weighted, device)
    node_hits = context.slice_hits(obj_idx, query_idx) if context is not None else None
    hits = node_hits[sampled] if node_hits is not None else None
    loss_context = prepare_paper_loss(sample, local_queries, hits, device)
    encoder = RelationEncoder(width).to(device)
    head = BinaryClusterMLP(width, 2 * width, num_layers=2).to(device)
    previous = getattr(cache, "paper_state", None) if cache is not None else None
    if previous is not None:
        encoder.load_state_dict(previous[0]); head.load_state_dict(previous[1])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    _record_stage(cache, "paper_preparation_seconds", time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        keyword_vectors = encoder(features, query_mean, keyword_from_queries)
        embeddings = torch.sparse.mm(object_aggregation, keyword_vectors)
        pi = torch.sigmoid(head(embeddings))
        loss, _ = paper_loss(pi, loss_context)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite BBC objective")
        loss.backward(); optimizer.step()
    with torch.no_grad():
        vectors = encoder(features, query_mean, keyword_from_queries)
        pi = torch.sigmoid(head(torch.sparse.mm(object_aggregation, vectors)))
        # Eq. (9): probability >= .5 is LEFT, represented by cluster label zero.
        labels = (pi < .5).cpu().numpy().astype(np.int64)
    if cache is not None:
        cache.paper_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
        cache.training_calls += 1
        cache.sample_object_epochs += len(sampled) * epochs
        if len(obj_idx) == objects.shape[0]: cache.root_epochs += epochs
        else: cache.nonroot_epochs += epochs
    _record_stage(cache, "paper_training_seconds", time.perf_counter() - start)
    start = time.perf_counter()
    result = _complete_external_partition(ao, aq, obj_idx, query_idx, sampled, labels,
                                          parent_cost, context, node_hits, objects, queries)
    _record_stage(cache, "paper_streaming_seconds", time.perf_counter() - start)
    return result
