# model/tree_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import time
import numpy as np
import parameter as param
from scipy.sparse import csr_matrix  # type: ignore

from model.model_split import bisplit_once, SplitResult
from model.query_split import LEFT_BIT, RIGHT_BIT
from model.split_cache import SplitModelCache
from utils.text_io import _norm_packed_bits, _write_tree_bin, _cleanup_legacy_split_files


from utils.packed_bitmap import make_leaf_keyword_object_bitmaps as _make_leaf_keyword_object_bitmaps


def _cleanup_legacy_split_files(out_dir: str):
    """
    删除历史遗留的 node_*.npy / tree.json，避免旧文件继续占空间。
    """
    if not os.path.exists(out_dir):
        return
    for fn in os.listdir(out_dir):
        if fn.endswith(".npy") or fn == "tree.json":
            path = os.path.join(out_dir, fn)
            if os.path.isfile(path):
                os.remove(path)



def _pack_rows_child_bits(child_bits_bool: np.ndarray) -> np.ndarray:
    """
    child_bits_bool: (K, C) bool
    return: (K, ceil(C/8)) uint8 packed (bitorder=little)
    """
    return np.packbits(child_bits_bool.astype(np.uint8), axis=1, bitorder="little")


def _save_internal_kw_child_bits(out_dir: str, node_id: int,
                                children_ids: np.ndarray,
                                kw_ids: np.ndarray,
                                kw_child_bits_packed: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"node_{node_id}_children_ids.npy"), children_ids.astype(np.int64))
    np.save(os.path.join(out_dir, f"node_{node_id}_kw_ids.npy"), kw_ids.astype(np.int64))
    np.save(os.path.join(out_dir, f"node_{node_id}_kw_child_bits.npy"), kw_child_bits_packed.astype(np.uint8))



def _keywords_in_objects(A_ow_all: csr_matrix, obj_idx_global: np.ndarray) -> np.ndarray:
    sub = A_ow_all[obj_idx_global]
    if sub.nnz == 0:
        return np.array([], dtype=np.int64)
    return np.unique(sub.indices).astype(np.int64)


def _save_internal_kw_lr_bits(
    out_dir: str,
    node_id: int,
    kw_ids: np.ndarray,
    kw_lr_bits: np.ndarray,
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"node_{node_id}_kw_ids.npy"), kw_ids)
    np.save(os.path.join(out_dir, f"node_{node_id}_kw_lr_bits.npy"), kw_lr_bits)


def _save_leaf_keyword_object_bitmaps(
    out_dir: str,
    node_id: int,
    A_ow_all: csr_matrix,
    obj_idx_global: np.ndarray,
):
    """
    叶节点保存：
      - obj_ids: 叶子全局对象 id
      - leaf_kw_ids: 叶子出现过的 keywords
      - leaf_kw_obj_bits: (K, ceil(|O|/8)) packed bitmap matrix
          第 k 行对应 leaf_kw_ids[k]
          bit=1 表示该 keyword 在该对象里存在
    """
    os.makedirs(out_dir, exist_ok=True)
    obj_idx_global = np.asarray(obj_idx_global, dtype=np.int64)
    np.save(os.path.join(out_dir, f"node_{node_id}_obj_ids.npy"), obj_idx_global)

    A_leaf = A_ow_all[obj_idx_global]  # (O, d)
    O = A_leaf.shape[0]

    if O == 0 or A_leaf.nnz == 0:
        np.save(os.path.join(out_dir, f"node_{node_id}_leaf_kw_ids.npy"), np.array([], dtype=np.int64))
        np.save(os.path.join(out_dir, f"node_{node_id}_leaf_kw_obj_bits.npy"), np.zeros((0, 0), dtype=np.uint8))
        return

    leaf_kw_ids = np.unique(A_leaf.indices).astype(np.int64)
    np.save(os.path.join(out_dir, f"node_{node_id}_leaf_kw_ids.npy"), leaf_kw_ids)

    A_csc = A_leaf.tocsc()
    indptr = A_csc.indptr
    indices = A_csc.indices

    bytes_per_row = (O + 7) // 8
    K = leaf_kw_ids.size
    kw_obj_bits = np.zeros((K, bytes_per_row), dtype=np.uint8)

    # 直接写 packed bytes
    for k, kw in enumerate(leaf_kw_ids):
        s = indptr[kw]
        e = indptr[kw + 1]
        rows = indices[s:e]  # local rows [0..O-1]
        for r in rows:
            byte = int(r) // 8
            bit = int(r) % 8
            kw_obj_bits[k, byte] |= (1 << bit)

    np.save(os.path.join(out_dir, f"node_{node_id}_leaf_kw_obj_bits.npy"), kw_obj_bits)


@dataclass
class TreeNodeMeta:
    node_id: int
    depth: int
    is_leaf: bool
    cost: float
    left: Optional[int]
    right: Optional[int]
    kind: str  # "internal" or "leaf"
    data: Dict[str, Any]



class TreeBuilder:

    def _reset_mem_tree_cache(self):
        self._mem_children_ids = {}               # node_id -> np.ndarray[int32]
        self._mem_leaf_obj_ids = {}               # node_id -> np.ndarray[int64]
        self._mem_leaf_kw_ids = {}                # node_id -> np.ndarray[int32]
        self._mem_leaf_kw_obj_bits = {}           # node_id -> np.ndarray[uint8]
        self._mem_internal_kw_ids = {}            # node_id -> np.ndarray[int32]
        self._mem_internal_kw_child_bits = {}     # node_id -> np.ndarray[uint8]


    def clear_mem_tree_cache(self):
        self._mem_children_ids.clear()
        self._mem_leaf_obj_ids.clear()
        self._mem_leaf_kw_ids.clear()
        self._mem_leaf_kw_obj_bits.clear()
        self._mem_internal_kw_ids.clear()
        self._mem_internal_kw_child_bits.clear()


    def _cache_leaf_payload(self, node_id: int, obj_ids, kw_ids, kw_obj_bits, *, take_ownership=False):
        obj_ids = np.ascontiguousarray(np.asarray(obj_ids, dtype=np.int64).reshape(-1))
        kw_ids = np.ascontiguousarray(np.asarray(kw_ids, dtype=np.int32).reshape(-1))

        O = int(obj_ids.size)
        row_bytes = (O + 7) // 8
        kw_obj_bits = _norm_packed_bits(
            kw_obj_bits,
            rows=int(kw_ids.size),
            row_bytes=row_bytes,
            name=f"leaf bits node={node_id}",
        )
        if not take_ownership:
            kw_obj_bits=kw_obj_bits.copy()

        self._mem_leaf_obj_ids[node_id] = obj_ids
        self._mem_leaf_kw_ids[node_id] = kw_ids
        self._mem_leaf_kw_obj_bits[node_id] = kw_obj_bits


    def _cache_internal_payload(self, node_id: int, children_ids, kw_ids, kw_child_bits):
        children_ids = np.ascontiguousarray(np.asarray(children_ids, dtype=np.int32).reshape(-1))
        kw_ids = np.ascontiguousarray(np.asarray(kw_ids, dtype=np.int32).reshape(-1))

        C = int(children_ids.size)
        row_bytes = (C + 7) // 8
        kw_child_bits = _norm_packed_bits(
            kw_child_bits,
            rows=int(kw_ids.size),
            row_bytes=row_bytes,
            name=f"internal bits node={node_id}",
        ).copy()

        self._mem_children_ids[node_id] = children_ids
        self._mem_internal_kw_ids[node_id] = kw_ids
        self._mem_internal_kw_child_bits[node_id] = kw_child_bits


    def get_children_ids_mem(self, node_id: int) -> np.ndarray:
        return self._mem_children_ids[node_id]


    def get_leaf_payload_mem(self, node_id: int):
        return (
            self._mem_leaf_obj_ids[node_id],
            self._mem_leaf_kw_ids[node_id],
            self._mem_leaf_kw_obj_bits[node_id],
        )


    def get_node_keywords_mem(self, node_id: int) -> np.ndarray:
        if self._meta[node_id].is_leaf:
            return self._mem_leaf_kw_ids[node_id]
        return self._mem_internal_kw_ids[node_id]


    def __init__(
        self,
        A_ow_all: csr_matrix,
        A_qw_all: csr_matrix,
        out_dir: str = "./tree_out",
        N_sample: Optional[int] = None,
        K_graph: Optional[int] = None,
        epochs: int = 300,
        lr: float = 0.0005,
        # balance_lambda: float = 1e-2,
        balance_lambda: float = 0,
        min_objects: int = 1,
        min_queries: int = 0,
        max_depth: int = 20,
        cost_improve_eps: float = 0.0,
        device: Optional[str] = None,
        visualize_root: bool = False,
        max_leaf_bytes: Optional[int] = None,
        training_patience: int = 8,
        partition_method: str = 'learned',
        sample_cap: Optional[int] = None,
        graph_workers: int = 1,
        graph_policy: str = 'exact',
        root_early_stopping: bool = False,
        root_min_relative_improvement: float = .001,
        root_restore_best_valid: bool = False,
        root_min_epochs: int = 0,
        model_width: int = 64,
    ):
        self.A_ow_all = A_ow_all
        self.A_qw_all = A_qw_all
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.N_sample = N_sample
        if sample_cap is not None and sample_cap<2:
            raise ValueError('sample_cap must be at least 2')
        self.sample_cap=sample_cap
        if graph_workers<1:raise ValueError('graph_workers must be positive')
        self.graph_workers=graph_workers
        if graph_policy not in ('exact','fused_exact','fused_bitset','bounded8','bounded16','bounded8_half','adaptive16','adaptive8_half','near','adaptive_near'):raise ValueError('Unknown graph_policy')
        self.graph_policy=graph_policy
        if root_min_relative_improvement<0:raise ValueError('root_min_relative_improvement must be nonnegative')
        self.root_early_stopping=root_early_stopping
        self.root_min_relative_improvement=root_min_relative_improvement
        self.root_restore_best_valid=root_restore_best_valid
        if root_min_epochs<0:raise ValueError('root_min_epochs must be nonnegative')
        self.root_min_epochs=root_min_epochs
        if model_width<1:raise ValueError('model_width must be positive')
        self.model_width=model_width
        self.K_graph = K_graph
        self.epochs = epochs
        self.lr = lr
        self.balance_lambda = balance_lambda

        self.min_objects = min_objects
        self.min_queries = min_queries
        self.max_depth = max_depth
        self.cost_improve_eps = cost_improve_eps
        self.device = device
        self.visualize_root = visualize_root
        self.max_leaf_bytes = max_leaf_bytes
        if training_patience < 1:
            raise ValueError('training_patience must be positive')
        self.training_patience = training_patience
        if partition_method not in ('learned','fi_stream','fi_linear','fi_fm','greedy_cost','kmeans','multilevel','multilevel_refine','learned_refine','budget_refine','budget_group_refine','learned_group_refine','frozen_refine','active','active_refine','adaptive_active','cached_pointwise','direct_search','progressive','threshold_search','threshold_guarded'):
            raise ValueError('Unknown partition_method')
        if partition_method.startswith('fi_') and (root_early_stopping or root_restore_best_valid):
            raise ValueError('FI experiment uses fixed root epochs; root early-stop/checkpoint options are not supported')
        self.partition_method = partition_method

        self.model_cache = SplitModelCache()

        self._next_id = 0
        self._meta: Dict[int, TreeNodeMeta] = {}
        self._root_id: Optional[int] = None

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _build(self, obj_idx: np.ndarray, query_idx: np.ndarray, depth: int, keywords=None) -> int:
        node_id = self._new_id()
        obj_idx = np.asarray(obj_idx, dtype=np.int64)
        query_idx = np.asarray(query_idx, dtype=np.int64)

        # ---- node statistics for DP (Break redundant nodes) ----
        WN = int(query_idx.size)

        if WN == 0:
            sum_q_len_minus1 = 0
        else:
            q_lens = self.build_context.query_lengths[query_idx]
            sum_q_len_minus1 = int(np.maximum(q_lens - 1, 0).sum())

        node_keyword_ids=keywords if keywords is not None else _keywords_in_objects(self.A_ow_all,obj_idx)
        omegaN = int(node_keyword_ids.size)
        leaf_fits_budget=(self.max_leaf_bytes is None or omegaN*((obj_idx.size+7)//8)<=self.max_leaf_bytes)
        skip_training=leaf_fits_budget and (obj_idx.size<=self.min_objects or query_idx.size<=self.min_queries)
        sample_count = self.N_sample or min(obj_idx.size, max(2, int(np.ceil(obj_idx.size / 100))))
        if self.sample_cap is not None:sample_count=min(sample_count,self.sample_cap)
        graph_neighbors = self.K_graph or max(1, int(np.ceil(np.log2(max(2, sample_count)))))


        # try split (S1+S2)
        split: SplitResult = bisplit_once(
            A_ow_all=self.A_ow_all,
            A_qw_all=self.A_qw_all,
            obj_idx=obj_idx,
            query_idx=query_idx,
            N_sample=sample_count,
            K_graph=graph_neighbors,
            epochs=self.epochs,
            lr=self.lr,
            balance_lambda=self.balance_lambda,
            device=self.device,
            cache=self.model_cache,
            is_root=(depth == 0),
            freeze_gcn=(depth > 0),
            mlp_epochs=self.epochs,
            patience=self.training_patience,
            context=self.build_context,
            parent_keyword_count=omegaN if self.build_context.positive_objects else None,
            partition_method=self.partition_method,
            graph_workers=self.graph_workers,
            graph_policy=self.graph_policy,
            root_early_stopping=self.root_early_stopping,
            root_min_relative_improvement=self.root_min_relative_improvement,
            root_restore_best_valid=self.root_restore_best_valid,
            root_min_epochs=self.root_min_epochs,
            model_width=self.model_width,
            skip_training=skip_training,
            node_keyword_ids=node_keyword_ids,
            vis_root=self.visualize_root,
        )

        valid_children = (split.obj_left.size > 0 and split.obj_right.size > 0)
        # improved = (split.cost_parent - split.cost_children) > self.cost_improve_eps
        improved = (split.cost_parent - split.cost_children) > self.cost_improve_eps
        # ---------- leaf ----------
        if (not valid_children) or (not improved):
            leaf_bytes = omegaN * ((obj_idx.size + 7) // 8)
            if self.max_leaf_bytes is not None and leaf_bytes > self.max_leaf_bytes:
                raise MemoryError(f'Leaf {node_id} requires {leaf_bytes} packed bitmap bytes '
                                  f'({obj_idx.size} objects, {omegaN} keywords), exceeding '
                                  f'configured {self.max_leaf_bytes}-byte allocation budget')

            # if query_idx.size > 100000:
            #     print(f"split false ---sum_q_len_minus1: {sum_q_len_minus1}")
            #     print(f"split false ---obj_idxsize: {obj_idx.size}")
            #     print(f"split false ---query_idxsize: {query_idx.size}")
            #     split: SplitResult = bisplit_once(
            #         A_ow_all=self.A_ow_all,
            #         A_qw_all=self.A_qw_all,
            #         obj_idx=obj_idx,
            #         query_idx=query_idx,
            #         N_sample=self.N_sample,
            #         K_graph=self.K_graph,
            #         epochs=self.epochs,
            #         lr=self.lr,
            #         balance_lambda=self.balance_lambda,
            #         device=self.device,
            #         cache=self.model_cache,
            #         is_root=(depth == 0),
            #         freeze_gcn=(depth > 0),
            #         mlp_epochs=30,
            #         patience=50,
            #         bugDe=True,
            #     )

            leaf_started=time.perf_counter()
            leaf_obj_ids, leaf_kw_ids, leaf_kw_obj_bits = _make_leaf_keyword_object_bitmaps(
                self.A_ow_all, obj_idx
            )
            # The bitmap builder returns a fresh, private array. Transferring
            # it avoids a second full-size allocation for multi-GB leaves.
            self._cache_leaf_payload(node_id, leaf_obj_ids, leaf_kw_ids, leaf_kw_obj_bits, take_ownership=True)
            self._leaf_bitmap_seconds+=time.perf_counter()-leaf_started

            self._meta[node_id] = TreeNodeMeta(
                node_id=node_id,
                depth=depth,
                is_leaf=True,
                cost=float(split.cost_parent),
                left=None,
                right=None,
                kind="leaf",
                data={
                    "obj_count": int(leaf_obj_ids.size),
                    "kw_count": int(leaf_kw_ids.size),
                    "omega": omegaN,
                    "WN": WN,
                    "sum_q_len_minus1": sum_q_len_minus1,
                },
            )

            print(f"depth: {depth}, is_leaf: True, node_id: {node_id}")
            return node_id

        # ---------- internal ----------
        # 先递归，得到 child ids
        left_id = self._build(split.obj_left, split.q_left, depth + 1, split.left_keywords)
        right_id = self._build(split.obj_right, split.q_right, depth + 1, split.right_keywords)
        children_ids = np.array([left_id, right_id], dtype=np.int32)

        # 计算 keyword union + kw->child bits
        kw_ids = split.keyword_ids
        kw_child_bits_packed = split.keyword_bits.reshape(-1, 1)

        self._cache_internal_payload(
            node_id=node_id,
            children_ids=children_ids,
            kw_ids=kw_ids,
            kw_child_bits=kw_child_bits_packed,
        )

        print(f"depth: {depth}, is_leaf: False, node_id: {node_id}")

        self._meta[node_id] = TreeNodeMeta(
            node_id=node_id,
            depth=depth,
            is_leaf=False,
            cost=float(param.weighted_cost(
                2 * omegaN + 32 * omegaN + 64,
                param.EPSILON * WN
                + param.ALFA * sum_q_len_minus1 * 2
                + param.BETA * WN * 2
                + param.GAMA * (split.q_left.size + split.q_right.size))),
            left=None,
            right=None,
            kind="internal",
            data={
                "child_count": 2,
                "kw_count": int(kw_ids.size),
                "omega": omegaN,
                "WN": WN,
                "sum_q_len_minus1": sum_q_len_minus1,
            },
        )
        return node_id

    def build_tree(self, root_obj_idx: np.ndarray, root_query_idx: np.ndarray) -> int:
        from utils.cost_model import get_root_costs
        import parameter
        from model.build_context import BuildContext
        t0 = time.perf_counter()
        self.build_context = BuildContext(self.A_ow_all, self.A_qw_all)
        refs = get_root_costs(self.A_ow_all[root_obj_idx], self.A_qw_all[root_query_idx],
                             object_query_hits=self.build_context.slice_hits(root_obj_idx, root_query_idx))
        parameter.set_root_costs(*refs)
        reference_seconds = time.perf_counter() - t0
        self._reset_mem_tree_cache()
        self._leaf_bitmap_seconds=0.
        self.model_cache.stage_seconds.clear()
        self.model_cache.root_epochs=self.model_cache.nonroot_epochs=0
        self.model_cache.prepruned_nodes=0
        self.model_cache.cached_embeddings=None
        self.model_cache.training_calls=self.model_cache.direct_accepted=self.model_cache.direct_fallback=self.model_cache.sample_object_epochs=0
        t0 = time.perf_counter()
        self._root_id = self._build(root_obj_idx, root_query_idx, depth=0)
        recursive_build_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.save_tree()   # 仍然写本地，但只写 tree.bin
        self.build_timings = {
            'reference_cost_seconds': reference_seconds,
            'recursive_build_seconds': recursive_build_seconds,
            'binary_export_seconds': time.perf_counter() - t0,
            'shared_hit_cache_bytes': self.build_context.bytes,
            'node_stages': dict(self.model_cache.stage_seconds,leaf_bitmap_seconds=self._leaf_bitmap_seconds),
            'unattributed_recursive_seconds': recursive_build_seconds-sum(self.model_cache.stage_seconds.values())-self._leaf_bitmap_seconds,
            'root_training_epochs': self.model_cache.root_epochs,
            'nonroot_training_epochs': self.model_cache.nonroot_epochs,
            'prepruned_nodes': self.model_cache.prepruned_nodes,
            'training_calls': self.model_cache.training_calls,
            'direct_accepted': self.model_cache.direct_accepted,
            'direct_fallback': self.model_cache.direct_fallback,
            'sample_object_epochs': self.model_cache.sample_object_epochs,
        }
        return self._root_id

    # def save_tree(self):
    #     assert self._root_id is not None
    #     nodes = [self._meta[i].__dict__ for i in sorted(self._meta.keys())]
    #     out = {"root": int(self._root_id), "nodes": nodes}
    #     with open(os.path.join(self.out_dir, "tree.json"), "w", encoding="utf-8") as f:
    #         json.dump(out, f, ensure_ascii=False, indent=2)
    
    def save_tree(self):
        assert self._root_id is not None
        os.makedirs(self.out_dir, exist_ok=True)

        node_records = []
        for node_id in sorted(self._meta.keys()):
            m = self._meta[node_id]

            if m.is_leaf:
                obj_ids, kw_ids, bits = self.get_leaf_payload_mem(node_id)
                node_records.append({
                    "node_id": int(node_id),
                    "is_leaf": True,
                    "obj_ids": obj_ids,
                    "kw_ids": kw_ids,
                    "bits": bits,
                })
            else:
                children_ids = self.get_children_ids_mem(node_id)
                kw_ids = self._mem_internal_kw_ids[node_id]
                bits = self._mem_internal_kw_child_bits[node_id]
                node_records.append({
                    "node_id": int(node_id),
                    "is_leaf": False,
                    "children_ids": children_ids,
                    "kw_ids": kw_ids,
                    "bits": bits,
                })

        _cleanup_legacy_split_files(self.out_dir)

        out_path = os.path.join(self.out_dir, "tree.bin")
        _write_tree_bin(out_path, int(self._root_id), node_records)
