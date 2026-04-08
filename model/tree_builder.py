# model/tree_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import json
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore

from model.model_split import bisplit_once, SplitResult
from model.query_split import LEFT_BIT, RIGHT_BIT
from model.split_cache import SplitModelCache
from utils.text_io import _norm_packed_bits, _write_tree_bin, _cleanup_legacy_split_files


def _make_leaf_keyword_object_bitmaps(A_ow_all, obj_idx: np.ndarray):
    obj_idx = np.ascontiguousarray(np.asarray(obj_idx, dtype=np.int64).reshape(-1))
    O = int(obj_idx.size)

    if O == 0:
        return (
            obj_idx,
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=np.uint8),
        )

    Aow = A_ow_all[obj_idx].tocsr()
    kw_ids = _keywords_in_objects(A_ow_all, obj_idx).astype(np.int64)
    K = int(kw_ids.size)

    row_bytes = (O + 7) // 8
    if K == 0:
        return (
            obj_idx,
            kw_ids,
            np.zeros((0, row_bytes), dtype=np.uint8),
        )

    kw_pos = {int(kw): i for i, kw in enumerate(kw_ids.tolist())}
    kw_obj_bool = np.zeros((K, O), dtype=bool)

    indptr = Aow.indptr
    indices = Aow.indices
    for o in range(O):
        kws = indices[indptr[o]:indptr[o + 1]]
        for kw in kws:
            i = kw_pos.get(int(kw), None)
            if i is not None:
                kw_obj_bool[i, o] = True

    kw_obj_bits = np.packbits(
        kw_obj_bool.astype(np.uint8),
        axis=1,
        bitorder="little"
    ).astype(np.uint8, copy=False)

    return obj_idx, kw_ids, kw_obj_bits


def _cleanup_legacy_split_files(out_dir: str):
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


    def _cache_leaf_payload(self, node_id: int, obj_ids, kw_ids, kw_obj_bits):
        obj_ids = np.ascontiguousarray(np.asarray(obj_ids, dtype=np.int64).reshape(-1))
        kw_ids = np.ascontiguousarray(np.asarray(kw_ids, dtype=np.int32).reshape(-1))

        O = int(obj_ids.size)
        row_bytes = (O + 7) // 8
        kw_obj_bits = _norm_packed_bits(
            kw_obj_bits,
            rows=int(kw_ids.size),
            row_bytes=row_bytes,
            name=f"leaf bits node={node_id}",
        ).copy()

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
        N_sample: int = 10000,
        K_graph: int = 20,
        epochs: int = 100,
        lr: float = 1e-3,
        # balance_lambda: float = 1e-2,
        balance_lambda: float = 0,
        min_objects: int = 256,
        min_queries: int = 0,
        max_depth: int = 20,
        cost_improve_eps: float = 0.0,
        device: Optional[str] = None,
    ):
        self.A_ow_all = A_ow_all
        self.A_qw_all = A_qw_all
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.N_sample = N_sample
        self.K_graph = K_graph
        self.epochs = epochs
        self.lr = lr
        self.balance_lambda = balance_lambda

        self.min_objects = min_objects
        self.min_queries = min_queries
        self.max_depth = max_depth
        self.cost_improve_eps = cost_improve_eps
        self.device = device

        self.model_cache = SplitModelCache()

        self._next_id = 0
        self._meta: Dict[int, TreeNodeMeta] = {}
        self._root_id: Optional[int] = None

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _build(self, obj_idx: np.ndarray, query_idx: np.ndarray, depth: int) -> int:
        node_id = self._new_id()
        obj_idx = np.asarray(obj_idx, dtype=np.int64)
        query_idx = np.asarray(query_idx, dtype=np.int64)

        # ---- node statistics for DP (Break redundant nodes) ----
        WN = int(query_idx.size)

        if WN == 0:
            sum_q_len_minus1 = 0
        else:
            Aqw_sub = self.A_qw_all[query_idx]  # csr
            q_lens = np.diff(Aqw_sub.indptr).astype(np.int64)
            sum_q_len_minus1 = int(np.maximum(q_lens - 1, 0).sum())

        omegaN = int(_keywords_in_objects(self.A_ow_all, obj_idx).size)


        # try split (S1+S2)
        split: SplitResult = bisplit_once(
            A_ow_all=self.A_ow_all,
            A_qw_all=self.A_qw_all,
            obj_idx=obj_idx,
            query_idx=query_idx,
            N_sample=self.N_sample,
            K_graph=self.K_graph,
            epochs=self.epochs,
            lr=self.lr,
            balance_lambda=self.balance_lambda,
            device=self.device,
            cache=self.model_cache,
            is_root=(depth == 0),
            freeze_gcn=(depth > 0),
            mlp_epochs=30,
            patience=30,
        )

        valid_children = (split.obj_left.size > 0 and split.obj_right.size > 0)
        # improved = (split.cost_parent - split.cost_children) > self.cost_improve_eps
        improved = (split.cost_parent - split.cost_children) > split.cost_parent * 0.1
        # ---------- leaf ----------
        if (not valid_children) or (not improved):

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

            leaf_obj_ids, leaf_kw_ids, leaf_kw_obj_bits = _make_leaf_keyword_object_bitmaps(
                self.A_ow_all, obj_idx
            )
            self._cache_leaf_payload(node_id, leaf_obj_ids, leaf_kw_ids, leaf_kw_obj_bits)

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
        left_id = self._build(split.obj_left, split.q_left, depth + 1)
        right_id = self._build(split.obj_right, split.q_right, depth + 1)
        children_ids = np.array([left_id, right_id], dtype=np.int32)

        # 计算 keyword union + kw->child bits
        left_kw = _keywords_in_objects(self.A_ow_all, split.obj_left).astype(np.int64)
        right_kw = _keywords_in_objects(self.A_ow_all, split.obj_right).astype(np.int64)
        kw_ids = np.union1d(left_kw, right_kw).astype(np.int64)

        left_set = set(left_kw.tolist())
        right_set = set(right_kw.tolist())

        K = int(kw_ids.size)
        kw_child_bool = np.zeros((K, 2), dtype=bool)
        for i, kw in enumerate(kw_ids):
            ikw = int(kw)
            if ikw in left_set:
                kw_child_bool[i, 0] = True
            if ikw in right_set:
                kw_child_bool[i, 1] = True

        kw_child_bits_packed = _pack_rows_child_bits(kw_child_bool)  # (K, 1) uint8

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
            cost=float(split.cost_parent),
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
        self._reset_mem_tree_cache()
        self._root_id = self._build(root_obj_idx, root_query_idx, depth=0)
        self.save_tree()   # 仍然写本地，但只写 tree.bin
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