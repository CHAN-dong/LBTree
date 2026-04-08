# model/tree_node_break.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import shutil
import numpy as np

import parameter as param
from utils.text_io import _norm_packed_bits, _write_tree_bin

# ========== helpers: packing keyword->child bitmap ==========
def _pack_kw_child_bits(kw_child_bool: np.ndarray) -> np.ndarray:
    """
    kw_child_bool: (K, C) bool
    return: (K, ceil(C/8)) uint8 packed (bitorder=little)
    """
    if kw_child_bool.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.packbits(kw_child_bool.astype(np.uint8), axis=1, bitorder="little")


# ========== cost used in DP: Cost(N'_{l+r}) per Eq.(33)(34) ==========
def _cost_updated_node(omega: int, WN: int, sum_q_len_minus1: int,
                       branches: int, ws_left: int, ws_right: int) -> float:
    storage = omega * branches + 32 * omega + 32 * branches
    query = (param.EPSILON * WN
             + param.ALFA * sum_q_len_minus1 * branches
             + param.BETA * branches
             + param.GAMA * (ws_left + ws_right))

    return float(param.w_s * param.W_S * storage + param.w_q * param.W_Q * query)

def _print_dp_state(st, chosen_k: int, indent: str = ""):
    K = len(st.CT) - 1 
    print(indent + f"DPState table: K={K}, chosen k={chosen_k}")

    print(indent + f"{'k':>4}  {'CT':>14}  {'WS':>10}  {'PT(l,r)':>12}  mark")
    for kk in range(1, K + 1):
        l, r = st.PT[kk]
        mark = "<-- chosen" if kk == chosen_k else ""
        print(indent + f"{kk:4d}  {st.CT[kk]:14.6g}  {st.WS[kk]:10d}  ({l:2d},{r:2d})  {mark}")

# ========== DP containers ==========
@dataclass
class DPState:
    # 1-indexed arrays: CT[k], WS[k], PT[k]=(l,r)
    CT: List[float]
    WS: List[int]
    PT: List[Tuple[int, int]]


@dataclass
class NewNode:
    """
    Multi-branch node for the broken tree (in memory).
    Leaves reuse original leaf files; internal nodes will be re-materialized.
    """
    old_id: int
    is_leaf: bool
    children: List["NewNode"]
    # for serialization
    data: Dict[str, Any]


# ========== main DP + materialization ==========
class TreeNodeBreaker:
    """
    Implements Algorithm 2 (Break redundant nodes) using the in-memory meta produced by TreeBuilder.

    Assumptions:
    - builder._meta[node_id].data contains:
        omega, WN, sum_q_len_minus1
      and for internal nodes:
        children_ids.npy exists and equals [left_id, right_id]
      and leaf nodes have leaf bitmap files already written.

    Output:
    - Writes a new multi-branch tree json + internal node bitmaps to out_dir_break
    - Leaves are copied (or referenced) from the original out_dir.
    """
    def __init__(self, builder, out_dir_break: str, copy_leaves: bool = True):
        self.builder = builder
        self.meta = builder._meta
        self.old_out_dir = builder.out_dir
        self.out_dir_break = out_dir_break
        self.copy_leaves = copy_leaves
        os.makedirs(self.out_dir_break, exist_ok=True)

        # old node_id -> NewNode(s) cache for materialization by (node_id, k)
        self._mat_cache: Dict[Tuple[int, int], List[NewNode]] = {}

        # dp cache
        self._dp: Dict[int, DPState] = {}

        # new node id allocation
        self._new_id = 0
        self._new_meta: Dict[int, Dict[str, Any]] = {}

    def _alloc_new_id(self) -> int:
        nid = self._new_id
        self._new_id += 1
        return nid

    def _children_of(self, node_id: int) -> Optional[Tuple[int, int]]:
        m = self.meta[node_id]
        if m.is_leaf:
            return None
        ch = self.builder.get_children_ids_mem(node_id)
        if int(ch.size) != 2:
            raise ValueError(f"node {node_id} is expected to be binary before break, got {int(ch.size)} children")
        return int(ch[0]), int(ch[1])

    def _node_stats(self, node_id: int) -> Tuple[int, int, int]:
        """
        return (omega, WN, sum_q_len_minus1)
        """
        d = self.meta[node_id].data
        return int(d.get("omega", 0)), int(d.get("WN", 0)), int(d.get("sum_q_len_minus1", 0))

    # -------- DPMinimalCostTree (postorder) --------
    def _dp_compute(self, node_id: int) -> DPState:
        if node_id in self._dp:
            return self._dp[node_id]

        m = self.meta[node_id]
        omega, WN, sum_q_len_minus1 = self._node_stats(node_id)

        if m.is_leaf:
            leaf_cost = float(m.cost) if (m.cost is not None) else 0.0

            CT = [float("inf"), leaf_cost]
            WS = [0, WN]  # leaf: each query accesses exactly 1 branch => sum branch hits = |W_N|
            PT = [(0, 0), (1, 1)]
            st = DPState(CT=CT, WS=WS, PT=PT)
            self._dp[node_id] = st
            return st

        left_id, right_id = self._children_of(node_id)  # type: ignore
        stL = self._dp_compute(left_id)
        stR = self._dp_compute(right_id)

        # DP size: possible branch counts = [1..(maxL+maxR)]
        maxL = len(stL.CT) - 1
        maxR = len(stR.CT) - 1
        maxK = maxL + maxR

        CT = [float("inf")] * (maxK + 1)
        WS = [0] * (maxK + 1)
        PT = [(0, 0)] * (maxK + 1)

        # enumerate all (l,r) combinations (Algorithm 2 line 11-21)
        for l in range(1, maxL + 1):
            for r in range(1, maxR + 1):
                # break case: k = l + r, CT[k] = CT_L[l]+CT_R[r]
                k = l + r
                cost_br = stL.CT[l] + stR.CT[r]
                if cost_br < CT[k]:
                    CT[k] = cost_br
                    WS[k] = stL.WS[l] + stR.WS[r]
                    PT[k] = (l, r)

                # retain case: CT[1] = min_{l,r} CT_L[l]+CT_R[r]+Cost(N'_{l+r})
                # Cost(N'_{l+r}) uses Eq.(33)(34)
                cost_upd = _cost_updated_node(
                    omega=omega,
                    WN=WN,
                    sum_q_len_minus1=sum_q_len_minus1,
                    branches=(l + r),
                    ws_left=stL.WS[l],
                    ws_right=stR.WS[r],
                )
                cost_keep = stL.CT[l] + stR.CT[r] + cost_upd
                if cost_keep < CT[1]:
                    CT[1] = cost_keep
                    WS[1] = WN  # Algorithm 2 line 16 sets WS_N[1]=|W_N|
                    PT[1] = (l, r)

        st = DPState(CT=CT, WS=WS, PT=PT)
        self._dp[node_id] = st
        return st


    # -------- DPBackTrace + materialize into multi-branch tree --------
    def _materialize(self, node_id: int, k: int, depth: int = 0) -> List[NewNode]:
        """
        Return list of branch roots produced by subtree(node_id) under state k.
        If k==1 => returns [a retained node]
        If k>1  => returns concatenated branches of its children (node broken)
        """
        key = (node_id, k)
        if key in self._mat_cache:
            return self._mat_cache[key]

        m = self.meta[node_id]
        st = self._dp_compute(node_id)

        # leaf always retained
        if m.is_leaf:
            nn = NewNode(old_id=node_id, is_leaf=True, children=[], data={"kind": "leaf", "old_id": node_id})
            self._mat_cache[key] = [nn]
            return [nn]

        left_id, right_id = self._children_of(node_id)  # type: ignore
        l, r = st.PT[k]

        indent = "  " * depth
        
        # _print_dp_state(st, k, indent=indent)
        l, r = st.PT[k]
        # print(indent + f"chosen: PT[{k}] = ({l},{r}), CT[{k}]={st.CT[k]}, WS[{k}]={st.WS[k]}")

        if k == 1:
            # retain node: its children are l branches from left + r branches from right
            left_branches = self._materialize(left_id, l, depth + 1)
            right_branches = self._materialize(right_id, r, depth + 1)
            nn = NewNode(old_id=node_id, is_leaf=False, children=left_branches + right_branches,
                         data={"kind": "internal", "old_id": node_id, "k": k, "chosen": [l, r]})
            out = [nn]
        else:
            # break node: return branches directly
            out = self._materialize(left_id, l) + self._materialize(right_id, r)

        self._mat_cache[key] = out
        return out

    # -------- write new tree + internal bitmaps --------
    def _subtree_keywords(self, new_node: NewNode) -> np.ndarray:
        if new_node.is_leaf:
            return np.asarray(self.builder.get_node_keywords_mem(new_node.old_id), dtype=np.int64)

        if not new_node.children:
            return np.array([], dtype=np.int64)

        all_kw = [self._subtree_keywords(ch) for ch in new_node.children]
        non_empty = [x for x in all_kw if x.size > 0]
        if len(non_empty) == 0:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(non_empty)).astype(np.int64)

    def _materialize_and_save(self, new_root: NewNode) -> int:
        node_records = []
        self._new_meta = {}

        def dfs(node: NewNode, depth: int) -> int:
            new_id = self._alloc_new_id()

            if node.is_leaf:
                obj_ids, kw_ids, kw_obj_bits = self.builder.get_leaf_payload_mem(node.old_id)

                obj_ids = np.ascontiguousarray(np.asarray(obj_ids, dtype=np.int64).reshape(-1))
                kw_ids = np.ascontiguousarray(np.asarray(kw_ids, dtype=np.int32).reshape(-1))

                O = int(obj_ids.size)
                row_bytes = (O + 7) // 8
                K = int(kw_ids.size)
                kw_obj_bits = _norm_packed_bits(
                    kw_obj_bits, rows=K, row_bytes=row_bytes, name=f"break leaf bits node={new_id}"
                )

                node_records.append({
                    "node_id": int(new_id),
                    "is_leaf": True,
                    "obj_ids": obj_ids,
                    "kw_ids": kw_ids,
                    "bits": kw_obj_bits,
                })

                self._new_meta[new_id] = {
                    "node_id": int(new_id),
                    "depth": int(depth),
                    "is_leaf": True,
                    "kind": "leaf",
                    "source_old_id": int(node.old_id),
                }
                return new_id

            child_new_ids = []
            for ch in node.children:
                child_new_ids.append(dfs(ch, depth + 1))
            child_new_ids_np = np.ascontiguousarray(np.asarray(child_new_ids, dtype=np.int32))

            child_kws = [set(self._subtree_keywords(ch).tolist()) for ch in node.children]
            if any(len(s) > 0 for s in child_kws):
                kw_ids = np.unique(
                    np.concatenate([np.asarray(list(s), dtype=np.int64) for s in child_kws if len(s) > 0])
                )
            else:
                kw_ids = np.asarray([], dtype=np.int64)

            C = len(node.children)
            K = int(kw_ids.size)

            kw_child_bool = np.zeros((K, C), dtype=bool)
            for i, kw in enumerate(kw_ids.tolist()):
                for c in range(C):
                    if kw in child_kws[c]:
                        kw_child_bool[i, c] = True

            kw_child_bits = _pack_kw_child_bits(kw_child_bool)
            row_bytes = (C + 7) // 8
            kw_child_bits = _norm_packed_bits(
                kw_child_bits, rows=K, row_bytes=row_bytes, name=f"break internal bits node={new_id}"
            )

            node_records.append({
                "node_id": int(new_id),
                "is_leaf": False,
                "children_ids": child_new_ids_np,
                "kw_ids": kw_ids,
                "bits": kw_child_bits,
            })

            self._new_meta[new_id] = {
                "node_id": int(new_id),
                "depth": int(depth),
                "is_leaf": False,
                "kind": "internal",
                "child_count": int(C),
                "kw_count": int(K),
                "source_old_id": int(node.old_id),
            }
            return new_id

        new_root_id = dfs(new_root, 0)

        for fn in os.listdir(self.out_dir_break):
            if fn.endswith(".npy") or fn == "tree.json":
                path = os.path.join(self.out_dir_break, fn)
                if os.path.isfile(path):
                    os.remove(path)

        out_path = os.path.join(self.out_dir_break, "tree.bin")
        _write_tree_bin(out_path, int(new_root_id), node_records)

        return new_root_id



    # def _materialize_and_save(self, new_root: NewNode) -> int:
    #     """
    #     Assign new ids, save internal kw_child_bits, and save new tree.json.
    #     Leaves are copied or referenced.
    #     """
    #     def dfs(node: NewNode, depth: int) -> int:
    #         new_id = self._alloc_new_id()

    #         if node.is_leaf:
    #             # copy leaf files
    #             if self.copy_leaves:
    #                 for suffix in ["obj_ids.npy", "leaf_kw_ids.npy", "leaf_kw_obj_bits.npy"]:
    #                     src = os.path.join(self.old_out_dir, f"node_{node.old_id}_{suffix}")
    #                     dst = os.path.join(self.out_dir_break, f"node_{new_id}_{suffix}")
    #                     if os.path.exists(src):
    #                         shutil.copy2(src, dst)
    #             self._new_meta[new_id] = {
    #                 "node_id": new_id,
    #                 "depth": depth,
    #                 "is_leaf": True,
    #                 "kind": "leaf",
    #                 "data": {
    #                     "obj_ids": f"node_{new_id}_obj_ids.npy",
    #                     "kw_ids": f"node_{new_id}_leaf_kw_ids.npy",
    #                     "kw_obj_bits": f"node_{new_id}_leaf_kw_obj_bits.npy",
    #                     "source_old_id": int(node.old_id),
    #                 }
    #             }
    #             return new_id

    #         # internal: first save children
    #         child_new_ids = []
    #         for ch in node.children:
    #             child_new_ids.append(dfs(ch, depth + 1))
    #         child_new_ids_np = np.array(child_new_ids, dtype=np.int64)

    #         # compute kw union and kw->child membership bits
    #         child_kws = [set(self._subtree_keywords(ch).tolist()) for ch in node.children]
    #         kw_ids = np.unique(np.concatenate([np.array(list(s), dtype=np.int64) for s in child_kws if len(s) > 0])) \
    #                  if any(len(s) > 0 for s in child_kws) else np.array([], dtype=np.int64)

    #         C = len(node.children)
    #         K = int(kw_ids.size)
    #         kw_child_bool = np.zeros((K, C), dtype=bool)
    #         for i, kw in enumerate(kw_ids.tolist()):
    #             for c in range(C):
    #                 if kw in child_kws[c]:
    #                     kw_child_bool[i, c] = True

    #         kw_child_bits = _pack_kw_child_bits(kw_child_bool)

    #         # save
    #         np.save(os.path.join(self.out_dir_break, f"node_{new_id}_children_ids.npy"), child_new_ids_np)
    #         np.save(os.path.join(self.out_dir_break, f"node_{new_id}_kw_ids.npy"), kw_ids.astype(np.int64))
    #         np.save(os.path.join(self.out_dir_break, f"node_{new_id}_kw_child_bits.npy"), kw_child_bits.astype(np.uint8))

    #         self._new_meta[new_id] = {
    #             "node_id": new_id,
    #             "depth": depth,
    #             "is_leaf": False,
    #             "kind": "internal",
    #             "data": {
    #                 "children_ids": f"node_{new_id}_children_ids.npy",
    #                 "kw_ids": f"node_{new_id}_kw_ids.npy",
    #                 "kw_child_bits": f"node_{new_id}_kw_child_bits.npy",
    #                 "child_count": int(C),
    #                 "kw_count": int(K),
    #                 "source_old_id": int(node.old_id),
    #             }
    #         }
    #         return new_id

    #     new_root_id = dfs(new_root, 0)

    #     # write json
    #     nodes = [self._new_meta[i] for i in sorted(self._new_meta.keys())]
    #     out = {"root": int(new_root_id), "nodes": nodes}
    #     with open(os.path.join(self.out_dir_break, "tree.json"), "w", encoding="utf-8") as f:
    #         json.dump(out, f, ensure_ascii=False, indent=2)

    #     return new_root_id

    def run(self, root_old_id: int) -> Dict[str, Any]:
        st_root = self._dp_compute(root_old_id)

        roots = self._materialize(root_old_id, 1)
        assert len(roots) == 1
        new_root = roots[0]

        new_root_id = self._materialize_and_save(new_root)

        # DP perspective:
        cost_before = float(self._dp_compute(root_old_id).CT[1])   # retained root cost
        # after-break cost = optimal "broken" cost among k>=2, i.e. min_k CT[k]
        st = self._dp_compute(root_old_id)
        if len(st.CT) <= 2:
            cost_after = cost_before
            best_k = 1
        else:
            best_k = int(np.argmin(st.CT[1:])) + 1
            cost_after = float(st.CT[best_k])

        return {
            "old_root": int(root_old_id),
            "new_root": int(new_root_id),
            "break_dir": self.out_dir_break,
            "cost_before_dp": cost_before,
            "cost_after_dp": cost_after,
            "best_k_root": int(best_k),
        }


def break_redundant_nodes(builder, root_id: int, out_dir_break: str = "./tree_out_break",
                          copy_leaves: bool = True) -> Dict[str, Any]:
    """
    Convenience wrapper.
    """
    breaker = TreeNodeBreaker(builder, out_dir_break=out_dir_break, copy_leaves=copy_leaves)
    return breaker.run(root_id)
