# utils/text_io.py
import csv
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import json
import os


import struct

_TREE_MAGIC = b"LBTREE1\x00"
_TREE_VERSION = 1


def _norm_i32_1d(arr, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.int32, copy=False))


def _norm_i64_1d(arr, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))


def _norm_packed_bits(bits, rows: int, row_bytes: int, name: str) -> np.ndarray:
    if rows == 0:
        return np.zeros((0, row_bytes), dtype=np.uint8)

    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={bits.shape}")
    if bits.shape[0] != rows or bits.shape[1] != row_bytes:
        raise ValueError(
            f"{name} shape mismatch: expected ({rows}, {row_bytes}), got {bits.shape}"
        )
    return np.ascontiguousarray(bits)


def _write_tree_bin(out_path: str, root_id: int, node_records: list):
    """
    node_records:
      leaf:
        {
          "node_id": int,
          "is_leaf": True,
          "obj_ids": np.ndarray[int64] shape [O],
          "kw_ids": np.ndarray[int32/int64] shape [K],
          "bits": np.ndarray[uint8] shape [K, ceil(O/8)]
        }

      internal:
        {
          "node_id": int,
          "is_leaf": False,
          "children_ids": np.ndarray[int32/int64] shape [C],
          "kw_ids": np.ndarray[int32/int64] shape [K],
          "bits": np.ndarray[uint8] shape [K, ceil(C/8)]
        }
    """
    node_records = sorted(node_records, key=lambda x: int(x["node_id"]))

    with open(out_path, "wb") as f:
        f.write(_TREE_MAGIC)
        f.write(struct.pack("<i", int(_TREE_VERSION)))
        f.write(struct.pack("<i", int(root_id)))
        f.write(struct.pack("<i", int(len(node_records))))

        for rec in node_records:
            node_id = int(rec["node_id"])
            is_leaf = bool(rec["is_leaf"])

            f.write(struct.pack("<i", node_id))
            f.write(struct.pack("<B", 1 if is_leaf else 0))

            if is_leaf:
                obj_ids = _norm_i64_1d(rec["obj_ids"], "obj_ids")
                kw_ids = _norm_i32_1d(rec["kw_ids"], "kw_ids")

                O = int(obj_ids.size)
                row_bytes = (O + 7) // 8
                bits = _norm_packed_bits(rec["bits"], rows=int(kw_ids.size), row_bytes=row_bytes, name="leaf bits")

                f.write(struct.pack("<i", O))
                if O > 0:
                    f.write(obj_ids.tobytes())

                K = int(kw_ids.size)
                f.write(struct.pack("<i", K))
                for i in range(K):
                    f.write(struct.pack("<i", int(kw_ids[i])))
                    if row_bytes > 0:
                        f.write(bits[i].tobytes())

            else:
                children_ids = _norm_i32_1d(rec["children_ids"], "children_ids")
                kw_ids = _norm_i32_1d(rec["kw_ids"], "kw_ids")

                C = int(children_ids.size)
                row_bytes = (C + 7) // 8
                bits = _norm_packed_bits(rec["bits"], rows=int(kw_ids.size), row_bytes=row_bytes, name="internal bits")

                f.write(struct.pack("<i", C))
                if C > 0:
                    f.write(children_ids.tobytes())

                K = int(kw_ids.size)
                f.write(struct.pack("<i", K))
                for i in range(K):
                    f.write(struct.pack("<i", int(kw_ids[i])))
                    if row_bytes > 0:
                        f.write(bits[i].tobytes())



_WORD_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return _WORD_RE.findall(text)

def _build_csr_from_tokens(tokens_list: List[Set[str]], vocab: Dict[str, int]):
    from scipy.sparse import csr_matrix  # type: ignore
    rows, cols, data = [], [], []
    for i, toks in enumerate(tokens_list):
        for t in toks:
            j = vocab.get(t, None)
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(1)
    n = len(tokens_list)
    d = len(vocab)
    return csr_matrix((data, (rows, cols)), shape=(n, d), dtype=np.int8)

def build_Aow_split_from_objects(
    csv_path: str,
    N: int,
    text_column: str = "Title",
    delimiter: str = ",",
    encoding: str = "utf-8",
):
    all_tokens: List[Set[str]] = []

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            text = (row.get(text_column) or "").strip()
            all_tokens.append(set(tokenize(text)))

    n_all = len(all_tokens)
    N = min(N, n_all)
    tokens_sample = all_tokens[:N]
    tokens_rest = all_tokens[N:]

    # vocab from sample only
    vocab: Dict[str, int] = {}
    for toks in tokens_sample:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)

    A_ow_sample = _build_csr_from_tokens(tokens_sample, vocab)
    A_ow_rest = _build_csr_from_tokens(tokens_rest, vocab)
    return A_ow_sample, A_ow_rest, vocab, tokens_sample, tokens_rest


def read_csv_build_inverted_index(
    csv_path: str,
    text_column: str = "Title",
    has_header: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8",
    title_col_if_no_header: int = 1,   
) -> Tuple[List[Set[str]], Dict[str, List[int]], Dict[str, int]]:
    docs_tokens: List[Set[str]] = []
    inv_index: Dict[str, List[int]] = defaultdict(list)
    vocab: Dict[str, int] = {}

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        if has_header:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return [], {}, {}

            header_lc = [h.strip() for h in header]

            title_col = None
            for i, h in enumerate(header_lc):
                if h == text_column:
                    title_col = i
                    break

            for row_id, cells in enumerate(reader):
                if not cells:
                    continue

                if title_col is not None and len(header_lc) == 1 and header_lc[0] == "Title" and len(cells) >= 2:
                    text = cells[1].strip()

                elif title_col is not None and title_col < len(cells):
                    text = cells[title_col].strip()

                else:
                    text = " ".join(cells).strip()

                toks = set(tokenize(text))
                docs_tokens.append(toks)
                for t in toks:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                    inv_index[t].append(row_id)
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for row_id, cells in enumerate(reader):
                if not cells:
                    continue

                if len(cells) > title_col_if_no_header:
                    text = cells[title_col_if_no_header].strip()
                else:
                    text = " ".join(cells).strip()

                toks = set(tokenize(text))
                docs_tokens.append(toks)
                for t in toks:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                    inv_index[t].append(row_id)

    for t in inv_index:
        inv_index[t].sort()

    return docs_tokens, dict(inv_index), vocab

def build_binary_matrix_from_inv_index(
    num_rows: int,
    vocab: Dict[str, int],
    inv_index: Dict[str, List[int]],
) -> Any:
    d = len(vocab)

    try:
        from scipy.sparse import csr_matrix  # type: ignore

        rows, cols, data = [], [], []
        for term, row_ids in inv_index.items():
            if term not in vocab:
                continue
            j = vocab[term]
            for i in row_ids:
                if 0 <= i < num_rows:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
        return csr_matrix((data, (rows, cols)), shape=(num_rows, d), dtype=int)

    except ImportError:
        X = [[0] * d for _ in range(num_rows)]
        for term, row_ids in inv_index.items():
            if term not in vocab:
                continue
            j = vocab[term]
            for i in row_ids:
                if 0 <= i < num_rows:
                    X[i][j] = 1
        return X


def build_Aow_from_objects(
    object_csv_path: str,
    object_text_column: str = "Title",
    max_objects: Optional[int] = None,
    save_vocab_path: Optional[str] = None,   
) -> Tuple[Any, Dict[str, int], Dict[str, List[int]]]:
    """
    Read dataset(objects.csv) and build:
      - A_ow: [num_objects, vocab_size]
      - vocab: term -> col_id
      - inv_o: term -> list of object_ids (local ids 0..n-1)
    Optionally save vocab to a json file compatible with C++ loader.
    """
    obj_tokens, inv_o_full, vocab_full = read_csv_build_inverted_index(
        object_csv_path, text_column=object_text_column, has_header=True
    )

    if max_objects is not None:
        n = min(max_objects, len(obj_tokens))
    else:
        n = len(obj_tokens)

    inv_o: Dict[str, List[int]] = {}
    vocab: Dict[str, int] = {}
    for term, postings in inv_o_full.items():
        postings2 = [i for i in postings if i < n]
        if not postings2:
            continue
        vocab[term] = len(vocab)
        inv_o[term] = postings2

    A_ow = build_binary_matrix_from_inv_index(n, vocab, inv_o)

    if save_vocab_path is not None:
        os.makedirs(os.path.dirname(save_vocab_path) or ".", exist_ok=True)
        with open(save_vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"[build_Aow_from_objects] Saved vocab to: {save_vocab_path} (size={len(vocab)})")

    return A_ow, vocab, inv_o


def build_Aqw_from_workload(
    workload_csv_path: str,
    vocab: Dict[str, int],
    query_text_column: str = "Title",
) -> Tuple[Any, Dict[str, List[int]]]:
    """
    Read workload(queries.csv) and build:
      - A_qw: [num_queries, vocab_size]  (columns aligned with objects' vocab)
      - inv_q: term -> list of query_ids
    Notes:
      - terms not in vocab are ignored (typical choice in the paper where d=|Ω_O'|).
    """
    q_tokens, inv_q_raw, _vocab_q = read_csv_build_inverted_index(
        workload_csv_path, text_column=query_text_column, has_header=True
    )
    m = len(q_tokens)

    inv_q: Dict[str, List[int]] = {
        t: ids for t, ids in inv_q_raw.items() if t in vocab
    }

    A_qw = build_binary_matrix_from_inv_index(m, vocab, inv_q)
    return A_qw, inv_q


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