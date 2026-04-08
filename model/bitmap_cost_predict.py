# -*- coding: utf-8 -*-
"""
Predict bitmap query time and estimate bitmap index storage size.

Usage:
    python predict_bitmap_cost_and_storage.py \
        --dataset /path/to/dataset.csv \
        --workload /path/to/workload.csv \
        --out_dir ./bitmap_cost_output

Cost model:
    time_ns = epsilon + alpha * (qk * N) + beta * N + gamma * R

where:
    N  = number of objects in dataset
    qk = query keyword count
    R  = result_count of the query (AND semantics by default)

Supported input formats:
1) CSV:
   - single-column CSV: each row is one object/query, cell contains keywords separated by spaces/commas/; or |
   - multi-column CSV: each non-empty cell in the row is treated as one keyword
2) TXT:
   - each line is one object/query, tokens separated by spaces/commas/; or |
3) JSONL:
   - each line is a JSON object
   - it will try fields in order: ["keywords", "tokens", "query", "text", "content"]
   - if the chosen field is a list -> directly use it
   - if it is a string -> split into tokens

Notes:
- For bitmap index semantics, repeated keywords inside the same object are deduplicated.
- Storage size reported is for dense bitmap index:
    packed_bytes     = V * ceil(N / 8)
    aligned_u64_bytes = V * ceil(N / 64) * 8
  where V = number of distinct keywords.
"""

import os
import re
import json
import math
import argparse
from collections import defaultdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("Please install pandas first: pip install pandas")


# ============================================================
# Default cost model parameters (your fitted analytical model)
# ============================================================
DEFAULT_EPSILON = 380.17
DEFAULT_ALPHA   = 0.106
DEFAULT_BETA    = 0.0
DEFAULT_GAMMA   = 1.749


# ============================================================
# Tokenization / Parsing
# ============================================================
_SPLIT_RE = re.compile(r"[,\s;|]+")


def split_keywords_from_string(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    toks = [t.strip() for t in _SPLIT_RE.split(s) if t.strip()]
    return toks


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def row_to_tokens_from_series(row: pd.Series):
    """
    CSV row -> token list

    Rule:
    - if row has exactly 1 non-empty cell, split that cell
    - otherwise, treat each non-empty cell as one token
    """
    vals = []
    for v in row.tolist():
        if pd.isna(v):
            continue
        sv = str(v).strip()
        if sv == "":
            continue
        vals.append(sv)

    if len(vals) == 0:
        return []

    if len(vals) == 1:
        return split_keywords_from_string(vals[0])

    # multi-column mode: each cell is treated as one keyword
    return vals


def load_rows_from_csv(path: str):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        toks = row_to_tokens_from_series(row)
        rows.append(toks)
    return rows


def load_rows_from_txt(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(split_keywords_from_string(line))
    return rows


def load_rows_from_jsonl(path: str):
    candidate_fields = ["keywords", "tokens", "query", "text", "content"]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            value = None
            for k in candidate_fields:
                if k in obj:
                    value = obj[k]
                    break

            if value is None:
                raise ValueError(
                    f"JSONL line {line_id + 1}: cannot find any of {candidate_fields}"
                )

            if isinstance(value, list):
                toks = [str(x).strip() for x in value if str(x).strip()]
            else:
                toks = split_keywords_from_string(str(value))

            rows.append(toks)
    return rows


def load_keyword_rows(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        rows = load_rows_from_csv(path)
    elif ext in [".txt", ".tsv"]:
        rows = load_rows_from_txt(path)
    elif ext in [".jsonl", ".json"]:
        rows = load_rows_from_jsonl(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. "
            f"Supported: .csv, .txt, .tsv, .jsonl, .json"
        )

    # remove empty rows
    rows = [r for r in rows if len(r) > 0]
    return rows


# ============================================================
# Build bitmap-like inverted structure
# ============================================================
def build_postings(dataset_rows):
    """
    Build postings for exact result_count evaluation.

    Returns:
        postings: dict[str, np.ndarray(sorted object ids)]
        stats: dict
    """
    postings_temp = defaultdict(list)

    total_raw_keyword_occurrences = 0
    total_object_keyword_pairs = 0  # unique keywords per object, summed over objects

    for obj_id, raw_tokens in enumerate(dataset_rows):
        raw_tokens = [str(t).strip() for t in raw_tokens if str(t).strip()]
        total_raw_keyword_occurrences += len(raw_tokens)

        uniq_tokens = unique_preserve_order(raw_tokens)
        total_object_keyword_pairs += len(uniq_tokens)

        for kw in uniq_tokens:
            postings_temp[kw].append(obj_id)

    postings = {}
    for kw, ids in postings_temp.items():
        postings[kw] = np.asarray(ids, dtype=np.int32)

    N = len(dataset_rows)
    V = len(postings)

    stats = {
        "num_objects": int(N),
        "distinct_keywords": int(V),
        "dataset_total_keyword_occurrences_raw": int(total_raw_keyword_occurrences),
        "dataset_total_object_keyword_pairs_unique": int(total_object_keyword_pairs),
        "avg_raw_keywords_per_object": float(total_raw_keyword_occurrences / N) if N > 0 else 0.0,
        "avg_unique_keywords_per_object": float(total_object_keyword_pairs / N) if N > 0 else 0.0,
    }
    return postings, stats


# ============================================================
# Query evaluation
# ============================================================
def and_result_count(query_tokens, postings):
    """
    Exact result count under AND semantics.
    """
    q = unique_preserve_order([str(t).strip() for t in query_tokens if str(t).strip()])
    if len(q) == 0:
        return 0

    arrs = []
    for kw in q:
        arr = postings.get(kw, None)
        if arr is None or arr.size == 0:
            return 0
        arrs.append(arr)

    arrs.sort(key=lambda x: x.size)

    cur = arrs[0]
    for arr in arrs[1:]:
        cur = np.intersect1d(cur, arr, assume_unique=True)
        if cur.size == 0:
            return 0
    return int(cur.size)


def or_result_count(query_tokens, postings):
    """
    Exact result count under OR semantics.
    """
    q = unique_preserve_order([str(t).strip() for t in query_tokens if str(t).strip()])
    if len(q) == 0:
        return 0

    arrs = []
    for kw in q:
        arr = postings.get(kw, None)
        if arr is not None and arr.size > 0:
            arrs.append(arr)

    if len(arrs) == 0:
        return 0

    cur = arrs[0]
    for arr in arrs[1:]:
        cur = np.union1d(cur, arr)
    return int(cur.size)


def predict_query_time_ns(N, qk, R, epsilon, alpha, beta, gamma):
    # analytical model:
    # y = epsilon + alpha*(qk*N) + beta*N + gamma*R
    y = epsilon + alpha * (qk * N) + beta * N + gamma * R
    return max(float(y), 0.0)


# ============================================================
# Storage size estimation for dense bitmap index
# ============================================================
def estimate_bitmap_storage_size(num_objects, distinct_keywords, vocabulary=None):
    """
    Dense bitmap storage:
      1) tightly packed bits:       V * ceil(N / 8)
      2) uint64-aligned bitmaps:    V * ceil(N / 64) * 8

    vocabulary_bytes_utf8 is optional, just for reference.
    """
    N = int(num_objects)
    V = int(distinct_keywords)

    packed_bytes = V * ((N + 7) // 8)
    aligned_u64_bytes = V * (((N + 63) // 64) * 8)

    vocab_bytes_utf8 = 0
    if vocabulary is not None:
        vocab_bytes_utf8 = int(sum(len(str(k).encode("utf-8")) for k in vocabulary))

    return {
        "bitmap_packed_bytes": int(packed_bytes),
        "bitmap_packed_kib": float(packed_bytes / 1024.0),
        "bitmap_packed_mib": float(packed_bytes / 1024.0 / 1024.0),

        "bitmap_u64_aligned_bytes": int(aligned_u64_bytes),
        "bitmap_u64_aligned_kib": float(aligned_u64_bytes / 1024.0),
        "bitmap_u64_aligned_mib": float(aligned_u64_bytes / 1024.0 / 1024.0),

        "vocabulary_bytes_utf8_only": int(vocab_bytes_utf8),
        "vocabulary_kib_utf8_only": float(vocab_bytes_utf8 / 1024.0),
    }


# ============================================================
# Main pipeline
# ============================================================
def run_prediction(
    dataset_path,
    workload_path,
    out_dir,
    epsilon=DEFAULT_EPSILON,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    gamma=DEFAULT_GAMMA,
    query_mode="AND",
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) load
    dataset_rows = load_keyword_rows(dataset_path)
    workload_rows = load_keyword_rows(workload_path)

    if len(dataset_rows) == 0:
        raise ValueError("Dataset is empty after parsing.")
    if len(workload_rows) == 0:
        raise ValueError("Workload is empty after parsing.")

    # 2) build postings
    postings, ds_stats = build_postings(dataset_rows)
    N = ds_stats["num_objects"]
    V = ds_stats["distinct_keywords"]

    # 3) estimate storage size
    storage = estimate_bitmap_storage_size(
        num_objects=N,
        distinct_keywords=V,
        vocabulary=postings.keys(),
    )

    # 4) predict each query
    records = []
    total_pred_ns = 0.0

    query_mode_upper = str(query_mode).upper().strip()
    if query_mode_upper not in {"AND", "OR"}:
        raise ValueError("query_mode must be either 'AND' or 'OR'")

    for qid, raw_query in enumerate(workload_rows):
        q_tokens = unique_preserve_order([str(t).strip() for t in raw_query if str(t).strip()])
        qk = len(q_tokens)

        if query_mode_upper == "AND":
            R = and_result_count(q_tokens, postings)
        else:
            R = or_result_count(q_tokens, postings)

        pred_ns = predict_query_time_ns(
            N=N,
            qk=qk,
            R=R,
            epsilon=epsilon,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        total_pred_ns += pred_ns

        records.append({
            "query_id": int(qid),
            "query_keywords": " ".join(q_tokens),
            "query_keyword_count": int(qk),
            "result_count": int(R),
            "pred_query_time_ns": float(pred_ns),
            "pred_query_time_us": float(pred_ns / 1e3),
            "pred_query_time_ms": float(pred_ns / 1e6),
        })

    pred_df = pd.DataFrame(records)
    pred_csv_path = os.path.join(out_dir, "query_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    summary = {
        "dataset_path": dataset_path,
        "workload_path": workload_path,
        "query_mode": query_mode_upper,

        "cost_model": {
            "formula": "time_ns = epsilon + alpha*(qk*N) + beta*N + gamma*R",
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        },

        "dataset_stats": ds_stats,
        "index_storage_size": storage,

        "workload_stats": {
            "num_queries": int(len(pred_df)),
            "avg_query_keyword_count": float(pred_df["query_keyword_count"].mean()) if len(pred_df) > 0 else 0.0,
            "avg_result_count": float(pred_df["result_count"].mean()) if len(pred_df) > 0 else 0.0,
            "total_pred_query_time_ns": float(pred_df["pred_query_time_ns"].sum()) if len(pred_df) > 0 else 0.0,
            "avg_pred_query_time_ns": float(pred_df["pred_query_time_ns"].mean()) if len(pred_df) > 0 else 0.0,
            "avg_pred_query_time_us": float(pred_df["pred_query_time_us"].mean()) if len(pred_df) > 0 else 0.0,
            "avg_pred_query_time_ms": float(pred_df["pred_query_time_ms"].mean()) if len(pred_df) > 0 else 0.0,
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # print concise summary
    print("\n==================== Summary ====================")
    print(f"Dataset path   : {dataset_path}")
    print(f"Workload path  : {workload_path}")
    print(f"Query mode     : {query_mode_upper}")
    print(f"N objects      : {N}")
    print(f"V keywords     : {V}")
    print(f"#Queries       : {len(pred_df)}")
    print()
    print("Cost model:")
    print(f"  time_ns = {epsilon} + {alpha}*(qk*N) + {beta}*N + {gamma}*R")
    print()
    print("Dense bitmap storage:")
    print(f"  packed bytes      : {storage['bitmap_packed_bytes']}  "
          f"({storage['bitmap_packed_mib']:.6f} MiB)")
    print(f"  uint64 aligned    : {storage['bitmap_u64_aligned_bytes']}  "
          f"({storage['bitmap_u64_aligned_mib']:.6f} MiB)")
    print()
    print("Predicted workload time:")
    print(f"  total ns          : {summary['workload_stats']['total_pred_query_time_ns']:.3f}")
    print(f"  avg ns/query      : {summary['workload_stats']['avg_pred_query_time_ns']:.3f}")
    print(f"  avg us/query      : {summary['workload_stats']['avg_pred_query_time_us']:.6f}")
    print()
    print(f"Saved per-query predictions to: {pred_csv_path}")
    print(f"Saved summary to              : {summary_path}")
    print("=================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict bitmap query time and estimate bitmap index storage size."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--workload", type=str, required=True, help="Path to workload file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)

    parser.add_argument(
        "--query_mode",
        type=str,
        default="AND",
        choices=["AND", "OR", "and", "or"],
        help="Keyword query semantics: AND or OR. Default = AND",
    )
    return parser.parse_args()


def main():

    for i in range(2, 11):
        run_prediction(
            dataset_path="/root/dong_11.02/MLKQI1.29/dataset/dblp_titles.csv",
            workload_path=f"/root/dong_11.02/MLKQI1.29/dataset/dblp/workloads/test_t{i}.csv",
            out_dir="/root/dong_11.02/MLKQI1.29/bitmap_cost_output"
        )


if __name__ == "__main__":
    main()