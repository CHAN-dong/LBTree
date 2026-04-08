# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# =========================
# Config (edit these)
# =========================
DATASET_CSV  = "./dataset/dataset.csv"      # can be "" or None
WORKLOAD_CSV = "./dataset/workload_all.csv"     # can be "" or None

SKIP_FIRST_NONEMPTY_LINE_AS_HEADER = True  # your example has "Title"
MIN_TOKENS_PER_LINE = 1

# TF-IDF config (shared encoder)
MIN_DF = 2
MAX_DF = 0.95
MAX_FEATURES = 50000

# Optional SVD before UMAP (helps large vocab / faster)
SVD_DIM = 100          # set 0 to disable

# Reducer
USE_UMAP = True        # if umap not installed, will fallback to t-SNE (approx shared space)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_SEED = 42

# Plot
POINT_SIZE = 10
POINT_ALPHA = 0.85

OUT_DIR = "."
OUT_DATASET_PNG  = "dataset.png"
OUT_WORKLOAD_PNG = "workload_all.png"


# =========================
# Helpers
# =========================
def _exists(path):
    return path is not None and str(path).strip() != "" and os.path.exists(path)

def read_token_lines_csv(path: str, skip_header: bool = True):
    """
    Read a csv-like text file where each non-empty line is a token sequence.
    Example:
      Title
      kw295 kw371
      kw213 kw374 kw0 kw1
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    if skip_header and len(lines) > 0:
        lines = lines[1:]  # drop first non-empty line (e.g., "Title")

    cleaned = []
    for ln in lines:
        ln = ln.replace('"', " ").replace("'", " ")
        ln = re.sub(r"[,\t;]+", " ", ln)   # treat comma/tab/; as separators
        ln = re.sub(r"\s+", " ", ln).strip()
        toks = [t for t in ln.split(" ") if t]
        if len(toks) >= MIN_TOKENS_PER_LINE:
            cleaned.append(" ".join(toks))
    return cleaned

def fit_vectorizer(texts_fit):
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",  # keeps kw123 / english tokens
        lowercase=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        norm="l2"
    )
    X_fit = vec.fit_transform(texts_fit)
    return vec, X_fit

def transform_texts(vec, texts):
    return vec.transform(texts)

def svd_fit_transform(X):
    if SVD_DIM and SVD_DIM > 0 and X.shape[1] > SVD_DIM:
        svd = TruncatedSVD(n_components=SVD_DIM, random_state=RANDOM_SEED)
        Z = svd.fit_transform(X)
        Z = normalize(Z)
        return svd, Z
    return None, X

def svd_transform(svd, X):
    if svd is None:
        return X
    Z = svd.transform(X)
    Z = normalize(Z)
    return Z

def fit_umap(Z_anchor):
    import umap
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_SEED
    )
    emb_anchor = reducer.fit_transform(Z_anchor)
    return reducer, emb_anchor

def fit_tsne_concat(Z_a, Z_b=None):
    from sklearn.manifold import TSNE
    A = Z_a.toarray() if hasattr(Z_a, "toarray") else np.asarray(Z_a)
    if Z_b is not None:
        B = Z_b.toarray() if hasattr(Z_b, "toarray") else np.asarray(Z_b)
        Z_all = np.vstack([A, B])
    else:
        Z_all = A

    perplex = min(30, max(5, (Z_all.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplex,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED
    )
    emb_all = tsne.fit_transform(Z_all)
    if Z_b is None:
        return emb_all, None  # only anchor
    emb_a = emb_all[:A.shape[0]]
    emb_b = emb_all[A.shape[0]:]
    return emb_a, emb_b

def plot_scatter_onecolor(emb, title, out_path, xlim=None, ylim=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], s=POINT_SIZE, alpha=POINT_ALPHA)  # single color
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[ok] saved: {out_path}")
    plt.show()


# =========================
# Main
# =========================
def main():
    has_dataset = _exists(DATASET_CSV)
    has_workload = _exists(WORKLOAD_CSV)
    if not has_dataset and not has_workload:
        raise RuntimeError("Both DATASET_CSV and WORKLOAD_CSV are empty/nonexistent.")

    dataset_texts = read_token_lines_csv(DATASET_CSV, SKIP_FIRST_NONEMPTY_LINE_AS_HEADER) if has_dataset else []
    workload_texts = read_token_lines_csv(WORKLOAD_CSV, SKIP_FIRST_NONEMPTY_LINE_AS_HEADER) if has_workload else []

    print(f"[info] dataset lines:  {len(dataset_texts)}")
    print(f"[info] workload lines: {len(workload_texts)}")

    # Fit encoder on dataset if exists, else on workload
    fit_texts = dataset_texts if has_dataset else workload_texts
    if len(fit_texts) < 5:
        raise RuntimeError("Too few lines to fit TF-IDF (need >= 5).")

    vec, X_fit = fit_vectorizer(fit_texts)

    X_dataset = transform_texts(vec, dataset_texts) if has_dataset else None
    X_workload = transform_texts(vec, workload_texts) if has_workload else None

    # SVD: fit on encoder-fit texts, then transform both
    svd, Z_fit = svd_fit_transform(X_fit)
    Z_dataset = svd_transform(svd, X_dataset) if has_dataset else None
    Z_workload = svd_transform(svd, X_workload) if has_workload else None

    os.makedirs(OUT_DIR, exist_ok=True)

    # Prefer UMAP for true shared transform space
    if USE_UMAP:
        try:
            # Anchor space on dataset if exists else workload
            Z_anchor = Z_dataset if has_dataset else Z_workload
            reducer, emb_anchor = fit_umap(Z_anchor)
            method = "UMAP"

            if has_dataset:
                emb_dataset = emb_anchor
            else:
                emb_dataset = None

            if has_workload:
                if has_dataset:
                    emb_workload = reducer.transform(Z_workload)
                else:
                    emb_workload = emb_anchor
            else:
                emb_workload = None

        except Exception as e:
            print("[warn] UMAP not available/failed, fallback to t-SNE. Reason:", e)
            USE_TSNE = True
        else:
            USE_TSNE = False
    else:
        USE_TSNE = True

    # t-SNE fallback (approx shared space by concat fit)
    if 'USE_TSNE' in locals() and USE_TSNE:
        method = "t-SNE"
        if has_dataset and has_workload:
            emb_dataset, emb_workload = fit_tsne_concat(Z_dataset, Z_workload)
        elif has_dataset:
            emb_dataset, _ = fit_tsne_concat(Z_dataset, None)
            emb_workload = None
        else:
            emb_workload, _ = fit_tsne_concat(Z_workload, None)
            emb_dataset = None

    # Shared axis limits for fair comparison
    all_embs = []
    if emb_dataset is not None: all_embs.append(emb_dataset)
    if emb_workload is not None: all_embs.append(emb_workload)
    all_xy = np.vstack(all_embs)
    pad_x = 0.05 * (all_xy[:, 0].max() - all_xy[:, 0].min() + 1e-9)
    pad_y = 0.05 * (all_xy[:, 1].max() - all_xy[:, 1].min() + 1e-9)
    xlim = (all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ylim = (all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)

    if has_dataset:
        plot_scatter_onecolor(
            emb_dataset,
            title=f"Dataset 2D Distribution ({method})",
            out_path=os.path.join(OUT_DIR, OUT_DATASET_PNG),
            xlim=xlim, ylim=ylim
        )

    if has_workload:
        plot_scatter_onecolor(
            emb_workload,
            title=f"Workload 2D Distribution ({method})",
            out_path=os.path.join(OUT_DIR, OUT_WORKLOAD_PNG),
            xlim=xlim, ylim=ylim
        )

    print("[done]")

if __name__ == "__main__":
    main()
