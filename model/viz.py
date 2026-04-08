# model/viz.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.decomposition import TruncatedSVD


def embed_2d_svd(X_csr: csr_matrix, random_state: int = 0) -> np.ndarray:
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    Z = svd.fit_transform(X_csr)
    return Z.astype(np.float32)


def plot_overlay_s1_s2(
    Z_all_2d: np.ndarray,          # (n,2)
    sample_local: np.ndarray,      # (n_s,)
    cluster_id_sample: np.ndarray, # (n_s,) 0/1
    obj_left_local: np.ndarray,    # (nL,) S2 left (all objects)
    obj_right_local: np.ndarray,   # (nR,) S2 right (all objects)
    out_path: str = "./debug/split_overlay.png",
    title: str = "",
    point_size: float = 12.0,
    base_alpha: float = 0.20,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = Z_all_2d.shape[0]
    sample_local = np.asarray(sample_local, dtype=np.int64)
    cluster_id_sample = np.asarray(cluster_id_sample, dtype=np.int64)
    obj_left_local = np.asarray(obj_left_local, dtype=np.int64)
    obj_right_local = np.asarray(obj_right_local, dtype=np.int64)

    sample_mask = np.zeros(n, dtype=bool)
    sample_mask[sample_local] = True

    # sample split sets
    s_left = sample_local[cluster_id_sample == 0]
    s_right = sample_local[cluster_id_sample == 1]

    # streaming = S2 assigned but NOT in sample
    left_stream = obj_left_local[~sample_mask[obj_left_local]]
    right_stream = obj_right_local[~sample_mask[obj_right_local]]

    fig = plt.figure(figsize=(6.5, 6.0))
    ax = plt.gca()

    # Stage-0 base: all grey
    ax.scatter(Z_all_2d[:, 0], Z_all_2d[:, 1], s=point_size, alpha=base_alpha, label="others")

    # Stage-2 overlay first (light colors), so sample (dark) sits on top
    # Streaming-left: light blue, Streaming-right: light green (you can change if you want)
    if left_stream.size > 0:
        ax.scatter(Z_all_2d[left_stream, 0], Z_all_2d[left_stream, 1],
                   s=point_size, alpha=0.50, label="stream-left")
    if right_stream.size > 0:
        ax.scatter(Z_all_2d[right_stream, 0], Z_all_2d[right_stream, 1],
                   s=point_size, alpha=0.50, label="stream-right")

    # Stage-1 overlay (dark colors)
    # Sample-left: blue, Sample-right: green (dark)
    if s_left.size > 0:
        ax.scatter(Z_all_2d[s_left, 0], Z_all_2d[s_left, 1],
                   s=point_size, alpha=0.95, label="sample-left")
    if s_right.size > 0:
        ax.scatter(Z_all_2d[s_right, 0], Z_all_2d[s_right, 1],
                   s=point_size, alpha=0.95, label="sample-right")

    ax.set_title(title if title else "Overlay: S1(sample, dark) + S2(streaming, light)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
