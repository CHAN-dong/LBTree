import os
import json
import random
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("install pandas: pip install pandas")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config (EDIT HERE)
# =========================
CSV_TRAIN = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/bitmap_samples_controlled.csv"
CSV_PLOT  = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/bitmap_plot_testset_controlled.csv"

CKPT_PATH = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/Mc_mlp_cost_model.pt"
META_PATH = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/Mc_mlp_cost_model_meta.json"

# Features you want
FEATURE_COLS = [
    "num_objects",
    "dataset_total_keyword_occurrences",
    "dataset_distinct_keywords",
    "query_keyword_count",
    "result_count",
]
TARGET_COL = "query_time_ns"

# Data preprocessing
AGGREGATE_DUPLICATES_BY_FEATURES = True   # median per feature combo
USE_LOG_TARGET = True

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training
SEED = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 40

# MLP
HIDDEN_DIMS = [64, 64, 32]
DROPOUT = 0.05

# Export plot data
EXPORT_XLSX = True
EXPORT_XLSX_PATH = "Mc_mlp_plot_line_point_data.xlsx"
EXPORT_CSV_DIR = "Mc_mlp_plot_line_point_data_csv"  # fallback if openpyxl missing


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_safe(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def r2_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @staticmethod
    def fit(x: np.ndarray):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return Standardizer(mean=mean, std=std)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_and_prepare(csv_path, feature_cols, target_col, aggregate_duplicates=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    df = df[df[target_col] >= 0].reset_index(drop=True)

    if aggregate_duplicates:
        df = (
            df.groupby(feature_cols, as_index=False)[target_col]
            .median()
            .reset_index(drop=True)
        )

    if len(df) == 0:
        raise ValueError("No valid rows after preprocessing.")
    return df


def split_indices(n, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().reshape(-1)
        preds.append(out)
        ys.append(yb.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(ys)


def train_mlp(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_raw = df[TARGET_COL].to_numpy(dtype=np.float64)

    train_idx, val_idx, test_idx = split_indices(len(df), seed=SEED)

    X_train_raw, y_train_raw = X[train_idx], y_raw[train_idx]
    X_val_raw, y_val_raw = X[val_idx], y_raw[val_idx]
    X_test_raw, y_test_raw = X[test_idx], y_raw[test_idx]

    # standardize X
    x_std = Standardizer.fit(X_train_raw)
    X_train = x_std.transform(X_train_raw)
    X_val = x_std.transform(X_val_raw)
    X_test = x_std.transform(X_test_raw)

    # transform y
    if USE_LOG_TARGET:
        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)
        y_test = np.log1p(y_test_raw)
    else:
        y_train, y_val, y_test = y_train_raw.copy(), y_val_raw.copy(), y_test_raw.copy()

    # standardize y (for stable training)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    train_ds = ArrayDataset(X_train, y_train_n)
    val_ds = ArrayDataset(X_val, y_val_n)
    test_ds = ArrayDataset(X_test, y_test_n)

    # IMPORTANT: train loader shuffle=True for training
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # IMPORTANT: eval loaders shuffle=False (fixes your earlier train-metric bug)
    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(in_dim=X.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # val loss
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vlosses.append(float(criterion(pred, yb).item()))
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = float(np.mean(vlosses)) if vlosses else 0.0

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {"model": model.state_dict(), "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {epoch}, best epoch={best_state['epoch']}")
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint saved.")

    model.load_state_dict(best_state["model"])

    def inv_target(y_norm):
        y_fit = y_norm * y_std + y_mean
        if USE_LOG_TARGET:
            y_out = np.expm1(y_fit)
        else:
            y_out = y_fit
        return np.maximum(y_out, 0.0)

    # Evaluate on train/val/test (raw ns)
    pred_train_n, _ = predict_loader(model, train_eval_loader, device)
    pred_val_n, _ = predict_loader(model, val_loader, device)
    pred_test_n, _ = predict_loader(model, test_loader, device)

    y_train_pred = inv_target(pred_train_n)
    y_val_pred = inv_target(pred_val_n)
    y_test_pred = inv_target(pred_test_n)

    metrics = {
        "train": {
            "n": int(len(y_train_raw)),
            "mae_ns": mae_np(y_train_raw, y_train_pred),
            "mape": mape_safe(y_train_raw, y_train_pred, eps=1.0),
            "r2": r2_np(y_train_raw, y_train_pred),
        },
        "val": {
            "n": int(len(y_val_raw)),
            "mae_ns": mae_np(y_val_raw, y_val_pred),
            "mape": mape_safe(y_val_raw, y_val_pred, eps=1.0),
            "r2": r2_np(y_val_raw, y_val_pred),
        },
        "test": {
            "n": int(len(y_test_raw)),
            "mae_ns": mae_np(y_test_raw, y_test_pred),
            "mape": mape_safe(y_test_raw, y_test_pred, eps=1.0),
            "r2": r2_np(y_test_raw, y_test_pred),
        },
    }

    # Save checkpoint (contains scaling params)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
        "hidden_dims": HIDDEN_DIMS,
        "dropout": float(DROPOUT),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "use_log_target": bool(USE_LOG_TARGET),
        "x_mean": x_std.mean.astype(np.float64),
        "x_std": x_std.std.astype(np.float64),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val),
    }
    torch.save(ckpt, CKPT_PATH)

    meta = {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "use_log_target": USE_LOG_TARGET,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "aggregate_duplicates": AGGREGATE_DUPLICATES_BY_FEATURES,
        "best_epoch": int(best_state["epoch"]),
        "best_val_loss": float(best_val),
        "metrics": metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model, metrics

@torch.no_grad()
def predict_df_with_ckpt(df: pd.DataFrame, ckpt_path: str):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    feature_cols = ckpt["feature_cols"]
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float64)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float64)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])
    use_log_target = bool(ckpt["use_log_target"])

    X = df[feature_cols].to_numpy(dtype=np.float64)
    Xn = (X - x_mean) / x_std

    model = MLPRegressor(
        in_dim=int(ckpt["in_dim"]),
        hidden_dims=ckpt["hidden_dims"],
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    xb = torch.tensor(Xn, dtype=torch.float32)
    pred_n = model(xb).cpu().numpy().reshape(-1)

    pred_fit = pred_n * y_std + y_mean
    if use_log_target:
        pred = np.expm1(pred_fit)
    else:
        pred = pred_fit
    return np.maximum(pred, 0.0)


def evaluate_plot_testset_metrics(
    plot_testset_csv: str,
    ckpt_path: str,
    feature_cols: list,
    target_col: str = "query_time_ns",
):
    if not os.path.exists(plot_testset_csv):
        raise FileNotFoundError(f"Plot testset CSV not found: {plot_testset_csv}")

    df = pd.read_csv(plot_testset_csv)

    required_cols = ["plot_type"] + list(set(feature_cols + [target_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in plot testset CSV: {missing}")

    for c in required_cols:
        if c != "plot_type":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in required_cols if c != "plot_type"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows in plot testset after cleaning.")

    df["query_time_ns_pred"] = predict_df_with_ckpt(df, ckpt_path)

    def summarize(sub):
        yt = sub[target_col].to_numpy(dtype=np.float64)
        yp = sub["query_time_ns_pred"].to_numpy(dtype=np.float64)
        return {
            "n": int(len(sub)),
            "mae_ns": mae_np(yt, yp),
            "mape": mape_safe(yt, yp, eps=1.0),
            "r2": r2_np(yt, yp),
        }

    results = {
        "overall": summarize(df),
        "by_plot_type": {}
    }
    for pt, g in df.groupby("plot_type"):
        results["by_plot_type"][str(pt)] = summarize(g)

    print("\n=== Controlled Plot Testset Metrics ===")
    ov = results["overall"]
    print(f"[overall] n={ov['n']} | MAE(ns)={ov['mae_ns']:.3f} | MAPE={ov['mape']:.6f} | R^2={ov['r2']:.6f}")
    for pt in ["num_objects", "query_keyword_count", "result_count"]:
        if pt in results["by_plot_type"]:
            m = results["by_plot_type"][pt]
            print(f"[{pt}] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")

    return results, df


def export_controlled_plot_data_to_xlsx_or_csv(
    plot_testset_csv: str,
    ckpt_path: str,
    feature_cols: list,
    target_col: str = "query_time_ns",
    out_excel_path: str = "Mc_mlp_plot_line_point_data.xlsx",
    out_csv_dir: str = "Mc_mlp_plot_line_point_data_csv",
):
    """
    Export blue points and red line data for 3 plots.
    If openpyxl is missing, fallback to CSV files.
    """
    if not os.path.exists(plot_testset_csv):
        raise FileNotFoundError(f"Plot testset CSV not found: {plot_testset_csv}")

    df = pd.read_csv(plot_testset_csv)

    required_cols = ["plot_type"] + list(set(feature_cols + [target_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in plot testset CSV: {missing}")

    for c in required_cols:
        if c != "plot_type":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in required_cols if c != "plot_type"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows in plot testset after cleaning.")

    df["query_time_ns_pred"] = predict_df_with_ckpt(df, ckpt_path)

    def build_points_and_line(df_sub: pd.DataFrame, x_col: str):
        d_points = df_sub[[x_col, target_col, "query_time_ns_pred"]].copy()
        d_points = d_points.sort_values(by=x_col, kind="mergesort").reset_index(drop=True)
        d_points = d_points.rename(columns={
            target_col: "query_time_ns_true",
            "query_time_ns_pred": "query_time_ns_pred_point"
        })

        d_line = (
            d_points.groupby(x_col, as_index=False)["query_time_ns_pred_point"]
            .mean()
            .rename(columns={"query_time_ns_pred_point": "query_time_ns_pred_line_mean"})
            .sort_values(by=x_col, kind="mergesort")
            .reset_index(drop=True)
        )
        d_true_line = (
            d_points.groupby(x_col, as_index=False)["query_time_ns_true"]
            .mean()
            .rename(columns={"query_time_ns_true": "query_time_ns_true_line_mean"})
            .sort_values(by=x_col, kind="mergesort")
            .reset_index(drop=True)
        )
        d_line = d_line.merge(d_true_line, on=x_col, how="left")
        return d_points, d_line

    df_n  = df[df["plot_type"] == "num_objects"].copy().reset_index(drop=True)
    df_qk = df[df["plot_type"] == "query_keyword_count"].copy().reset_index(drop=True)
    df_rc = df[df["plot_type"] == "result_count"].copy().reset_index(drop=True)

    n_points, n_line = build_points_and_line(df_n, "num_objects")
    qk_points, qk_line = build_points_and_line(df_qk, "query_keyword_count")
    rc_points, rc_line = build_points_and_line(df_rc, "result_count")

    # Try xlsx
    if EXPORT_XLSX:
        try:
            import openpyxl  # noqa: F401
            with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
                n_points.to_excel(writer, sheet_name="num_objects_points", index=False)
                n_line.to_excel(writer, sheet_name="num_objects_line", index=False)

                qk_points.to_excel(writer, sheet_name="qk_points", index=False)
                qk_line.to_excel(writer, sheet_name="qk_line", index=False)

                rc_points.to_excel(writer, sheet_name="result_points", index=False)
                rc_line.to_excel(writer, sheet_name="result_line", index=False)

            print(f"\nSaved point/line data to XLSX: {out_excel_path}")
            return
        except Exception as e:
            print("\n[WARN] XLSX export failed (openpyxl missing or other issue). Falling back to CSV.")
            print("Reason:", str(e))

    # Fallback to CSV
    os.makedirs(out_csv_dir, exist_ok=True)
    n_points.to_csv(os.path.join(out_csv_dir, "num_objects_points.csv"), index=False)
    n_line.to_csv(os.path.join(out_csv_dir, "num_objects_line.csv"), index=False)

    qk_points.to_csv(os.path.join(out_csv_dir, "qk_points.csv"), index=False)
    qk_line.to_csv(os.path.join(out_csv_dir, "qk_line.csv"), index=False)

    rc_points.to_csv(os.path.join(out_csv_dir, "result_points.csv"), index=False)
    rc_line.to_csv(os.path.join(out_csv_dir, "result_line.csv"), index=False)

    print(f"\nSaved point/line data to CSV dir: {out_csv_dir}")

from sklearn.linear_model import LinearRegression
import numpy as np

def fit_theoretical_cost_model_positive(df, target_col="query_time_ns", use_split=True, seed=42):
    # X = [qk*N, N, R], y = time
    N = df["num_objects"].to_numpy(dtype=np.float64)
    qk = df["query_keyword_count"].to_numpy(dtype=np.float64)
    R = df["result_count"].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    X = np.stack([qk * N, N, R], axis=1)

    def metric_pack(y_true, y_pred):
        return {
            "n": int(len(y_true)),
            "mae_ns": mae_np(y_true, y_pred),
            "mape": mape_safe(y_true, y_pred, eps=1.0),
            "r2": r2_np(y_true, y_pred),
        }

    if use_split:
        n = len(y)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        tr = idx[:n_train]
        va = idx[n_train:n_train + n_val]
        te = idx[n_train + n_val:]

        # positive=True => coef >= 0（对 alpha,beta,gamma）
        reg = LinearRegression(positive=True, fit_intercept=True)
        reg.fit(X[tr], y[tr])

        yhat_tr = reg.predict(X[tr])
        yhat_va = reg.predict(X[va])
        yhat_te = reg.predict(X[te])

        metrics = {
            "train": metric_pack(y[tr], yhat_tr),
            "val": metric_pack(y[va], yhat_va),
            "test": metric_pack(y[te], yhat_te),
        }
    else:
        reg = LinearRegression(positive=True, fit_intercept=True)
        reg.fit(X, y)
        yhat = reg.predict(X)
        metrics = {"overall": metric_pack(y, yhat)}

    epsilon = float(reg.intercept_)
    alpha, beta, gamma = [float(v) for v in reg.coef_]

    params = {"epsilon": epsilon, "alpha": alpha, "beta": beta, "gamma": gamma}

    print("\n=== Theoretical Cost Model Fit (positive coef) ===")
    print("Model: y = epsilon + alpha*(qk*N) + beta*N + gamma*R")
    print(f"epsilon = {params['epsilon']:.6f}")
    print(f"alpha   = {params['alpha']:.12e}  (>=0)")
    print(f"beta    = {params['beta']:.12e}  (>=0)")
    print(f"gamma   = {params['gamma']:.12e}  (>=0)")

    if "overall" in metrics:
        m = metrics["overall"]
        print(f"[overall] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")
    else:
        for split in ["train", "val", "test"]:
            m = metrics[split]
            print(f"[{split}] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")

    return params, metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict_theory(params, N, qk, R):
    # y = eps + alpha*(qk*N) + beta*N + gamma*R
    return (params["epsilon"]
            + params["alpha"] * (qk * N)
            + params["beta"] * N
            + params["gamma"] * R)

def plot_theory_vs_true_from_controlled_csv(
    plot_csv="bitmap_plot_testset_controlled.csv",
    params=None,
    out_prefix="theory_fit",
    default_N=100000,
    default_qk=4,
    default_rc=10,
):
    if params is None:
        raise ValueError("params is None")

    df = pd.read_csv(plot_csv)

    # --- 1) x = num_objects (fixed qk=4, rc=10) ---
    d1 = df[df["plot_type"] == "num_objects"].copy()
    d1 = d1[(d1["query_keyword_count"] == default_qk) & (d1["result_count"] == default_rc)]
    d1 = d1.sort_values("num_objects").reset_index(drop=True)
    d1["pred"] = predict_theory(params, d1["num_objects"].to_numpy(), default_qk, default_rc)

    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(d1["num_objects"], d1["query_time_ns"], s=18, alpha=0.85, label="Ground Truth")
    plt.plot(d1["num_objects"], d1["pred"], linewidth=2.0, label="Analytical Model")
    plt.xlabel("num_objects")
    plt.ylabel("query_time_ns")
    plt.title("Analytical model vs ground truth (qk=4, result_count=10)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_num_objects.png", dpi=200)
    plt.close()

    # --- 2) x = query_keyword_count (fixed N=100000, rc=10) ---
    d2 = df[df["plot_type"] == "query_keyword_count"].copy()
    d2 = d2[(d2["num_objects"] == default_N) & (d2["result_count"] == default_rc)]
    d2 = d2.sort_values("query_keyword_count").reset_index(drop=True)

    d2["pred_point"] = predict_theory(params, default_N, d2["query_keyword_count"].to_numpy(), default_rc)
    d2_line = d2.groupby("query_keyword_count", as_index=False)["pred_point"].mean()

    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(d2["query_keyword_count"], d2["query_time_ns"], s=18, alpha=0.85, label="Ground Truth")
    plt.plot(d2_line["query_keyword_count"], d2_line["pred_point"], linewidth=2.0, label="Analytical Model")
    plt.xlabel("query_keyword_count")
    plt.ylabel("query_time_ns")
    plt.title("Analytical model vs ground truth (N=100000, result_count=10)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_query_keyword_count.png", dpi=200)
    plt.close()

    # --- 3) x = result_count (fixed N=100000, qk=4; rc in [0,200]) ---
    d3 = df[df["plot_type"] == "result_count"].copy()
    d3 = d3[(d3["num_objects"] == default_N) & (d3["query_keyword_count"] == default_qk)]
    d3 = d3.sort_values("result_count").reset_index(drop=True)
    d3["pred"] = predict_theory(params, default_N, default_qk, d3["result_count"].to_numpy())

    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(d3["result_count"], d3["query_time_ns"], s=18, alpha=0.85, label="Ground Truth")
    plt.plot(d3["result_count"], d3["pred"], linewidth=2.0, label="Analytical Model")
    plt.xlabel("result_count")
    plt.ylabel("query_time_ns")
    plt.title("Analytical model vs ground truth (N=100000, qk=4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_result_count.png", dpi=200)
    plt.close()

    print("\nSaved theory-vs-true plots:")
    print(f" - {out_prefix}_num_objects.png")
    print(f" - {out_prefix}_query_keyword_count.png")
    print(f" - {out_prefix}_result_count.png")

from sklearn.linear_model import LinearRegression
import numpy as np

def fit_theoretical_cost_model_all_positive(df, target_col="query_time_ns", use_split=True, seed=42):
    """
    Fit with non-negativity constraints for ALL params:
      y = epsilon + alpha*(qk*N) + beta*N + gamma*R
    Constraints:
      epsilon >= 0, alpha >= 0, beta >= 0, gamma >= 0

    Implementation:
      Build X = [1, qk*N, N, R] and fit LinearRegression(positive=True, fit_intercept=False).
    """
    required = ["num_objects", "query_keyword_count", "result_count", target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for theoretical model fitting: {missing}")

    N = df["num_objects"].to_numpy(dtype=np.float64)
    qk = df["query_keyword_count"].to_numpy(dtype=np.float64)
    R = df["result_count"].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)

    # Design matrix with constant column included (epsilon becomes a non-negative coefficient)
    X = np.stack([np.ones_like(y), qk * N, N, R], axis=1)

    def metric_pack(y_true, y_pred):
        return {
            "n": int(len(y_true)),
            "mae_ns": mae_np(y_true, y_pred),
            "mape": mape_safe(y_true, y_pred, eps=1.0),
            "r2": r2_np(y_true, y_pred),
        }

    # Helper fit+predict
    def fit_reg(Xa, ya):
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(Xa, ya)
        return reg

    if use_split:
        n = len(y)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        tr = idx[:n_train]
        va = idx[n_train:n_train + n_val]
        te = idx[n_train + n_val:]

        reg = fit_reg(X[tr], y[tr])
        yhat_tr = reg.predict(X[tr])
        yhat_va = reg.predict(X[va])
        yhat_te = reg.predict(X[te])

        metrics = {
            "train": metric_pack(y[tr], yhat_tr),
            "val": metric_pack(y[va], yhat_va),
            "test": metric_pack(y[te], yhat_te),
        }
    else:
        reg = fit_reg(X, y)
        yhat = reg.predict(X)
        metrics = {"overall": metric_pack(y, yhat)}

    # Coefs correspond to [epsilon, alpha, beta, gamma]
    theta = reg.coef_.astype(np.float64)
    params = {
        "epsilon": float(theta[0]),
        "alpha": float(theta[1]),
        "beta": float(theta[2]),
        "gamma": float(theta[3]),
    }

    print("\n=== Theoretical Cost Model Fit (ALL params >= 0) ===")
    print("Model: y = epsilon + alpha*(qk*N) + beta*N + gamma*R")
    print("Constraint: epsilon, alpha, beta, gamma >= 0")
    print(f"epsilon = {params['epsilon']:.6f}  (>=0)")
    print(f"alpha   = {params['alpha']:.12e}  (>=0)")
    print(f"beta    = {params['beta']:.12e}  (>=0)")
    print(f"gamma   = {params['gamma']:.12e}  (>=0)")

    if "overall" in metrics:
        m = metrics["overall"]
        print(f"[overall] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")
    else:
        for split in ["train", "val", "test"]:
            m = metrics[split]
            print(f"[{split}] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")

    return params, metrics

def main():
    set_seed(SEED)

    df = load_and_prepare(
        CSV_TRAIN,
        FEATURE_COLS,
        TARGET_COL,
        aggregate_duplicates=AGGREGATE_DUPLICATES_BY_FEATURES,
    )

    print("Loaded train rows:", len(df))
    print("Feature columns:", FEATURE_COLS)
    print("Target column:", TARGET_COL)
    print(df.head())

    _, metrics = train_mlp(df)

    print("\n=== MLP Evaluation ===")
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(f"[{split}] n={m['n']} | MAE(ns)={m['mae_ns']:.3f} | MAPE={m['mape']:.6f} | R^2={m['r2']:.6f}")

    # Evaluate controlled plot testset
    evaluate_plot_testset_metrics(
        plot_testset_csv=CSV_PLOT,
        ckpt_path=CKPT_PATH,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    # Export point/line data for Excel plotting (no charts generated here)
    export_controlled_plot_data_to_xlsx_or_csv(
        plot_testset_csv=CSV_PLOT,
        ckpt_path=CKPT_PATH,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        out_excel_path=EXPORT_XLSX_PATH,
        out_csv_dir=EXPORT_CSV_DIR,
    )

    params_th, metrics_th = fit_theoretical_cost_model_all_positive(
        df=df,
        target_col=TARGET_COL,
        use_split=True,
        seed=SEED,
    )
    
    plot_theory_vs_true_from_controlled_csv(
        plot_csv=CSV_PLOT,
        params=params_th,
        out_prefix="theory_fit_pos",
        default_N=100000,
        default_qk=4,
        default_rc=10,
    )



if __name__ == "__main__":
    main()