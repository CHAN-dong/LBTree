"""Fit Eq. (2) to measured bitmap latency samples, never to analytic labels."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from model.query_latency import QueryLatencyMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    with args.samples.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 10: raise ValueError("At least ten measured samples are required")
    names = ["num_objects", "query_keyword_count", "result_count"]
    x = torch.tensor([[float(row[k]) for k in names] for row in rows])
    y = torch.tensor([float(row["query_time_ns"]) for row in rows])
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (x < 0).any() or (y <= 0).any():
        raise ValueError("Samples must contain finite, nonnegative features and positive measured times")
    order = torch.randperm(len(rows)); cut = max(1, int(.8 * len(rows)))
    train, validation = order[:cut], order[cut:]
    model = QueryLatencyMLP()
    with torch.no_grad():
        logx = torch.log1p(x[train])
        model.mean.copy_(logx.mean(0)); model.scale.copy_(logx.std(0).clamp_min(1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(args.epochs):
        for batch in train.split(1024):
            optimizer.zero_grad(); estimate = model(*x[batch].T)
            loss = ((torch.log1p(estimate) - torch.log1p(y[batch])) ** 2).mean()
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        error = (model(*x[validation].T) - y[validation]).abs().mean().item()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": 32, "feature_names": names, "state_dict": model.state_dict(),
                "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                "seed": args.seed, "validation_mae_ns": error,
                "train_count": len(train), "validation_count": len(validation)}, args.checkpoint)
    print(f"Mq saved; held-out MAE={error:.3f} ns; samples={len(rows)}")


if __name__ == "__main__": main()
