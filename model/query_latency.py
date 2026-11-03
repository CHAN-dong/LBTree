"""Offline-trained three-input Mq(n, query length, result size), Eq. (2)."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class QueryLatencyMLP(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("scale", torch.ones(3))

    def forward(self, n, u, r):
        n, u, r = torch.broadcast_tensors(n, u, r)
        features = torch.stack((n, u, r), dim=-1)
        features = (torch.log1p(features.clamp_min(0)) - self.mean) / self.scale
        log_time = torch.nn.functional.softplus(self.layers(features).squeeze(-1))
        return torch.expm1(log_time)


def load_query_model(path):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    if data.get("feature_names") != ["num_objects", "query_keyword_count", "result_count"]:
        raise ValueError("Mq checkpoint must use exactly (n, u, |Rq|), in this order")
    model = QueryLatencyMLP(data["width"])
    model.load_state_dict(data["state_dict"])
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def predict(model, n, u, r):
    tensors = [v for v in (n, u, r) if isinstance(v, torch.Tensor)]
    device = tensors[0].device if tensors else torch.device("cpu")
    dtype = tensors[0].dtype if tensors else torch.float32
    # Frozen parameters still allow gradients through the three inputs.
    model = model.to(device=device, dtype=dtype)
    values = [torch.as_tensor(v, dtype=dtype, device=device) for v in (n, u, r)]
    result = model(*values)
    if tensors: return result
    array = result.detach().cpu().numpy()
    return float(array) if array.ndim == 0 else array
