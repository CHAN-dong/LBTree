# model/cluster_mlp.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClusterMLP(nn.Module):
    """
    emb: [N, d]
    outputs: logit [N] -> pi = sigmoid(logit / tau)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze(1)  # [N]

class ClusterMLP(nn.Module):
    """
    emb: [N, d]
    logits: [N, C]
    probs: [N, C] via softmax
    """
    def __init__(
        self,
        in_dim: int,
        num_clusters: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        self.in_dim = in_dim
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, num_clusters))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_clusters))

        self.layers = nn.ModuleList(layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits

def cluster_probabilities(
    emb: torch.Tensor,
    num_clusters: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    if device is None:
        device = emb.device.type if hasattr(emb, "device") else "cpu"

    emb = emb.to(device)
    model = ClusterMLP(
        in_dim=emb.shape[1],
        num_clusters=num_clusters,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(emb)
        probs = torch.softmax(logits, dim=1)
    return probs
