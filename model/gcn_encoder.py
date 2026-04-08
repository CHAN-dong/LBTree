# model/gcn_encoder.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def csr_to_torch_sparse(X_csr) -> torch.Tensor:
    """
    Convert scipy csr_matrix to torch sparse COO tensor (float32).
    """
    # X_csr: scipy.sparse.csr_matrix
    X_csr = X_csr.tocoo()
    indices = np.vstack([X_csr.row, X_csr.col]).astype(np.int64)
    values = X_csr.data.astype(np.float32)
    i = torch.from_numpy(indices)
    v = torch.from_numpy(values)
    return torch.sparse_coo_tensor(i, v, size=X_csr.shape).coalesce()


def build_normalized_adj(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Build normalized adjacency:  Â = D^{-1/2} (A + I) D^{-1/2}
    where A is weighted adjacency from edge_index/edge_weight.

    Returns a torch sparse COO tensor of shape [N, N].
    """
    # edge_index: (2, E), edge_weight: (E,)
    src = torch.from_numpy(edge_index[0].astype(np.int64))
    dst = torch.from_numpy(edge_index[1].astype(np.int64))
    w = torch.from_numpy(edge_weight.astype(np.float32))

    if add_self_loops:
        self_idx = torch.arange(num_nodes, dtype=torch.int64)
        src = torch.cat([src, self_idx], dim=0)
        dst = torch.cat([dst, self_idx], dim=0)
        w = torch.cat([w, torch.ones(num_nodes, dtype=torch.float32)], dim=0)

    # Sparse adjacency A (COO)
    indices = torch.stack([src, dst], dim=0)  # [2, E']
    A = torch.sparse_coo_tensor(indices, w, size=(num_nodes, num_nodes)).coalesce()

    # Degree: deg[i] = sum_j A[i, j]
    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize values: w_ij * deg_i^-0.5 * deg_j^-0.5
    idx = A.indices()
    vals = A.values()
    norm_vals = vals * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]

    A_norm = torch.sparse_coo_tensor(idx, norm_vals, size=A.shape).coalesce()
    return A_norm


class WeightedGCNEncoder(nn.Module):
    """
    A simple weighted GCN encoder:
      H0 = X_sparse @ W0
      H_{l+1} = ReLU(Â @ H_l @ W_l)  (last layer optionally no ReLU)
    """
    def __init__(
        self,
        vocab_dim: int,      # d (keywords)
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # X_sparse @ W0 : [N,d] @ [d,hidden] -> [N,hidden]
        self.W0 = nn.Parameter(torch.empty(vocab_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W0)

        # GCN layers: linear transforms after message passing
        self.linears = nn.ModuleList()
        if num_layers == 1:
            self.linears.append(nn.Linear(hidden_dim, out_dim, bias=True))
        else:
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            self.linears.append(nn.Linear(hidden_dim, out_dim, bias=True))

    def forward(self, X_sparse: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # X_sparse: torch sparse [N,d], A_norm: torch sparse [N,N]
        H = torch.sparse.mm(X_sparse, self.W0)  # [N,hidden]
        H = F.relu(H)

        for li, lin in enumerate(self.linears):
            H = F.dropout(H, p=self.dropout, training=self.training)
            H = torch.sparse.mm(A_norm, H)       # message passing
            H = lin(H)                           # transform
            if li != len(self.linears) - 1:
                H = F.relu(H)

        return H  # [N,out_dim]


def gcn_encode_graph(
    g: Dict[str, Any],
    hidden_dim: int = 64,
    out_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience function:
      input: g from build_cooccurrence_graph()
      output: embeddings [N,out_dim]
    """
    X = g["X"]  # usually scipy csr
    edge_index = g["edge_index"]
    edge_weight = g["edge_weight"]

    if hasattr(X, "shape"):
        num_nodes, vocab_dim = X.shape
    else:
        raise ValueError("scipy csr_matrix")

    X_sparse = csr_to_torch_sparse(X)

    A_norm = build_normalized_adj(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        add_self_loops=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_sparse = X_sparse.to(device)
    A_norm = A_norm.to(device)

    model = WeightedGCNEncoder(
        vocab_dim=vocab_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.eval()
    with torch.no_grad():
        emb = model(X_sparse, A_norm)  # [N,out_dim]
    return emb
