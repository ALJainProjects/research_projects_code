# src/ct_gssn/data/utils_data.py
"""
Shared data utilities for CT-GSSN loaders.
"""
from __future__ import annotations
from typing import Optional
import torch


def to_float32(x):
    return x.float() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)


def build_knn_adjacency(x_node_feats: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Build kNN adjacency from node features and row-normalize.

    Args
    ----
    x_node_feats : (N, d) tensor
    k : int
        Number of neighbors.

    Returns
    -------
    adj : (N, N) torch.float32
        Row-normalized, zero-diagonal, symmetric kNN graph.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    arr = x_node_feats.detach().cpu().numpy()
    N = arr.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(arr)
    graph = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        _, idx = nbrs.kneighbors(arr[i : i + 1])
        for j in idx[0]:
            if j != i:
                graph[i, j] = 1.0
    graph = ((graph + graph.T) > 0).float()
    deg = graph.sum(dim=1, keepdim=True).clamp_min(1.0)
    graph = graph / deg
    return graph


def constant_deltas(L: int, dt: float = 1.0) -> torch.Tensor:
    """
    Create a constant step-size vector.

    Parameters
    ----------
    L : int
        Sequence length
    dt : float
        Constant step size

    Returns
    -------
    deltas : (L,) torch.float32
    """
    return torch.ones(L, dtype=torch.float32) * dt


def zscore_(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    In-place-ish z-score normalization for x.

    Args
    ----
    x : (N, L, P)
    mask : (N, L), optional
        If provided, compute stats over observed positions only.

    Returns
    -------
    x_norm : (N, L, P)
    """
    if mask is None:
        mu = x.mean()
        sigma = x.std().clamp_min(eps)
        return (x - mu) / sigma
    obs = (mask.unsqueeze(-1) > 0).expand_as(x)
    if obs.any():
        mu = x[obs].mean()
        sigma = x[obs].std().clamp_min(eps)
        return (x - mu) / sigma
    # If no observations, return x unchanged
    return x


def simple_impute_(x: torch.Tensor, mask: torch.Tensor, mode: str = "ffill") -> torch.Tensor:
    """
    Simple imputation for missing x values where mask==0.

    Args
    ----
    x : (N, L, P)
    mask : (N, L)
    mode : {"ffill","zero"}

    Returns
    -------
    x_imp : (N, L, P)
    """
    if mode not in ("ffill", "zero"):
        return x
    N, L, P = x.shape
    x_imp = x.clone()
    obs = mask > 0
    if mode == "zero":
        x_imp[~obs.unsqueeze(-1).expand(N, L, P)] = 0.0
        return x_imp

    # forward-fill along time per node and channel
    for n in range(N):
        last = None
        for t in range(L):
            if obs[n, t]:
                last = x_imp[n, t].clone()
            else:
                if last is not None:
                    x_imp[n, t] = last
                else:
                    x_imp[n, t] = 0.0  # fallback at prefix
    return x_imp
