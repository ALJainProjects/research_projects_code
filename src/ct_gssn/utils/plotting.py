# src/ct_gssn/utils/plotting.py
"""
Plotting helpers (Plotly) for CT-GSSN.

- plot_series_plotly:           single 1D series (pred vs true)
- plot_multinode_series:        small-multiple lines by node index
- plot_adjacency_heatmap:       visualize (static) adjacency
- plot_dynamic_adj_heatmap:     show (L, N, N) as a time-indexed heatmap via frames
- plot_stability_spectrum:      histogram + violin of eigenvalues Re(λ) from symmetric(A)
- plot_deltas_hist:             distribution of irregular Δt
- plot_deltas_series:           Δt timeline
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import torch


# ----------------------------- utils ----------------------------- #

def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).squeeze()
    assert x.ndim == 1, f"Expected 1D, got {x.shape}"
    return x


# ----------------------------- basic plots ----------------------------- #

def plot_series_plotly(pred_1d: torch.Tensor, true_1d: torch.Tensor) -> go.Figure:
    """
    Plot a single predicted series vs. ground truth.

    Args:
        pred_1d, true_1d: (L,) tensors or arrays

    Returns:
        plotly.graph_objects.Figure
    """
    pred = _ensure_1d(_to_np(pred_1d))
    true = _ensure_1d(_to_np(true_1d))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true, mode="lines", name="Ground Truth"))
    fig.add_trace(go.Scatter(y=pred, mode="lines", name="Prediction"))
    fig.update_layout(title="CT-GSSN Forecast", xaxis_title="t", yaxis_title="value")
    return fig


def plot_multinode_series(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    nodes: Optional[Sequence[int]] = None,
    max_nodes: int = 6,
    title: str = "CT-GSSN Forecasts (subset of nodes)",
) -> go.Figure:
    """
    Plot several nodes' series as small multiples.

    Args:
        y_pred: (B,N,L,1) or (N,L,1) or (L,1)
        y_true: same shape as y_pred
        mask:   (B,N,L) or None (used only for styling dots on observed points)
        nodes:  iterable of node indices to show; if None, use range(min(N, max_nodes))
        max_nodes: cap the number of panels
    """
    # Squeeze batch/channel if present
    yp = _to_np(y_pred)
    yt = _to_np(y_true)
    if yp.ndim == 4:
        yp = yp[0, :, :, 0]  # (N,L)
        yt = yt[0, :, :, 0]
        m = _to_np(mask[0]) if mask is not None else None
    elif yp.ndim == 3:
        yp = yp[:, :, 0]
        yt = yt[:, :, 0]
        m = _to_np(mask) if mask is not None else None
    elif yp.ndim == 2:  # (L,1)
        yp = yp[:, 0][None, :]
        yt = yt[:, 0][None, :]
        m = None
    else:
        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

    N, L = yp.shape
    if nodes is None:
        nodes = list(range(min(N, max_nodes)))
    else:
        nodes = list(nodes)[:max_nodes]

    fig = go.Figure()
    for i, n in enumerate(nodes):
        offset = i * (yt[n].max() - yt[n].min() + 1e-6) * 0.2  # slight baseline shift to reduce overlap
        fig.add_trace(go.Scatter(y=yt[n] + 0 * offset, mode="lines", name=f"y_true[n={n}]",
                                 legendgroup=f"n{n}", hovertemplate="t=%{x}, y=%{y}"))
        fig.add_trace(go.Scatter(y=yp[n] + 0 * offset, mode="lines", name=f"y_pred[n={n}]",
                                 legendgroup=f"n{n}", hovertemplate="t=%{x}, y=%{y}",
                                 line=dict(dash="dash")))
        if m is not None:
            obs_idx = np.where(m[n] > 0.5)[0]
            if len(obs_idx) > 0:
                fig.add_trace(go.Scatter(x=obs_idx, y=yt[n, obs_idx] + 0 * offset,
                                         mode="markers", name=f"obs[n={n}]",
                                         legendgroup=f"n{n}", showlegend=False,
                                         marker=dict(size=4, symbol="x")))
    fig.update_layout(title=title, xaxis_title="t", yaxis_title="value")
    return fig


# ----------------------------- adjacency ----------------------------- #

def plot_adjacency_heatmap(adj: torch.Tensor, title: str = "Adjacency") -> go.Figure:
    """
    Heatmap of (N,N) adjacency.
    """
    A = _to_np(adj)
    if A.ndim != 2:
        raise ValueError(f"Expected (N,N), got {A.shape}")
    fig = go.Figure(data=go.Heatmap(z=A))
    fig.update_layout(title=title, xaxis_title="j", yaxis_title="i")
    return fig


def plot_dynamic_adj_heatmap(adj_seq: torch.Tensor, title: str = "Dynamic Adjacency (time-indexed)") -> go.Figure:
    """
    Animate (L,N,N) adjacency over time using frames.
    """
    A = _to_np(adj_seq)
    if A.ndim != 3:
        raise ValueError(f"Expected (L,N,N), got {A.shape}")
    L = A.shape[0]
    fig = go.Figure(
        data=[go.Heatmap(z=A[0])],
        frames=[go.Frame(data=[go.Heatmap(z=A[t])], name=str(t)) for t in range(L)],
    )
    fig.update_layout(
        title=title,
        xaxis_title="j",
        yaxis_title="i",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "steps": [{"method": "animate", "args": [[str(t)], {"mode": "immediate"}], "label": str(t)} for t in range(L)]
        }],
    )
    return fig


# ----------------------------- stability / A ----------------------------- #

def plot_stability_spectrum(A_seq: torch.Tensor, title: str = "Stability spectrum: eig(Re(S)) per time/layer") -> go.Figure:
    """
    Visualize eigenvalues of the symmetric part S = 0.5*(A + A^T).

    Args:
        A_seq: (L_layers, B, L, N, D, D) or (B, L, N, D, D) or (N, D, D)
    """
    A = _to_np(A_seq)

    # Squeeze generic shapes down to a list of eigenvalues
    if A.ndim == 6:              # (L_layers,B,L,N,D,D)
        A = A.reshape(-1, A.shape[-2], A.shape[-1])  # (K, D, D)
    elif A.ndim == 5:            # (B,L,N,D,D)
        A = A.reshape(-1, A.shape[-2], A.shape[-1])
    elif A.ndim == 3:            # (N,D,D) or (D,D)
        if A.shape[0] != A.shape[1]:  # (D,D)
            A = A[None, ...]
    else:
        raise ValueError(f"Unsupported A_seq shape: {A_seq.shape}")

    eigvals = []
    for k in range(A.shape[0]):
        S = 0.5 * (A[k] + A[k].T)
        # Eigenvalues of symmetric are real
        evals = np.linalg.eigvalsh(S).astype(np.float32)
        eigvals.append(evals)
    eigvals = np.stack(eigvals, axis=0)  # (K, D)

    fig = go.Figure()
    # Violin for distribution across all time/layers/batch
    fig.add_trace(go.Violin(y=eigvals.flatten(), name="eig(S)", box_visible=True, meanline_visible=True))
    # Histogram for a different perspective
    fig.add_trace(go.Histogram(y=eigvals.flatten(), name="hist", opacity=0.5))
    fig.update_layout(barmode="overlay", title=title, xaxis_title="", yaxis_title="Eigenvalue")
    return fig


# ----------------------------- delta diagnostics ----------------------------- #

def plot_deltas_hist(deltas: torch.Tensor, title: str = "Δt distribution") -> go.Figure:
    """
    Histogram of Δt. Accepts (L,) or (B,L).
    """
    d = _to_np(deltas)
    d = d.reshape(-1)
    fig = go.Figure(data=[go.Histogram(x=d, nbinsx=min(60, max(10, int(len(d) ** 0.5))))])
    fig.update_layout(title=title, xaxis_title="Δt", yaxis_title="count")
    return fig


def plot_deltas_series(deltas: torch.Tensor, title: str = "Δt over time") -> go.Figure:
    """
    Line plot of Δt over time. If (B,L), averages over batch and shades +/- std.
    """
    d = _to_np(deltas)
    if d.ndim == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=d, mode="lines+markers", name="Δt"))
        fig.update_layout(title=title, xaxis_title="t", yaxis_title="Δt")
        return fig
    elif d.ndim == 2:
        mu = d.mean(axis=0)
        sd = d.std(axis=0)
        t = np.arange(d.shape[1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=mu, mode="lines", name="mean Δt"))
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),
                                 y=np.concatenate([mu - sd, (mu + sd)[::-1]]),
                                 fill="toself", mode="lines", name="±1 std", opacity=0.2,
                                 line=dict(width=0)))
        fig.update_layout(title=title, xaxis_title="t", yaxis_title="Δt")
        return fig
    else:
        raise ValueError(f"Unexpected deltas shape: {deltas.shape}")
