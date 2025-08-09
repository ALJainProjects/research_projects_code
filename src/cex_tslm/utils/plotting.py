"""
Plotting helpers for CEX-TSLM.

Includes:
- plot_forecast: single-panel forecast plot for selected dimensions
- plot_multi_forecast_grid: grid of subplots across multiple dims
- plot_attention_heatmap: time (queries) × documents attention visualization
- plot_doc_attention_bars: top-k document attention bars at a specific time index
- save_plotly: convenience saver to HTML/PNG (PNG requires kaleido)

All functions accept torch tensors and will handle device/cpu moves.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Union
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Forecast plots
# ---------------------------------------------------------------------------

def plot_forecast(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    dims: Optional[Sequence[int]] = None,
    title: str = "CEX-TSLM Forecast",
) -> go.Figure:
    """
    Plot forecast vs ground truth for one or more dimensions in a single panel.

    Args:
        pred:   (H, d) forecast tensor/array
        target: (H, d) ground truth tensor/array
        dims:   which dimensions to plot; default [0]
        title:  figure title
    Returns:
        plotly.graph_objects.Figure
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    assert p.shape[:1] == t.shape[:1], f"Mismatch in horizon length: {p.shape} vs {t.shape}"

    H = p.shape[0]
    d = p.shape[1] if p.ndim == 2 else 1
    if d == 1:
        dims = [0]
        p = p.reshape(H, 1)
        t = t.reshape(H, 1)
    elif dims is None:
        dims = [0]

    fig = go.Figure()
    for i in dims:
        if i < 0 or i >= p.shape[1]:
            continue
        fig.add_trace(go.Scatter(y=t[:, i], mode="lines", name=f"GT dim {i}"))
        fig.add_trace(go.Scatter(y=p[:, i], mode="lines", name=f"Pred dim {i}"))

    fig.update_layout(
        title=title,
        xaxis_title="t",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_multi_forecast_grid(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    dims: Optional[Sequence[int]] = None,
    cols: int = 3,
    title: str = "CEX-TSLM Forecast (Grid)",
) -> go.Figure:
    """
    Grid of subplots for multiple dimensions.

    Args:
        pred, target: (H, d)
        dims:         which dims to show; default all dims (up to 12 for sanity)
        cols:         number of columns in the grid
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)
    H, d = p.shape[0], (p.shape[1] if p.ndim == 2 else 1)
    if d == 1:
        dims = [0]
        p = p.reshape(H, 1)
        t = t.reshape(H, 1)
    else:
        if dims is None:
            dims = list(range(min(d, 12)))

    rows = int(np.ceil(len(dims) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"dim {i}" for i in dims])
    r = c = 1
    for i in dims:
        fig.add_trace(go.Scatter(y=t[:, i], mode="lines", name=f"GT {i}", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(y=p[:, i], mode="lines", name=f"Pred {i}", showlegend=False), row=r, col=c)
        c += 1
        if c > cols:
            c = 1
            r += 1

    fig.update_layout(height=250 * rows, title_text=title)
    return fig


# ---------------------------------------------------------------------------
# Attention visualizations
# ---------------------------------------------------------------------------

def _head_reduce(attn_logits: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """
    Reduce across heads. attn_logits: (B, H, Lt, M) -> (B, Lt, M)
    """
    if reduce == "mean":
        return attn_logits.mean(axis=1)
    elif reduce == "max":
        return attn_logits.max(axis=1)
    else:
        raise ValueError("reduce must be 'mean' or 'max'")


def plot_attention_heatmap(
    attn_logits: Union[torch.Tensor, np.ndarray],
    doc_labels: Optional[List[str]] = None,
    time_labels: Optional[List[str]] = None,
    head_reduce: str = "mean",
    batch_index: int = 0,
    title: str = "Cross-Modal Attention (Time × Docs)",
) -> go.Figure:
    """
    Heatmap of attention over documents for each time query.

    Args:
        attn_logits: (B, H, Lt, M) pre-softmax logits
        doc_labels:  optional labels for documents (len=M)
        time_labels: optional labels for queries (len=Lt)
        head_reduce: 'mean' or 'max'
        batch_index: which batch item to plot
    """
    logits = _to_numpy(attn_logits)
    assert logits.ndim == 4, "attn_logits must be (B,H,Lt,M)"
    logits = logits[batch_index : batch_index + 1]  # (1,H,Lt,M)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    probs = _head_reduce(probs, head_reduce)[0]  # (Lt,M)

    Lt, M = probs.shape
    xlabels = doc_labels if (doc_labels is not None and len(doc_labels) == M) else list(range(M))
    ylabels = time_labels if (time_labels is not None and len(time_labels) == Lt) else list(range(Lt))

    fig = go.Figure(
        data=go.Heatmap(
            z=probs,
            x=xlabels,
            y=ylabels,
            coloraxis="coloraxis",
            hovertemplate="t=%{y}<br>doc=%{x}<br>p=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Documents (M)",
        yaxis_title="Time queries (Lt)",
        coloraxis={"colorscale": "Viridis"},
        height=max(400, 20 * Lt),
    )
    return fig


def plot_doc_attention_bars(
    attn_logits: Union[torch.Tensor, np.ndarray],
    t_index: int = 0,
    doc_labels: Optional[List[str]] = None,
    head_reduce: str = "mean",
    batch_index: int = 0,
    top_k: int = 12,
    title: str = "Top-k Document Attention at t",
) -> go.Figure:
    """
    Bar chart of top-k docs by attention at a single time index.

    Args:
        attn_logits: (B, H, Lt, M)
        t_index:     which time query to visualize
        doc_labels:  optional list of labels for M docs
        head_reduce: 'mean'|'max' aggregation across heads
        batch_index: item in batch
        top_k:       how many bars to display
    """
    logits = _to_numpy(attn_logits)
    assert logits.ndim == 4, "attn_logits must be (B,H,Lt,M)"
    logits = logits[batch_index : batch_index + 1]  # (1,H,Lt,M)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    probs = _head_reduce(probs, head_reduce)[0]  # (Lt,M)

    Lt, M = probs.shape
    t_index = int(np.clip(t_index, 0, Lt - 1))
    scores = probs[t_index]  # (M,)

    idx = np.argsort(scores)[::-1][: min(top_k, M)]
    vals = scores[idx]
    labels = [doc_labels[i] if (doc_labels and i < len(doc_labels)) else f"doc_{i}" for i in idx]

    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", hovertemplate="p=%{x:.3f}<extra></extra>"))
    fig.update_layout(
        title=f"{title}={t_index}",
        xaxis_title="Attention prob.",
        yaxis_title="Documents",
        height=max(300, 25 * len(idx)),
    )
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_plotly(fig: go.Figure, path: str) -> None:
    """
    Save a plotly Figure to disk. Format inferred by extension.

    Examples:
        save_plotly(fig, "out/plot.html")  # always works
        save_plotly(fig, "out/plot.png")   # requires 'kaleido' installed
    """
    path = str(path)
    if path.lower().endswith(".html"):
        fig.write_html(path, include_plotlyjs="cdn")
    else:
        try:
            fig.write_image(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save '{path}'. PNG/PDF export needs 'kaleido' "
                f"(pip install -U kaleido). Original error: {e}"
            )
