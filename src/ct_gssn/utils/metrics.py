# src/ct_gssn/utils/metrics.py
"""
Metrics for CT-GSSN (masked-aware).

All metrics accept shapes:
  y_pred: (B, N, L, C) or (B, N, L, 1)
  y_true: (B, N, L, C) or (B, N, L, 1)
  mask:   (B, N, L) or None   — 1=include, 0=ignore

Returned values are Python floats (or dicts of floats for breakdowns).
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch


# ------------------------- utilities ------------------------- #

def _coerce_channel_last1(y: torch.Tensor) -> torch.Tensor:
    """Ensure final dimension is channel; if missing, add 1-length channel."""
    if y.dim() == 3:  # (B, N, L)
        return y.unsqueeze(-1)
    return y


def _masked_reduce(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduce: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reduce x with (optional) mask over all but the last dimension(s) we care about.
    Here we reduce over all elements (scalar result).
    """
    if mask is None:
        if reduce == "mean":
            return x.mean()
        elif reduce == "sum":
            return x.sum()
        else:
            raise ValueError(f"Unknown reduce={reduce}")
    # Broadcast mask to match x without channel dimension
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    if reduce == "mean":
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom
    elif reduce == "sum":
        return (x * mask).sum()
    else:
        raise ValueError(f"Unknown reduce={reduce}")


def _masked_per_node_reduce(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Compute per-node mean of x (B,N,L[,C]) with optional mask (B,N,L).
    Returns: (N,) tensor (mean over B,L and channels).
    """
    B, N = x.shape[0], x.shape[1]
    # reduce channels by mean to keep it simple
    if x.dim() == 4:
        x = x.mean(dim=-1)  # (B,N,L)
    if mask is None:
        return x.mean(dim=(0, 2))  # (N,)
    # match dims
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)  # -> (B,N,L,1) possibly
    # now x is (B,N,L), mask is (B,N,L,1) or (B,N,L)
    mask_s = mask.squeeze(-1) if mask.dim() == 4 else mask
    denom = mask_s.sum(dim=(0, 2)).clamp_min(1.0)  # (N,)
    num = (x * mask_s).sum(dim=(0, 2))             # (N,)
    return num / denom


# ------------------------- core metrics ------------------------- #

@torch.no_grad()
def mse_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Mean Squared Error with optional mask.
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)
    diff2 = (y_pred - y_true) ** 2
    return float(_masked_reduce(diff2, mask, reduce="mean").item())


@torch.no_grad()
def mae_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Mean Absolute Error with optional mask.
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)
    diff = (y_pred - y_true).abs()
    return float(_masked_reduce(diff, mask, reduce="mean").item())


@torch.no_grad()
def rmse_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Root Mean Squared Error with optional mask.
    """
    return float(torch.sqrt(torch.tensor(mse_metric(y_pred, y_true, mask))).item())


@torch.no_grad()
def mape_metric_safe(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6
) -> float:
    """
    Mean Absolute Percentage Error (safe): |pred-true| / max(|true|, eps).
    Good when y_true can be near zero.
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)
    denom = y_true.abs().clamp_min(eps)
    mape = (y_pred - y_true).abs() / denom
    return float(_masked_reduce(mape, mask, reduce="mean").item())


@torch.no_grad()
def r2_score_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> float:
    """
    Coefficient of determination R^2 with optional mask.
    R^2 = 1 - SS_res / SS_tot
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)

    if mask is None:
        y_bar = y_true.mean()
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_bar) ** 2).sum().clamp_min(eps)
    else:
        # expand mask to channels
        m = mask
        while m.dim() < y_true.dim():
            m = m.unsqueeze(-1)
        denom = m.sum().clamp_min(1.0)
        y_bar = (y_true * m).sum() / denom

        ss_res = ((y_true - y_pred) ** 2 * m).sum()
        ss_tot = (((y_true - y_bar) ** 2) * m).sum().clamp_min(eps)

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2.item())


@torch.no_grad()
def explained_variance_metric(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> float:
    """
    Explained variance = 1 - Var(y - yhat) / Var(y)
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)

    if mask is None:
        err = y_true - y_pred
        var_err = err.var(unbiased=False)
        var_y = y_true.var(unbiased=False).clamp_min(eps)
    else:
        m = mask
        while m.dim() < y_true.dim():
            m = m.unsqueeze(-1)
        denom = m.sum().clamp_min(1.0)
        mean_y = (y_true * m).sum() / denom
        mean_e = ((y_true - y_pred) * m).sum() / denom
        var_y = (((y_true - mean_y) ** 2) * m).sum() / denom
        var_err = ((((y_true - y_pred) - mean_e) ** 2) * m).sum() / denom
        var_y = var_y.clamp_min(eps)
    ev = 1.0 - (var_err / var_y)
    return float(ev.item())


@torch.no_grad()
def masked_corr_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> float:
    """
    Pearson correlation between y_pred and y_true over masked entries.
    Returns the correlation averaged over channels (if any).
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)

    if mask is None:
        yp = y_pred - y_pred.mean()
        yt = y_true - y_true.mean()
        num = (yp * yt).sum()
        den = torch.sqrt((yp ** 2).sum().clamp_min(eps) * (yt ** 2).sum().clamp_min(eps))
        corr = num / den
        return float(corr.item())

    m = mask
    while m.dim() < y_true.dim():
        m = m.unsqueeze(-1)

    denom = m.sum().clamp_min(1.0)
    mu_p = (y_pred * m).sum() / denom
    mu_t = (y_true * m).sum() / denom
    yp = (y_pred - mu_p) * m
    yt = (y_true - mu_t) * m

    num = (yp * yt).sum()
    den = torch.sqrt((yp ** 2).sum().clamp_min(eps) * (yt ** 2).sum().clamp_min(eps))
    corr = num / den
    return float(corr.item())


# ------------------------- breakdowns ------------------------- #

@torch.no_grad()
def per_node_mse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """
    Return MSE per node {node_index: mse}.
    """
    y_pred = _coerce_channel_last1(y_pred)
    y_true = _coerce_channel_last1(y_true)
    diff2 = (y_pred - y_true) ** 2  # (B,N,L,1)

    if mask is None:
        vals = diff2.mean(dim=(0, 2, 3))  # (N,)
    else:
        m = mask.unsqueeze(-1)  # (B,N,L,1)
        denom = m.sum(dim=(0, 2, 3)).clamp_min(1.0)  # (N,)
        vals = (diff2 * m).sum(dim=(0, 2, 3)) / denom  # (N,)

    return {int(i): float(v.item()) for i, v in enumerate(vals)}


@torch.no_grad()
def summary_dict(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Handy bundle for logging. Keys are prefixed (e.g., 'val/').
    """
    p = prefix + "/" if prefix and not prefix.endswith("/") else prefix
    return {
        f"{p}mse": mse_metric(y_pred, y_true, mask),
        f"{p}mae": mae_metric(y_pred, y_true, mask),
        f"{p}rmse": rmse_metric(y_pred, y_true, mask),
        f"{p}mape_safe": mape_metric_safe(y_pred, y_true, mask),
        f"{p}r2": r2_score_metric(y_pred, y_true, mask),
        f"{p}explained_var": explained_variance_metric(y_pred, y_true, mask),
        f"{p}corr": masked_corr_metric(y_pred, y_true, mask),
    }
