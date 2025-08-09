#!/usr/bin/env python3
"""
Enhanced Plotly plotting for CT-GSSN predictions.

Features:
---------
- Can plot:
    * Single series (B,N,L,1) → pick batch idx and node idx
    * Multiple nodes in subplots
    * Multiple samples in a loop or grid
- Optional mask overlay to show observed points
- Optional confidence intervals (if provided separately)
- Save to HTML/PNG or display interactively
- Normalize series for visual clarity

Examples:
---------
python scripts/plot_preds.py \
    --pred_pt out/y_pred.pt \
    --true_pt out/y_true.pt \
    --mask_pt out/mask.pt \
    --batch-idx 0 \
    --nodes 0 1 2 \
    --save-html preds.html

python scripts/plot_preds.py \
    --pred_pt out/y_pred.pt \
    --true_pt out/y_true.pt \
    --loop-batches \
    --nodes 0 1 \
    --normalize
"""

import argparse
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def load_tensor(path):
    """Load tensor from .pt or .npy."""
    if path is None:
        return None
    ext = Path(path).suffix
    if ext == ".pt":
        return torch.load(path, map_location="cpu")
    elif ext == ".npy":
        return torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_series(*arrays):
    """Normalize arrays to [0,1] range across all inputs for visual clarity."""
    concat = torch.cat([a.reshape(-1) for a in arrays if a is not None])
    minv, maxv = concat.min(), concat.max()
    scale = (maxv - minv).clamp_min(1e-8)
    return [(a - minv) / scale for a in arrays]


def plot_sample(pred, true, mask=None, ci_low=None, ci_high=None, title=None, normalize=False):
    """
    Plot a single time series (1D arrays) with optional mask and CI.
    pred, true: (L,)
    mask: (L,)  1=observed, 0=missing
    ci_low, ci_high: (L,)
    """
    if normalize:
        pred, true, mask, ci_low, ci_high = normalize_series(pred, true, mask, ci_low, ci_high)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true, mode="lines", name="Ground Truth", line=dict(color="black")))
    fig.add_trace(go.Scatter(y=pred, mode="lines", name="Prediction", line=dict(color="royalblue")))

    if ci_low is not None and ci_high is not None:
        fig.add_trace(go.Scatter(
            y=ci_high, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            y=ci_low, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0,0,255,0.15)",
            name="Confidence Interval"
        ))

    if mask is not None:
        obs_idx = torch.nonzero(mask > 0.5).view(-1)
        fig.add_trace(go.Scatter(
            x=obs_idx, y=true[obs_idx], mode="markers",
            name="Observed", marker=dict(color="green", size=6, symbol="circle-open")
        ))

    fig.update_layout(
        title=title or "Prediction vs Ground Truth",
        xaxis_title="Time Step",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description="Enhanced Plotly plotting for CT-GSSN predictions")
    ap.add_argument("--pred_pt", required=True, help="Predictions tensor (.pt or .npy)")
    ap.add_argument("--true_pt", required=True, help="Ground truth tensor (.pt or .npy)")
    ap.add_argument("--mask_pt", help="Mask tensor (.pt or .npy) [optional]")
    ap.add_argument("--ci_low_pt", help="Lower bound CI tensor (.pt or .npy) [optional]")
    ap.add_argument("--ci_high_pt", help="Upper bound CI tensor (.pt or .npy) [optional]")

    ap.add_argument("--batch-idx", type=int, default=0, help="Batch index to plot")
    ap.add_argument("--nodes", type=int, nargs="+", default=[0], help="Node indices to plot")
    ap.add_argument("--normalize", action="store_true", help="Normalize series for visual clarity")

    ap.add_argument("--loop-batches", action="store_true", help="Loop over all batches and save plots")
    ap.add_argument("--save-html", type=str, help="Save plot(s) to HTML")
    ap.add_argument("--save-png", type=str, help="Save plot(s) to PNG (requires kaleido)")
    args = ap.parse_args()

    # Load tensors
    y_pred = load_tensor(args.pred_pt)
    y_true = load_tensor(args.true_pt)
    mask = load_tensor(args.mask_pt)
    ci_low = load_tensor(args.ci_low_pt)
    ci_high = load_tensor(args.ci_high_pt)

    if y_pred.ndim == 4:  # (B, N, L, 1)
        B, N, L, _ = y_pred.shape
    elif y_pred.ndim == 3:  # (B, N, L)
        B, N, L = y_pred.shape
    else:
        raise ValueError("Unsupported pred shape")

    def plot_for_batch(b):
        figs = []
        for n in args.nodes:
            pred_n = y_pred[b, n, :, 0] if y_pred.ndim == 4 else y_pred[b, n]
            true_n = y_true[b, n, :, 0] if y_true.ndim == 4 else y_true[b, n]
            mask_n = mask[b, n] if mask is not None else None
            ci_low_n = ci_low[b, n, :, 0] if ci_low is not None and ci_low.ndim == 4 else None
            ci_high_n = ci_high[b, n, :, 0] if ci_high is not None and ci_high.ndim == 4 else None

            fig = plot_sample(
                pred_n.cpu(), true_n.cpu(),
                mask=mask_n.cpu() if mask_n is not None else None,
                ci_low=ci_low_n.cpu() if ci_low_n is not None else None,
                ci_high=ci_high_n.cpu() if ci_high_n is not None else None,
                title=f"Batch {b}, Node {n}",
                normalize=args.normalize
            )
            figs.append(fig)
        return figs

    if args.loop_batches:
        out_dir = Path(args.save_html or args.save_png or "plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in range(B):
            figs = plot_for_batch(b)
            for i, fig in enumerate(figs):
                if args.save_html:
                    fig.write_html(str(out_dir / f"batch{b}_node{args.nodes[i]}.html"))
                if args.save_png:
                    fig.write_image(str(out_dir / f"batch{b}_node{args.nodes[i]}.png"))
                if not args.save_html and not args.save_png:
                    fig.show()
    else:
        figs = plot_for_batch(args.batch_idx)
        for fig in figs:
            if args.save_html:
                fig.write_html(args.save_html)
            if args.save_png:
                fig.write_image(args.save_png)
            if not args.save_html and not args.save_png:
                fig.show()


if __name__ == "__main__":
    main()
