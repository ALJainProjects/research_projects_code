#!/usr/bin/env python3
"""
Enhanced evaluator for CT-GSSN.

Features:
- Loads a saved state dict and runs multiple evaluations:
  * Forecasting metrics: MSE / MAE / MAPE
  * Per-node metrics (optional)
  * Imputation evaluation: randomly hide a fraction of observed inputs at test time
  * Robustness evaluation: inject random edge noise into the graph
  * Stability diagnostics: max eigenvalue of the symmetric part of A across layers/timesteps
- Supports datasets: MIMIC-III, METR-LA, Synthetic IMTS (from repo)
- Optional plotting and saving predictions

Examples
--------
python scripts/eval.py \
  --config configs/ct_gssn/mimic_base.yaml \
  --ckpt runs/ct_gssn/checkpoints/best.pt \
  --device cuda \
  --batch-size 8 \
  --impute-ratio 0.15 \
  --edge-noise 0.2 \
  --stability \
  --per-node \
  --save-preds out/mimic_preds.pt

Notes
-----
- This evaluator assumes the CT-GSSN forward returns (y_pred, aux) where:
    y_pred: (B, N, L, out_dim)
    aux: {"A": (L_layers, B, L, N, D, D), "c": (L_layers, B, L, N, C)} when return_aux=True
- It reads model/data config from YAML (with optional "inherit" base)
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

# --- Project imports ---
from ct_gssn.models.ct_gssn import CTGSSN, CTGSSNConfig
from ct_gssn.utils.metrics import mse_metric as _mse_metric  # guaranteed in repo
try:
    # Optional helpers
    from ct_gssn.utils.plotting import plot_series_plotly
except Exception:
    plot_series_plotly = None

# Datasets (some may be optional depending on your checkout)
try:
    from ct_gssn.data.mimic3 import MIMIC3Dataset, collate_mimic3
except Exception:
    MIMIC3Dataset, collate_mimic3 = None, None

try:
    from ct_gssn.data.metr_la import METRLADataset, collate_metrla
except Exception:
    METRLADataset, collate_metrla = None, None

try:
    from ct_gssn.data.irregular_dataset import SyntheticIMTSDataset, collate_imts
except Exception:
    SyntheticIMTSDataset, collate_imts = None, None


# ------------------------------ Utilities ------------------------------ #

def load_config(path: str) -> Dict:
    """Load YAML config with optional 'inherit' base."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "inherit" in cfg:
        base = cfg["inherit"]
        with open(base, "r") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit"})
        cfg = base_cfg
    return cfg


def mae_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Mean Absolute Error over masked positions.
    Shapes:
      y_pred: (B, N, L, 1)
      y_true: (B, N, L, 1)
      mask:   (B, N, L)
    """
    diff = (y_pred - y_true).abs().squeeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return float((diff * mask).sum() / denom)


def mape_metric(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error over masked positions.
    Adds eps to denominator for stability.
    """
    y_true_ = y_true.abs().clamp_min(eps)
    pct = ((y_pred - y_true).abs() / y_true_).squeeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return float((pct * mask).sum() / denom)


def per_node_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor
) -> Dict[int, Dict[str, float]]:
    """
    Compute MSE/MAE/MAPE per node.
    Returns: {node_idx: {"mse": ..., "mae": ..., "mape": ...}}
    """
    B, N, L, _ = y_pred.shape
    out = {}
    for n in range(N):
        m_n = mask[:, n]  # (B, L)
        if m_n.sum() == 0:
            out[n] = {"mse": float("nan"), "mae": float("nan"), "mape": float("nan")}
            continue
        ypn = y_pred[:, n]  # (B, L, 1)
        ytn = y_true[:, n]
        mse = float(_mse_metric(ypn.unsqueeze(1), ytn.unsqueeze(1), m_n.unsqueeze(1)))
        mae = mae_metric(ypn.unsqueeze(1), ytn.unsqueeze(1), m_n.unsqueeze(1))
        mape = mape_metric(ypn.unsqueeze(1), ytn.unsqueeze(1), m_n.unsqueeze(1))
        out[n] = {"mse": mse, "mae": mae, "mape": mape}
    return out


def add_edge_noise(adj: torch.Tensor, noise_ratio: float, symmetric: bool = True) -> torch.Tensor:
    """
    Inject random flips to adjacency entries off-diagonal.
      adj: (B, N, N) in {0,1} or weighted [0,1], we binarize for flips.
      noise_ratio: fraction of off-diagonal entries to flip.
    """
    if noise_ratio <= 0:
        return adj
    B, N, _ = adj.shape
    out = adj.clone()
    for b in range(B):
        A = (out[b] > 0).float()
        # off-diagonal indices
        idx = torch.nonzero(~torch.eye(N, dtype=torch.bool, device=adj.device), as_tuple=False)
        num_flip = int(noise_ratio * idx.size(0))
        if num_flip <= 0:
            continue
        perm = torch.randperm(idx.size(0), device=adj.device)[:num_flip]
        flips = idx[perm]
        A[flips[:, 0], flips[:, 1]] = 1 - A[flips[:, 0], flips[:, 1]]
        if symmetric:
            A[flips[:, 1], flips[:, 0]] = 1 - A[flips[:, 1], flips[:, 0]]
        out[b] = A
    return out


def stability_summary(A_seq: torch.Tensor) -> Dict[str, float]:
    """
    Given A_seq of shape (L_layers, B, L, N, D, D),
    compute eigenvalues of symmetric part and summarize.
    Returns: dict with max/mean eigenvalues across the batch/time/layers/nodes.
    """
    # symmetric part
    S = 0.5 * (A_seq + A_seq.transpose(-1, -2))  # (..., D, D)
    eigs = torch.linalg.eigvalsh(S)              # (..., D) real & sorted
    max_eig = eigs.max().item()
    mean_eig = eigs.mean().item()
    pos_frac = (eigs > 0).float().mean().item()
    return {"max_eig": max_eig, "mean_eig": mean_eig, "pos_frac": pos_frac}


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ------------------------------ Data Loading ------------------------------ #

def build_loader(cfg: Dict, split: str, batch_size: int, device: str) -> Tuple[DataLoader, str]:
    """
    Create dataset + dataloader from config.
    Recognized cfg["data"]["name"]: "mimic3", "metr_la", "synthetic"
    """
    name = cfg["data"]["name"].lower()
    root = cfg["data"].get("root", ".")
    if name == "mimic3":
        if MIMIC3Dataset is None:
            raise RuntimeError("MIMIC3Dataset not available. Ensure ct_gssn.data.mimic3 exists.")
        ds = MIMIC3Dataset(root=root, split=split)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_mimic3)
        return dl, "mimic3"
    elif name == "metr_la":
        if METRLADataset is None:
            raise RuntimeError("METRLADataset not available. Ensure ct_gssn.data.metr_la exists.")
        ds = METRLADataset(root=root, split=split)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_metrla)
        return dl, "metr_la"
    elif name == "synthetic":
        if SyntheticIMTSDataset is None:
            raise RuntimeError("SyntheticIMTSDataset not available.")
        # Let user control N,L etc from cfg if present
        dcfg = cfg["data"]
        ds = SyntheticIMTSDataset(
            num_samples=dcfg.get("num_samples", 128),
            N=dcfg.get("N", 12),
            L=dcfg.get("L", 48),
            input_dim=dcfg.get("input_dim", 4),
            out_dim=dcfg.get("out_dim", 1),
            obs_prob=dcfg.get("obs_prob", 0.6),
            seed=dcfg.get("seed", 42),
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_imts)
        return dl, "synthetic"
    else:
        raise ValueError(f"Unknown dataset name: {name}")


# ------------------------------ Core Eval Routines ------------------------------ #

@torch.no_grad()
def eval_forecasting(
    model: CTGSSN,
    loader: DataLoader,
    device: torch.device,
    per_node: bool = False,
    return_aux: bool = False,
    edge_noise: float = 0.0,
) -> Dict[str, object]:
    """
    Standard forecasting evaluation (no input corruption).
    Optionally perturbs the adjacency by `edge_noise` flips for robustness.
    """
    model.eval()
    mses, maes, mapes = [], [], []
    node_breakdown = None
    last_pred, last_true, last_mask = None, None, None
    last_aux = None

    for batch in loader:
        batch = move_to_device(batch, device)
        adj = batch["adj"]
        if edge_noise > 0:
            adj = add_edge_noise(adj, noise_ratio=edge_noise)

        y_pred, aux = model(batch["x"], batch["mask"], batch["deltas"], adj, return_aux=return_aux)
        mses.append(float(_mse_metric(y_pred, batch["y"], batch["mask"])))
        maes.append(mae_metric(y_pred, batch["y"], batch["mask"]))
        mapes.append(mape_metric(y_pred, batch["y"], batch["mask"]))

        # cache the last batch for optional plotting/saving
        last_pred, last_true, last_mask = y_pred.cpu(), batch["y"].cpu(), batch["mask"].cpu()
        if return_aux:
            last_aux = {k: v.cpu() for k, v in aux.items()}

    out = {
        "mse": sum(mses) / max(1, len(mses)),
        "mae": sum(maes) / max(1, len(maes)),
        "mape": sum(mapes) / max(1, len(mapes)),
        "last_pred": last_pred,
        "last_true": last_true,
        "last_mask": last_mask,
    }
    if per_node and last_pred is not None:
        node_breakdown = per_node_metrics(last_pred, last_true, last_mask)
        out["per_node"] = node_breakdown
    if return_aux and last_aux is not None:
        out["aux"] = last_aux
    return out


@torch.no_grad()
def eval_imputation(
    model: CTGSSN,
    loader: DataLoader,
    device: torch.device,
    impute_ratio: float = 0.15,
) -> Dict[str, float]:
    """
    Imputation evaluation: randomly hold out a fraction of *observed* inputs at test time,
    hide them from the model (mask->0) and evaluate error on those held-out points.
    """
    model.eval()
    mses, maes, mapes = [], [], []

    for batch in loader:
        batch = move_to_device(batch, device)

        # Construct hold-out mask from observed positions
        obs = (batch["mask"] > 0).float()  # (B, N, L)
        flat = obs.view(-1)
        k = max(1, int(impute_ratio * flat.sum().item()))
        idx = flat.nonzero(as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.numel(), device=device)[:k]]
        holdout = torch.zeros_like(flat)
        holdout[perm] = 1.0
        holdout = holdout.view_as(obs)

        # Hide these from the model (set mask->0 so input impulses are not used)
        mask_prime = (obs - holdout).clamp_min(0.0)

        # Forward with modified mask
        y_pred, _ = model(batch["x"], mask_prime, batch["deltas"], batch["adj"])

        # Evaluate only on the held-out positions
        mses.append(float(_mse_metric(y_pred, batch["y"], holdout)))
        maes.append(mae_metric(y_pred, batch["y"], holdout))
        mapes.append(mape_metric(y_pred, batch["y"], holdout))

    return {
        "impute_mse": sum(mses) / max(1, len(mses)),
        "impute_mae": sum(maes) / max(1, len(maes)),
        "impute_mape": sum(mapes) / max(1, len(mapes)),
    }


# ------------------------------ Main CLI ------------------------------ #

def parse_args():
    ap = argparse.ArgumentParser(description="Enhanced CT-GSSN evaluator")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--ckpt", required=True, help="Path to model state_dict .pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val-split", default=None, help="Override validation split name")

    # Extra eval options
    ap.add_argument("--per-node", action="store_true", help="Report per-node metrics")
    ap.add_argument("--impute-ratio", type=float, default=0.0, help="Fraction of observed inputs to hide for imputation eval")
    ap.add_argument("--edge-noise", type=float, default=0.0, help="Adjacency flip ratio for robustness eval")
    ap.add_argument("--stability", action="store_true", help="Compute stability diagnostics from A matrices")

    # Output controls
    ap.add_argument("--save-preds", type=str, default=None, help="Path to save a torch .pt dict with predictions/targets/masks")
    ap.add_argument("--plot", action="store_true", help="Plot the last sample prediction vs ground truth (requires plotly & utils)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config & model
    cfg = load_config(args.config)
    model = CTGSSN(CTGSSNConfig(**cfg["model"])).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Data loader
    split = args.val_split or cfg["data"].get("val_split", "val")
    val_loader, ds_name = build_loader(cfg, split=split, batch_size=args.batch_size, device=args.device)

    # 1) Forecasting (optionally with robustness noise and stability aux)
    forecast = eval_forecasting(
        model,
        val_loader,
        device=device,
        per_node=args.per_node,
        return_aux=args.stability,   # only fetch aux if required
        edge_noise=args.edge_noise,
    )
    print(f"[{ds_name}] Forecasting:  MSE={forecast['mse']:.6f}  MAE={forecast['mae']:.6f}  MAPE={forecast['mape']:.6f}")

    if args.per_node and "per_node" in forecast:
        print("Per-node metrics (first 10 shown):")
        for n, m in list(forecast["per_node"].items())[:10]:
            print(f"  node {n:3d}: mse={m['mse']:.6f}, mae={m['mae']:.6f}, mape={m['mape']:.6f}")

    # 2) Imputation (if requested)
    if args.impute_ratio > 0:
        impute = eval_imputation(model, val_loader, device=device, impute_ratio=args.impute_ratio)
        print(f"[{ds_name}] Imputation (ratio={args.impute_ratio:.2f}): "
              f"MSE={impute['impute_mse']:.6f}  MAE={impute['impute_mae']:.6f}  MAPE={impute['impute_mape']:.6f}")

    # 3) Stability diagnostics (if requested)
    if args.stability and "aux" in forecast and "A" in forecast["aux"]:
        stab = stability_summary(forecast["aux"]["A"])
        print(f"[{ds_name}] Stability: max_eig={stab['max_eig']:.6f}  mean_eig={stab['mean_eig']:.6f}  pos_frac={stab['pos_frac']:.6f}")

    # Optional: save predictions for further analysis
    if args.save_preds:
        out = {
            "y_pred": forecast["last_pred"],
            "y_true": forecast["last_true"],
            "mask": forecast["last_mask"],
        }
        Path(args.save_preds).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.save_preds)
        print(f"Saved last-batch predictions to: {args.save_preds}")

    # Optional: quick plot of the last sample
    if args.plot and plot_series_plotly is not None and forecast["last_pred"] is not None:
        try:
            # Plot first batch, first node, all timesteps
            pred = forecast["last_pred"][0, 0, :, 0].cpu()
            true = forecast["last_true"][0, 0, :, 0].cpu()
            fig = plot_series_plotly(pred, true)
            # We can't render Plotly inline here; just write HTML nearby
            out_html = Path("eval_plot.html")
            fig.write_html(str(out_html))
            print(f"Wrote interactive plot: {out_html.resolve()}")
        except Exception as e:
            print(f"Plotting failed: {e}")

    # Done
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
