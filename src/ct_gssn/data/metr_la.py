# src/ct_gssn/data/metr_la.py
"""
METR-LA dataset loader for CT-GSSN.

Expects output from scripts/convert_metr_la.py:
  data/metr-la_proc/{train,val,test}.pt

Each file is a list[dict] with keys:
  x: (N, L, 1) or (N, L, P)
  y: (N, L, 1) or (N, L, Q)
  mask: (N, L)
  deltas: (L,)
  adj: (N, N) or (L, N, N) or list of (N, N)
  meta: dict (optional)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from .utils_data import zscore_, simple_impute_, to_float32


class METRLADataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Directory containing {split}.pt produced by the converter.
    split : {"train","val","test"}
    use_dynamic_adj : bool
        If True and the stored 'adj' is a sequence of (N,N), it will be stacked to (L,N,N) per sample.
    normalize : {"none","per_sample","global"}
        - none: do nothing
        - per_sample: z-score per sample using x statistics
        - global: compute z-score stats over the whole split (cached in memory) and apply
    impute : {"none","ffill","zero"}
        Simple imputation for missing x where mask==0 (forward-fill along time, else zero).
    keep_meta : bool
        If True, return "meta" alongside tensors (collates will pass lists).
    dtype : torch.dtype
        Output dtype for tensors (default float32).
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        use_dynamic_adj: bool = False,
        normalize: str = "none",
        impute: str = "none",
        keep_meta: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.samples: List[Dict[str, Any]] = torch.load(self.root / f"{split}.pt", map_location="cpu")
        if not isinstance(self.samples, list) or len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root / f'{split}.pt'}")
        self.use_dynamic_adj = use_dynamic_adj
        self.normalize = normalize
        self.impute = impute
        self.keep_meta = keep_meta
        self.dtype = dtype

        # Pre-compute global stats if requested
        self._mu = None
        self._sigma = None
        if normalize == "global":
            # Concatenate along batch for stable mean/std. Shapes: x: (N,L,P)
            xs = []
            for s in self.samples:
                x = to_float32(s["x"])
                m = to_float32(s["mask"])
                # Use observed values only to compute stats
                obs = (m.unsqueeze(-1) > 0).expand_as(x)
                vals = x[obs]
                if vals.numel() > 0:
                    xs.append(vals)
            if len(xs) == 0:
                self._mu = torch.tensor(0.0)
                self._sigma = torch.tensor(1.0)
            else:
                cat = torch.cat(xs, dim=0)
                self._mu = cat.mean()
                self._sigma = cat.std().clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_dynamic_adj(self, adj: Any, L: int) -> torch.Tensor:
        if self.use_dynamic_adj:
            if isinstance(adj, list):
                adj = torch.stack([to_float32(a) for a in adj], dim=0)  # (L,N,N)
            elif torch.is_tensor(adj) and adj.dim() == 2:
                adj = adj.unsqueeze(0).repeat(L, 1, 1)  # make time-constant dynamic
        else:
            if isinstance(adj, list):
                adj = to_float32(adj[0])  # take first (assume stationary)
        return to_float32(adj)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        x = to_float32(s["x"]).to(self.dtype)      # (N,L,P)
        y = to_float32(s["y"]).to(self.dtype)      # (N,L,Q)
        mask = to_float32(s["mask"]).to(self.dtype)  # (N,L)
        deltas = to_float32(s["deltas"]).to(self.dtype)  # (L,)
        adj = self._maybe_dynamic_adj(s["adj"], L=x.shape[1]).to(self.dtype)

        # Impute x where mask==0 (optionally)
        if self.impute == "ffill":
            x = simple_impute_(x, mask, mode="ffill")
        elif self.impute == "zero":
            x = simple_impute_(x, mask, mode="zero")

        # Normalize
        if self.normalize == "per_sample":
            x = zscore_(x, mask)
        elif self.normalize == "global":
            x = (x - self._mu) / self._sigma

        out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
        if self.keep_meta and "meta" in s:
            out["meta"] = s["meta"]
        return out


def collate_metrla(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Standard collate for fixed-length sequences.
    Handles both static (N,N) and dynamic (L,N,N) adjacencies.
    """
    x = torch.stack([b["x"] for b in batch], dim=0)                # (B,N,L,P)
    y = torch.stack([b["y"] for b in batch], dim=0)                # (B,N,L,Q)
    mask = torch.stack([b["mask"] for b in batch], dim=0)          # (B,N,L)
    deltas = torch.stack([b["deltas"] for b in batch], dim=0)      # (B,L)
    adj0 = batch[0]["adj"]
    if adj0.dim() == 2:
        adj = torch.stack([b["adj"] for b in batch], dim=0)        # (B,N,N)
    else:
        adj = torch.stack([b["adj"] for b in batch], dim=0)        # (B,L,N,N)
    out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta") for b in batch]
    return out


def collate_metrla_pad(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Ragged → padded collate (if some samples have different L).
    Produces an additional 'pad_mask': (B,Lmax) with 1 where valid, else 0.
    """
    N = batch[0]["x"].shape[0]
    P = batch[0]["x"].shape[-1]
    Ls = [b["x"].shape[1] for b in batch]
    Lmax = max(Ls)
    Q = batch[0]["y"].shape[-1]

    def pad_T(x, Lmax):
        pad = Lmax - x.shape[1]
        if pad <= 0:
            return x
        return torch.nn.functional.pad(x, (0, 0, 0, pad))  # pad time on the right

    xs, ys, ms, ds, pads, adjs = [], [], [], [], [], []
    for b in batch:
        Li = b["x"].shape[1]
        xs.append(pad_T(b["x"], Lmax))         # (N,Lmax,P)
        ys.append(pad_T(b["y"], Lmax))         # (N,Lmax,Q)
        ms.append(pad_T(b["mask"], Lmax))      # (N,Lmax)
        ds.append(pad_T(b["deltas"].unsqueeze(0), Lmax).squeeze(0))  # (Lmax,)
        # adj: keep static per-sample here
        adj = b["adj"]
        if adj.dim() == 3 and adj.shape[0] == Li:
            # dynamic -> clip/pad in time if needed
            padA = Lmax - Li
            if padA > 0:
                adj = torch.nn.functional.pad(adj, (0, 0, 0, 0, 0, padA))
        adjs.append(adj if adj.dim() == 2 else adj[:Lmax])
        pad_mask = torch.zeros(Lmax, dtype=b["x"].dtype)
        pad_mask[:Li] = 1.0
        pads.append(pad_mask)

    out = {
        "x": torch.stack(xs, 0), "y": torch.stack(ys, 0), "mask": torch.stack(ms, 0),
        "deltas": torch.stack(ds, 0), "adj": torch.stack(adjs, 0),
        "pad_mask": torch.stack(pads, 0)
    }
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta") for b in batch]
    return out
