# src/ct_gssn/data/mimic3.py
"""
MIMIC-III dataset loader for CT-GSSN.

Expects output from scripts/convert_mimic3.py:
  data/mimic3_proc/{train,val,test}.pt

Each file is a list[dict] with keys:
  x: (N, L, P)
  y: (N, L, Q)
  mask: (N, L)
  deltas: (L,)
  adj: (N, N) or (L, N, N) (optional if we rebuild)
  meta: dict (optional)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

from .utils_data import build_knn_adjacency, zscore_, simple_impute_, to_float32


class MIMIC3Dataset(Dataset):
    """
    Parameters
    ----------
    root : str
        Directory with {split}.pt files.
    split : {"train","val","test"}
    adjacency : {"file","knn","identity"}
        - file: use 'adj' from sample if present
        - knn: recompute from mean node features (x.mean over time)
        - identity: I_N
    knn_k : int
        k for KNN graph if adjacency="knn".
    normalize : {"none","per_sample"}
        z-score on x (per sample) using observed mask.
    impute : {"none","ffill","zero"}
        Simple imputation for x at missing positions.
    keep_meta : bool
        Return 'meta' in samples if present.
    dtype : torch.dtype
        Output dtype (default float32).
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        adjacency: str = "file",
        knn_k: int = 8,
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
        self.adjacency = adjacency
        self.knn_k = knn_k
        self.normalize = normalize
        self.impute = impute
        self.keep_meta = keep_meta
        self.dtype = dtype

    def __len__(self):
        return len(self.samples)

    def _adj_from_policy(self, s: Dict[str, Any]) -> torch.Tensor:
        x = to_float32(s["x"])
        if self.adjacency == "file" and "adj" in s:
            adj = to_float32(s["adj"])
        elif self.adjacency == "identity":
            N = x.size(0)
            adj = torch.eye(N, dtype=torch.float32)
        elif self.adjacency == "knn":
            # use mean over time as node features
            feats = x.mean(dim=1)  # (N,P)
            adj = build_knn_adjacency(feats, k=self.knn_k)
        else:
            raise ValueError(f"Unknown adjacency policy: {self.adjacency}")
        return adj

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = to_float32(s["x"]).to(self.dtype)
        y = to_float32(s["y"]).to(self.dtype)
        mask = to_float32(s["mask"]).to(self.dtype)
        deltas = to_float32(s["deltas"]).to(self.dtype)
        adj = self._adj_from_policy(s).to(self.dtype)

        # Impute
        if self.impute == "ffill":
            x = simple_impute_(x, mask, mode="ffill")
        elif self.impute == "zero":
            x = simple_impute_(x, mask, mode="zero")

        # Normalize
        if self.normalize == "per_sample":
            x = zscore_(x, mask)

        out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
        if self.keep_meta and "meta" in s:
            out["meta"] = s["meta"]
        return out


def collate_mimic3(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Standard stack collate for fixed-length sequences.
    """
    x = torch.stack([b["x"] for b in batch], dim=0)          # (B,N,L,P)
    y = torch.stack([b["y"] for b in batch], dim=0)          # (B,N,L,Q)
    mask = torch.stack([b["mask"] for b in batch], dim=0)    # (B,N,L)
    deltas = torch.stack([b["deltas"] for b in batch], dim=0)# (B,L)
    adj = torch.stack([b["adj"] for b in batch], dim=0)      # (B,N,N)
    out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta") for b in batch]
    return out


def collate_mimic3_pad(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Ragged → padded collate for variable-length sequences.
    Produces 'pad_mask': (B,Lmax).
    """
    N = batch[0]["x"].shape[0]
    Ls = [b["x"].shape[1] for b in batch]
    Lmax = max(Ls)

    def pad_T(x, Lmax):
        pad = Lmax - x.shape[1]
        return x if pad <= 0 else torch.nn.functional.pad(x, (0, 0, 0, pad))

    xs, ys, ms, ds, pads, adjs = [], [], [], [], [], []
    for b in batch:
        Li = b["x"].shape[1]
        xs.append(pad_T(b["x"], Lmax))
        ys.append(pad_T(b["y"], Lmax))
        ms.append(pad_T(b["mask"], Lmax))
        ds.append(pad_T(b["deltas"].unsqueeze(0), Lmax).squeeze(0))
        adjs.append(b["adj"])
        pad_mask = torch.zeros(Lmax, dtype=b["x"].dtype)
        pad_mask[:Li] = 1.0
        pads.append(pad_mask)

    out = {
        "x": torch.stack(xs, 0),
        "y": torch.stack(ys, 0),
        "mask": torch.stack(ms, 0),
        "deltas": torch.stack(ds, 0),
        "adj": torch.stack(adjs, 0),
        "pad_mask": torch.stack(pads, 0),
    }
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta") for b in batch]
    return out
