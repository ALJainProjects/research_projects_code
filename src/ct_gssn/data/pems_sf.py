# src/ct_gssn/data/pems_sf.py
"""
PEMS-SF dataset loader (optional), mirroring METR-LA structure.

If you export the PEMS-SF dataset to the same schema as METR-LA:
  data/pems-sf_proc/{train,val,test}.pt

you can use this loader directly.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

from .utils_data import zscore_, simple_impute_, to_float32


class PEMSSF(Dataset):
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
        self.samples: List[Dict[str, Any]] = torch.load(self.root / f"{split}.pt", map_location="cpu")
        if not isinstance(self.samples, list) or len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root / f'{split}.pt'}")
        self.use_dynamic_adj = use_dynamic_adj
        self.normalize = normalize
        self.impute = impute
        self.keep_meta = keep_meta
        self.dtype = dtype

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = to_float32(s["x"]).to(self.dtype)
        y = to_float32(s["y"]).to(self.dtype)
        mask = to_float32(s["mask"]).to(self.dtype)
        deltas = to_float32(s["deltas"]).to(self.dtype)
        adj = s["adj"]
        if self.use_dynamic_adj and isinstance(adj, list):
            adj = torch.stack([to_float32(a) for a in adj], dim=0)
        else:
            adj = to_float32(adj)
        adj = adj.to(self.dtype)

        if self.impute == "ffill":
            x = simple_impute_(x, mask, mode="ffill")
        elif self.impute == "zero":
            x = simple_impute_(x, mask, mode="zero")

        if self.normalize == "per_sample":
            x = zscore_(x, mask)

        out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
        if self.keep_meta and "meta" in s:
            out["meta"] = s["meta"]
        return out
