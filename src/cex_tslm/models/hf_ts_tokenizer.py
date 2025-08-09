"""
Optional: numeric<->token helper for time series.

This module provides a *quantizer* for continuous TS values so you can:
  - Discretize (encode) TS into token ids for LM-style models
  - Reconstruct (decode) approximate values (e.g., for inspection/aux losses)
  - Auto-fit bin ranges from data with robust percentile clipping
  - Use per-dimension binning (each channel has its own range)
  - Choose quantization strategy: 'uniform' or 'gaussian'

Typical usage
-------------
tok = SimpleTSTokenizer(num_bins=256, method="uniform", per_dim=True)
tok.fit(train_batch_ts)                # (B,L,d) – computes per-dim ranges
ids, aux = tok.encode(ts)             # ids: (B,L,d), aux contains mask & edges
recon = tok.decode(ids)               # (B,L,d) reconstructed (bin centers)

Notes
-----
- PAD id is not reserved here; ids are in [0, num_bins-1].
- NaNs are supported: they get masked in aux["nan_mask"]; encoded NaNs clamp to closest bin.
- For 'gaussian' method we use mean/std per dim and map values to bins by standard score.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Literal

import torch


@dataclass
class _QuantParams:
    """
    Immutable-ish container for quantization parameters.
    If per_dim=True: vmin, vmax, mean, std are (d,) tensors; otherwise scalars.
    """
    num_bins: int
    per_dim: bool
    method: Literal["uniform", "gaussian"]
    vmin: Optional[torch.Tensor] = None
    vmax: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    eps: float = 1e-6

    def to(self, device: torch.device) -> "_QuantParams":
        qp = _QuantParams(**asdict(self))
        for k in ("vmin", "vmax", "mean", "std"):
            t = getattr(qp, k)
            if isinstance(t, torch.Tensor):
                setattr(qp, k, t.to(device))
        return qp


class SimpleTSTokenizer:
    """
    Discretize continuous time series to token ids and back.

    Args:
        num_bins: number of discrete bins (ids range [0, num_bins-1])
        vmin/vmax: initial global min/max used for 'uniform' (can be auto-fit)
        per_dim: if True, use separate ranges per channel (recommended)
        method: 'uniform' → linear bins between [vmin, vmax]
                'gaussian' → bins in z-score space using mean/std
        clip_percentiles: for .fit(); robust clipping (e.g., (0.5, 99.5))
        eps: numerical epsilon when normalizing
    """
    def __init__(
        self,
        num_bins: int = 256,
        vmin: float = -5.0,
        vmax: float = 5.0,
        per_dim: bool = False,
        method: Literal["uniform", "gaussian"] = "uniform",
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        eps: float = 1e-6,
    ):
        assert num_bins >= 2, "num_bins must be >= 2"
        assert method in ("uniform", "gaussian")
        self.num_bins = int(num_bins)
        self.method = method
        self.per_dim = bool(per_dim)
        self.clip_percentiles = clip_percentiles
        self.eps = float(eps)

        # Initialize quant params (can be refined by fit())
        if method == "uniform":
            self.qp = _QuantParams(
                num_bins=num_bins,
                per_dim=self.per_dim,
                method=method,
                vmin=torch.tensor(vmin, dtype=torch.float32),
                vmax=torch.tensor(vmax, dtype=torch.float32),
                eps=self.eps,
            )
        else:
            self.qp = _QuantParams(
                num_bins=num_bins,
                per_dim=self.per_dim,
                method=method,
                mean=torch.tensor(0.0, dtype=torch.float32),
                std=torch.tensor(1.0, dtype=torch.float32),
                eps=self.eps,
            )

    # ----------------------------- Fitting --------------------------------- #
    @torch.no_grad()
    def fit(self, ts: torch.Tensor) -> None:
        """
        Fit quantization params from data.

        Args:
            ts: (B, L, d) float tensor. NaNs ignored for stats.

        For 'uniform': sets vmin/vmax with robust percentile clipping.
        For 'gaussian': sets mean/std per dim, with std clamped to eps.
        """
        assert ts.ndim == 3, "ts must be (B,L,d)"
        device = ts.device
        B, L, d = ts.shape

        x = ts.reshape(B * L, d)
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            x = x.clone()
            x[nan_mask] = 0.0

        if self.per_dim:
            # per-dim stats
            if self.method == "uniform":
                lo, hi = self._percentiles(x, self.clip_percentiles)
                self.qp.vmin = lo.to(device)
                self.qp.vmax = hi.to(device)
            else:
                mean = self._safe_mean(x)
                std = self._safe_std(x, mean=mean)
                self.qp.mean = mean.to(device)
                self.qp.std = std.clamp_min(self.eps).to(device)
        else:
            # global stats
            if self.method == "uniform":
                lo, hi = self._percentiles(x, self.clip_percentiles)
                self.qp.vmin = torch.tensor(lo.min(), device=device)
                self.qp.vmax = torch.tensor(hi.max(), device=device)
            else:
                mean = self._safe_mean(x).mean()
                std = self._safe_std(x, mean=mean).mean()
                self.qp.mean = mean.to(device)
                self.qp.std = torch.tensor(std, device=device).clamp_min(self.eps)

    @staticmethod
    def _percentiles(x: torch.Tensor, pct: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-dimension percentiles computed in torch.
        x: (N, d)
        Returns (lo, hi) each (d,)
        """
        lo_p, hi_p = pct
        # sort per column and index percentiles
        x_sorted, _ = torch.sort(x, dim=0)
        N = x_sorted.size(0)
        lo_idx = torch.clamp((lo_p / 100.0) * (N - 1), 0, N - 1).long()
        hi_idx = torch.clamp((hi_p / 100.0) * (N - 1), 0, N - 1).long()
        lo = x_sorted[lo_idx, torch.arange(x_sorted.size(1))]
        hi = x_sorted[hi_idx, torch.arange(x_sorted.size(1))]
        return lo, hi

    @staticmethod
    def _safe_mean(x: torch.Tensor) -> torch.Tensor:
        # mean over rows (N,d)
        return x.mean(dim=0)

    @staticmethod
    def _safe_std(x: torch.Tensor, mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mean is None:
            mean = x.mean(dim=0)
        var = ((x - mean) ** 2).mean(dim=0)
        return torch.sqrt(var + 1e-8)

    # ----------------------------- Encoding -------------------------------- #
    @torch.no_grad()
    def encode(self, ts: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize continuous values to token ids.

        Args:
            ts: (B,L,d)

        Returns:
            ids: (B,L,d) int64 in [0, num_bins-1]
            aux: dict with:
                - "nan_mask": (B,L,d) bool where input was NaN
                - "bin_edges": (num_bins+1, d) if per_dim else (num_bins+1,)
        """
        assert ts.ndim == 3, "ts must be (B,L,d)"
        device = ts.device
        B, L, d = ts.shape

        # NaN handling
        nan_mask = torch.isnan(ts)
        x = torch.where(nan_mask, torch.zeros_like(ts), ts)

        # Prepare edges
        if self.method == "uniform":
            vmin, vmax = self._broadcast_range(device, d)
            edges = self._uniform_edges(vmin, vmax, d, device)  # (K+1, d?) or (K+1,)
            # Digitize
            ids = self._digitize(x, edges, right=False, per_dim=self.per_dim)
        else:  # gaussian
            mean, std = self._broadcast_mean_std(device, d)
            z = (x - mean) / (std + self.eps)
            # z-score edges covering ~[-4,4] by default
            edges = self._uniform_edges(
                vmin=torch.full((d,), -4.0, device=device) if self.per_dim else torch.tensor(-4.0, device=device),
                vmax=torch.full((d,),  4.0, device=device) if self.per_dim else torch.tensor( 4.0, device=device),
                d=d,
                device=device,
            )
            ids = self._digitize(z, edges, right=False, per_dim=self.per_dim)

        # clamp to valid range
        ids = ids.clamp_(0, self.num_bins - 1).to(torch.long)

        return ids, {"nan_mask": nan_mask, "bin_edges": edges}

    # ----------------------------- Decoding -------------------------------- #
    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Map token ids back to approximate numeric values via bin centers.

        Args:
            ids: (B,L,d) int64

        Returns:
            recon: (B,L,d) float32
        """
        assert ids.ndim == 3, "ids must be (B,L,d)"
        device = ids.device
        d = ids.shape[-1]

        if self.method == "uniform":
            vmin, vmax = self._broadcast_range(device, d)
            centers = self._uniform_centers(vmin, vmax, d, device)  # (K, d?) or (K,)
            recon = self._gather_centers(ids, centers, per_dim=self.per_dim)
        else:
            mean, std = self._broadcast_mean_std(device, d)
            z_centers = self._uniform_centers(
                vmin=torch.full((d,), -4.0, device=device) if self.per_dim else torch.tensor(-4.0, device=device),
                vmax=torch.full((d,),  4.0, device=device) if self.per_dim else torch.tensor( 4.0, device=device),
                d=d,
                device=device,
            )
            z = self._gather_centers(ids, z_centers, per_dim=self.per_dim)
            recon = z * (std + self.eps) + mean

        return recon

    # -------------------------- Internal helpers --------------------------- #
    def _broadcast_range(self, device: torch.device, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.qp.vmin is not None and self.qp.vmax is not None, "Call .fit() or supply vmin/vmax."
        if self.per_dim:
            assert self.qp.vmin.numel() == d and self.qp.vmax.numel() == d, "per_dim=True needs (d,) vmin/vmax"
            return self.qp.vmin.to(device), self.qp.vmax.to(device)
        else:
            vmin = torch.tensor(float(self.qp.vmin), device=device)
            vmax = torch.tensor(float(self.qp.vmax), device=device)
            return vmin, vmax

    def _broadcast_mean_std(self, device: torch.device, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.qp.mean is not None and self.qp.std is not None, "Call .fit() or supply mean/std."
        if self.per_dim:
            assert self.qp.mean.numel() == d and self.qp.std.numel() == d, "per_dim=True needs (d,) mean/std"
            return self.qp.mean.to(device), self.qp.std.to(device)
        else:
            mean = torch.tensor(float(self.qp.mean), device=device)
            std = torch.tensor(float(self.qp.std), device=device)
            return mean, std

    def _uniform_edges(
        self,
        vmin: torch.Tensor,
        vmax: torch.Tensor,
        d: int,
        device: torch.device,
    ) -> torch.Tensor:
        K = self.num_bins
        if self.per_dim:
            # (K+1, d)
            edges = torch.stack([torch.linspace(vmin[i].item(), vmax[i].item(), K + 1, device=device) for i in range(d)], dim=1)
        else:
            edges = torch.linspace(vmin.item(), vmax.item(), K + 1, device=device)  # (K+1,)
        return edges

    def _uniform_centers(
        self,
        vmin: torch.Tensor,
        vmax: torch.Tensor,
        d: int,
        device: torch.device,
    ) -> torch.Tensor:
        K = self.num_bins
        if self.per_dim:
            centers = torch.stack(
                [torch.linspace(vmin[i].item(), vmax[i].item(), K + 1, device=device) for i in range(d)], dim=1
            )
            centers = 0.5 * (centers[:-1] + centers[1:])  # (K, d)
        else:
            edges = torch.linspace(vmin.item(), vmax.item(), K + 1, device=device)  # (K+1,)
            centers = 0.5 * (edges[:-1] + edges[1:])  # (K,)
        return centers

    @staticmethod
    def _digitize(x: torch.Tensor, edges: torch.Tensor, right: bool, per_dim: bool) -> torch.Tensor:
        """
        Vectorized torch digitize.
        x: (B,L,d)
        edges: (K+1,) or (K+1,d)
        Returns bin indices in [0, K-1].
        """
        B, L, d = x.shape
        if per_dim:
            # Compare each dim separately: expand to (B,L,d,K+1)
            Kp1 = edges.size(0)
            x_exp = x.unsqueeze(-1)                  # (B,L,d,1)
            e_exp = edges.unsqueeze(0).unsqueeze(0)  # (1,1,K+1,d)
            e_exp = e_exp.permute(0, 1, 3, 2)        # (1,1,d,K+1)
            cmp = x_exp >= e_exp if not right else x_exp > e_exp
            idx = cmp.sum(dim=-1) - 1                # (B,L,d) in [-1..K-1]
        else:
            # shared edges
            cmp = (x.unsqueeze(-1) >= edges) if not right else (x.unsqueeze(-1) > edges)  # (B,L,d,K+1)
            idx = cmp.sum(dim=-1) - 1

        return idx.clamp(min=0)

    @staticmethod
    def _gather_centers(ids: torch.Tensor, centers: torch.Tensor, per_dim: bool) -> torch.Tensor:
        """
        ids: (B,L,d)
        centers: (K,) or (K,d)
        """
        if per_dim:
            # gather per-dim
            B, L, d = ids.shape
            K = centers.size(0)
            # Expand centers to (1,1,d,K) to index with ids over last axis
            C = centers.unsqueeze(0).unsqueeze(0)  # (1,1,K,d)
            C = C.permute(0, 1, 3, 2)              # (1,1,d,K)
            idx = ids.clamp(0, K - 1).unsqueeze(-1)  # (B,L,d,1)
            return torch.gather(C.expand(B, L, -1, -1), dim=-1, index=idx).squeeze(-1)
        else:
            # centers shared: (K,)
            K = centers.size(0)
            return centers[ids.clamp(0, K - 1)]

    # ----------------------------- IO utils -------------------------------- #
    def state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {
            "num_bins": torch.tensor(self.num_bins),
            "per_dim": torch.tensor(int(self.per_dim)),
            "method": torch.tensor(0 if self.method == "uniform" else 1),
            "eps": torch.tensor(self.eps),
        }
        for k in ("vmin", "vmax", "mean", "std"):
            v = getattr(self.qp, k)
            if isinstance(v, torch.Tensor):
                sd[k] = v.detach().clone()
        return sd

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.num_bins = int(sd["num_bins"].item())
        self.per_dim = bool(sd["per_dim"].item())
        self.method = "uniform" if int(sd["method"].item()) == 0 else "gaussian"
        self.eps = float(sd["eps"].item())
        # rebuild qp
        self.qp = _QuantParams(num_bins=self.num_bins, per_dim=self.per_dim, method=self.method, eps=self.eps)
        for k in ("vmin", "vmax", "mean", "std"):
            if k in sd:
                setattr(self.qp, k, sd[k].clone())
