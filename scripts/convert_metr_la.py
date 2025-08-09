#!/usr/bin/env python3
"""
METR-LA → CT-GSSN format (production-grade).

Outputs:
  data/metr-la_proc/{train,val,test}.pt

Each file is a torch-saved list[dict] with:
  x: (N,L,1)          # observed speeds per sensor per time index (no imputation)
  y: (N,L,1)          # next-step (or horizon) labels aligned to same length (shifted)
  mask: (N,L)         # 1 where x is observed (non-NaN), 0 otherwise
  deltas: (L,)        # Δt between consecutive indices (seconds)
  adj: (N,N)          # row-normalized static adjacency
  meta: dict          # sensor_ids, window_start, indices, etc.

Features
--------
- Load official METR-LA speed arrays from .npz (key "speed") or .h5 ("speed")
- NaN handling → masks; no filling; x stays 0 where mask==0
- Z-score per sensor using *training split only*; saves stats to JSON for reuse
- Adjacency:
  * --adj_npy path/to/adj.npy
  * --adj_csv CSV edge list (src,dst[,w])
  * --dist_npy distances.npy with --dist_thresh to build adjacency
  * --coords_npy coords.npy + --knn_k to build kNN adjacency
  * (row-normalized, zero diagonal; symmetric where applicable)
- Windowing: --L, --stride, --horizon (label shift); Same-length y is created by shifting x by H
- Splits:
  * default temporal split 8:1:1 by index
  * or provide --split_idx_npy with arrays "train", "val", "test" of start indices

Usage
-----
# Fast synth smoke test
python scripts/convert_metrla_ctgssn.py --synthesize --out_root data/metr-la_proc

# Real conversion
python scripts/convert_metrla_ctgssn.py \
  --speed_npz data/METR-LA/speed.npz \
  --adj_npy data/METR-LA/adj.npy \
  --L 288 --stride 144 --horizon 1 \
  --zscore --out_root data/metr-la_proc

# Using distances->adjacency
python scripts/convert_metrla_ctgssn.py \
  --speed_npz data/METR-LA/speed.npz \
  --dist_npy data/METR-LA/distances.npy --dist_thresh 0.1 \
  --L 288 --stride 144 --horizon 3 --zscore --out_root data/metr-la_proc
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


# ------------------------- IO helpers ------------------------- #

def load_speed_array(speed_npz: Optional[str], speed_h5: Optional[str]) -> np.ndarray:
    """Load METR-LA speeds into shape [T, N] (float32)."""
    if speed_npz:
        arr = np.load(speed_npz)
        # Support either direct array (.npz with single unnamed array) or a dict with 'speed'
        if hasattr(arr, "files") and "speed" in arr.files:
            x = arr["speed"].astype(np.float32)  # [T,N]
        elif hasattr(arr, "files") and len(arr.files) == 1:
            x = arr[arr.files[0]].astype(np.float32)
        else:
            # Could be np.savez of a plain ndarray with default key 'arr_0'
            x = arr.get("arr_0")
            if x is None:
                raise ValueError("speed_npz must contain array key 'speed' or single array.")
            x = x.astype(np.float32)
        return x
    if speed_h5:
        import h5py
        with h5py.File(speed_h5, "r") as f:
            x = f["speed"][:].astype(np.float32)
        return x
    raise ValueError("Provide --speed_npz or --speed_h5 (or use --synthesize).")


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def row_normalize(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1, keepdims=True).clip(min=1.0)
    A = A / deg
    np.fill_diagonal(A, 0.0)
    return A


# ------------------------- adjacency builders ------------------------- #

def load_adjacency_from_npy(path: Path) -> np.ndarray:
    return row_normalize(np.load(path).astype(np.float32))


def load_adjacency_from_csv(path: Path, src_col="src", dst_col="dst", w_col: Optional[str] = None, N: Optional[int] = None) -> np.ndarray:
    import csv
    # If N unknown, we'll infer max id + 1
    max_id = -1
    edges = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            s = int(r[src_col]); d = int(r[dst_col])
            w = float(r[w_col]) if w_col and r.get(w_col) is not None else 1.0
            edges.append((s, d, w))
            max_id = max(max_id, s, d)
    if N is None:
        N = max_id + 1
    A = np.zeros((N, N), dtype=np.float32)
    for s, d, w in edges:
        if s < N and d < N:
            A[s, d] = max(A[s, d], w)
            A[d, s] = max(A[d, s], w)
    return row_normalize(A)


def adjacency_from_distances(dist_npy: Path, thresh: float) -> np.ndarray:
    """Distances to adjacency by threshold; symmetric row-normalized."""
    D = np.load(dist_npy).astype(np.float32)  # [N,N] distances (larger=farther)
    A = (D <= thresh).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    A = ((A + A.T) > 0).astype(np.float32)
    return row_normalize(A)


def adjacency_from_coords(coords_npy: Path, knn_k: int) -> np.ndarray:
    """Build kNN over coordinates [N,2] using Euclidean distance."""
    coords = np.load(coords_npy).astype(np.float32)
    N = coords.shape[0]
    if knn_k <= 0 or knn_k >= N:
        raise ValueError("--knn_k must be in [1, N-1]")
    # pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    # kNN (exclude self)
    idxs = np.argsort(dist, axis=1)[:, 1:knn_k + 1]
    A = np.zeros((N, N), dtype=np.float32)
    rows = np.arange(N)[:, None]
    A[rows, idxs] = 1.0
    A = ((A + A.T) > 0).astype(np.float32)
    return row_normalize(A)


def decide_adjacency(
    N: int,
    args: argparse.Namespace,
    fallback_dense: bool = False,
) -> np.ndarray:
    if args.adj_npy:
        return load_adjacency_from_npy(Path(args.adj_npy))
    if args.adj_csv:
        return load_adjacency_from_csv(Path(args.adj_csv), src_col=args.adj_src_col, dst_col=args.adj_dst_col, w_col=args.adj_w_col, N=N)
    if args.dist_npy and args.dist_thresh is not None:
        return adjacency_from_distances(Path(args.dist_npy), float(args.dist_thresh))
    if args.coords_npy and args.knn_k > 0:
        return adjacency_from_coords(Path(args.coords_npy), args.knn_k)

    if fallback_dense:
        A = np.ones((N, N), dtype=np.float32)
        np.fill_diagonal(A, 0.0)
        return row_normalize(A)
    # default empty (model can still learn)
    return np.zeros((N, N), dtype=np.float32)


# ------------------------- normalization ------------------------- #

def compute_train_stats(train_windows: List[np.ndarray], train_masks: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Per-sensor mean/std over observed values only, computed on training windows.
    Returns dict with 'mu' and 'sigma' arrays of shape [N].
    """
    # Stack [B,N,L,1] -> combine observed entries
    N = train_windows[0].shape[0]
    obs_vals = [[] for _ in range(N)]
    for x, m in zip(train_windows, train_masks):
        # x: (N,L,1), m: (N,L)
        for i in range(N):
            v = x[i, m[i] > 0, 0]
            if v.size > 0:
                obs_vals[i].append(v)
    mu = np.zeros((N,), dtype=np.float32)
    sd = np.ones((N,), dtype=np.float32)
    for i in range(N):
        if len(obs_vals[i]) > 0:
            cat = np.concatenate(obs_vals[i], axis=0)
            mu[i] = cat.mean().astype(np.float32)
            sd[i] = cat.std().astype(np.float32) + 1e-6
        else:
            mu[i] = 0.0
            sd[i] = 1.0
    return {"mu": mu, "sigma": sd}


def apply_zscore(x: np.ndarray, mask: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """
    In-place zscore where mask==1. x shape (N,L,1).
    """
    N, L, _ = x.shape
    for i in range(N):
        m = mask[i] > 0
        x[i, m, 0] = (x[i, m, 0] - mu[i]) / sigma[i]


# ------------------------- windowing & labeling ------------------------- #

def make_windows(speed: np.ndarray, L: int, stride: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (seg, nxt) where:
      seg : [L, N]
      nxt : [L, N] shifted by +1 for next-step baseline; later we can generalize with horizon.
    """
    T, N = speed.shape
    outputs = []
    for start in range(0, T - L - 1, stride):
        seg = speed[start:start + L]            # [L,N]
        nxt = speed[start + 1:start + L + 1]    # [L,N]
        outputs.append((seg, nxt))
    return outputs


def shift_with_horizon(seg: np.ndarray, horizon: int) -> np.ndarray:
    """
    Given seg [L,N], produce labels y [L,N] shifted by +horizon.
    If beyond the end, last few are copied from the last available (keeps shape).
    """
    L, N = seg.shape
    y = np.empty_like(seg)
    if horizon <= 0:
        return seg.copy()
    if horizon >= L:
        # degenerate: copy last frame as labels
        y[:] = seg[-1]
        return y
    y[:L - horizon] = seg[horizon:]
    y[L - horizon:] = seg[-1]  # repeat last available
    return y


# ------------------------- main conversion ------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed_npz", type=str, default=None, help="path to METR-LA speed .npz (key 'speed')")
    ap.add_argument("--speed_h5", type=str, default=None, help="alternate: h5 with dataset 'speed'")
    ap.add_argument("--out_root", default="data/metr-la_proc")

    # Windowing / labeling
    ap.add_argument("--L", type=int, default=288, help="window length (time steps)")
    ap.add_argument("--stride", type=int, default=144, help="stride between windows")
    ap.add_argument("--horizon", type=int, default=1, help="label shift (steps). y is seg shifted by +H, same length.")

    # Adjacency options
    ap.add_argument("--adj_npy", type=str, default=None)
    ap.add_argument("--adj_csv", type=str, default=None)
    ap.add_argument("--adj_src_col", type=str, default="src")
    ap.add_argument("--adj_dst_col", type=str, default="dst")
    ap.add_argument("--adj_w_col", type=str, default=None)
    ap.add_argument("--dist_npy", type=str, default=None, help="distances.npy")
    ap.add_argument("--dist_thresh", type=float, default=None, help="threshold for distance->adjacency")
    ap.add_argument("--coords_npy", type=str, default=None, help="coords.npy [N,2]")
    ap.add_argument("--knn_k", type=int, default=0, help="if coords provided, build kNN graph")

    # Normalization
    ap.add_argument("--zscore", action="store_true", help="z-score per sensor using training stats")

    # Splitting
    ap.add_argument("--split_idx_npy", type=str, default=None, help="optional .npy with keys: train,val,test (start indices)")
    ap.add_argument("--temporal_ratio", type=str, default="0.8,0.1,0.1", help="temporal split ratios if no index file")

    # Misc
    ap.add_argument("--synthesize", action="store_true", help="generate synthetic METR-LA-like data")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load or synthesize speeds
    if args.synthesize:
        N = 207
        T = 12_000
        rng = np.random.default_rng(1337)
        base = rng.normal(loc=45.0, scale=10.0, size=(T, N)).astype(np.float32)
        # inject seasonal-ish pattern
        t = np.arange(T, dtype=np.float32)[:, None]
        season = 5.0 * np.sin(2 * np.pi * t / 288.0)
        speed = (base + season).astype(np.float32)  # [T,N]
        # introduce missing
        miss = rng.random(size=speed.shape) < 0.02
        speed[miss] = np.nan
    else:
        speed = load_speed_array(args.speed_npz, args.speed_h5)  # [T,N]

    T, N = speed.shape

    # Make windows (segments) and horizon-shifted targets
    raw_windows = make_windows(speed, args.L, args.stride)  # list of (seg[nan], nxt[nan])
    # Build adjacency once (static)
    A = decide_adjacency(N, args, fallback_dense=False)

    # Build dataset entries per window
    deltas = np.ones((args.L,), dtype=np.float32) * 300.0  # 5 minutes per step = 300 sec
    Xs: List[np.ndarray] = []
    Ms: List[np.ndarray] = []

    def nan_to_zero_with_mask(arr_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """arr_2d: [L,N] -> returns (x[N,L,1], mask[N,L]) with zeros where NaN and mask=0."""
        L_, N_ = arr_2d.shape
        arr_T = arr_2d.T  # [N,L]
        mask = (~np.isnan(arr_T)).astype(np.float32)
        x = np.nan_to_num(arr_T, nan=0.0).astype(np.float32)[..., None]  # [N,L,1]
        return x, mask

    entries: List[dict] = []
    for wi, (seg, _) in enumerate(raw_windows):
        # y uses horizon shift over seg (same length)
        y_seg = shift_with_horizon(seg, args.horizon)  # [L,N]
        x, mask = nan_to_zero_with_mask(seg)           # (N,L,1), (N,L)
        y, _ = nan_to_zero_with_mask(y_seg)

        entries.append({
            "x": x, "y": y, "mask": mask,
            "deltas": deltas.copy(),
            "adj": A,
            "meta": {"window_index": wi}
        })
        Xs.append(x); Ms.append(mask)

    # Split indices
    if args.split_idx_npy:
        idxs = np.load(args.split_idx_npy, allow_pickle=True).item()
        split_indices = {
            "train": list(map(int, idxs["train"])),
            "val": list(map(int, idxs["val"])),
            "test": list(map(int, idxs["test"])),
        }
    else:
        r_train, r_val, r_test = [float(x) for x in args.temporal_ratio.split(",")]
        r_sum = r_train + r_val + r_test
        r_train, r_val, r_test = r_train / r_sum, r_val / r_sum, r_test / r_sum
        n = len(entries)
        n_tr, n_va = int(n * r_train), int(n * (r_train + r_val))
        split_indices = {
            "train": list(range(0, n_tr)),
            "val": list(range(n_tr, n_va)),
            "test": list(range(n_va, n)),
        }

    # Compute z-score from training only, apply to all (where mask==1)
    stats = None
    if args.zscore:
        train_windows = [entries[i]["x"].copy() for i in split_indices["train"]]
        train_masks = [entries[i]["mask"].copy() for i in split_indices["train"]]
        stats = compute_train_stats(train_windows, train_masks)  # mu,sigma [N]
        for i in range(len(entries)):
            apply_zscore(entries[i]["x"], entries[i]["mask"], stats["mu"], stats["sigma"])
            apply_zscore(entries[i]["y"], entries[i]["mask"], stats["mu"], stats["sigma"])

        # Save stats for inference use
        save_json(out_root / "zscore_stats.json", {"mu": stats["mu"].tolist(), "sigma": stats["sigma"].tolist()})

    # Convert numpy entries to tensors and dump per split
    splits = {}
    for split in ("train", "val", "test"):
        items = []
        for idx in split_indices[split]:
            e = entries[idx]
            items.append({
                "x": torch.tensor(e["x"]),
                "y": torch.tensor(e["y"]),
                "mask": torch.tensor(e["mask"]),
                "deltas": torch.tensor(e["deltas"]),
                "adj": torch.tensor(e["adj"]),
                "meta": e["meta"],
            })
        splits[split] = items

    for split, items in splits.items():
        path = out_root / f"{split}.pt"
        torch.save(items, path)
        print(f"[ok] wrote {len(items):,} windows → {path}")

    # A tiny sanity print
    if splits["train"]:
        e0 = splits["train"][0]
        print("[shape] x:", tuple(e0["x"].shape),
              "y:", tuple(e0["y"].shape),
              "mask:", tuple(e0["mask"].shape),
              "deltas:", tuple(e0["deltas"].shape),
              "adj:", tuple(e0["adj"].shape))


if __name__ == "__main__":
    main()
