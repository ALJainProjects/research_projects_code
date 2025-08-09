#!/usr/bin/env python3
"""
Reference converter for MIMIC-III → CT-GSSN format.

This script DOES NOT fetch MIMIC-III. You must provide pre-extracted, de-identified
tables with schema (configurable via CLI):

    patient_id  |  variable  |  time              |  value
    ------------+------------+--------------------+---------
    100012      |  HR        |  2012-03-01 08:03  |  84.0
    100012      |  HR        |  2012-03-01 08:15  |  86.0
    100012      |  SpO2      |  2012-03-01 08:10  |  96.0
    ...

We build, per patient (or per sliding window of a patient), a dict:
  x      : Tensor (N, L, P)     # impulses at observed times (P=1 by default)
  y      : Tensor (N, L, Q)     # same-shape target (reconstruction by default, Q=1)
  mask   : Tensor (N, L)        # 1 when node observed at that global time index
  deltas : Tensor (L,)          # Δt_k in seconds between consecutive global times
  adj    : Tensor (N, N)        # static normalized adjacency (or dynamic if provided)
  meta   : dict                 # patient_id, node_names, time_index strings, etc.

We then save a list[dict] via torch.save(...), one file per split:
  data/mimic3_proc/train.pt
  data/mimic3_proc/val.pt
  data/mimic3_proc/test.pt

Usage examples
--------------
1) Synthetic smoke test:
   python scripts/convert_mimic3_ctgssn.py --out_root data/mimic3_proc --synthesize

2) Real conversion from CSV globs (with z-score and kNN adjacency):
   python scripts/convert_mimic3_ctgssn.py \
     --in_glob "data/mimic3_flat/train/*.csv" \
     --split train \
     --out_root data/mimic3_proc \
     --id_col patient_id --var_col variable --time_col time --val_col value \
     --top_n_nodes 24 --zscore --knn_k 8

3) From Parquet/JSONL, sliding windows, and static adjacency from CSV edgelist:
   python scripts/convert_mimic3_ctgssn.py \
     --in_glob "raw/train/*.parquet" \
     --split train --out_root data/mimic3_proc \
     --id_col pid --var_col label --time_col ts --val_col val \
     --window_L 288 --window_stride 144 \
     --adj_csv "graphs/var_edges.csv" --adj_src_col src --adj_dst_col dst

Notes
-----
- We do **not** impute at unobserved times; x stays zero and mask=0.
- We align on the **union** of timestamps across a patient's chosen nodes to preserve irregularity.
- For reproducibility, optionally z-score normalize **per node** using only observed values in the window.
- y defaults to the same signal as x (reconstruction). If you want a different target,
  e.g., future forecasting labels aligned to the same global grid, adapt the `build_y()` function.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch


# ----------------------------- Adjacency helpers ----------------------------- #

def load_adjacency_from_csv(
    nodes: List[str],
    csv_path: Path,
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: Optional[str] = None,
    symmetric: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Load a static adjacency from an edge list CSV and map it to current node order."""
    df = pd.read_csv(csv_path)
    idx_of = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)

    for _, r in df.iterrows():
        s, d = r[src_col], r[dst_col]
        if s not in idx_of or d not in idx_of:
            continue
        w = float(r[weight_col]) if weight_col and weight_col in r else 1.0
        A[idx_of[s], idx_of[d]] = w
        if symmetric:
            A[idx_of[d], idx_of[s]] = w

    if normalize:
        deg = A.sum(axis=1, keepdims=True).clip(min=1.0)
        A = A / deg
    np.fill_diagonal(A, 0.0)
    return A


def load_adjacency_from_npy(nodes: List[str], npy_path: Path, normalize: bool = True) -> np.ndarray:
    """Load a precomputed adjacency matrix and (optionally) row-normalize."""
    A = np.load(npy_path).astype(np.float32)
    if A.shape[0] != len(nodes) or A.shape[1] != len(nodes):
        raise ValueError(f"Adj shape {A.shape} != ({len(nodes)},{len(nodes)}). "
                         "Ensure node order matches or map properly before saving .npy.")
    if normalize:
        deg = A.sum(axis=1, keepdims=True).clip(min=1.0)
        A = A / deg
    np.fill_diagonal(A, 0.0)
    return A


def build_knn_adjacency(node_feats: np.ndarray, k: int = 8) -> np.ndarray:
    """Simple kNN graph (symmetric, row-normalized) over node feature vectors."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("scikit-learn is required for --knn_k > 0 adjacency building.") from e

    N = node_feats.shape[0]
    k = min(max(1, k), max(1, N - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(node_feats)
    graph = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        _, idxs = nbrs.kneighbors(node_feats[i:i + 1])
        for j in idxs[0]:
            if j != i:
                graph[i, j] = 1.0
    graph = ((graph + graph.T) > 0).astype(np.float32)
    deg = graph.sum(axis=1, keepdims=True).clip(min=1.0)
    graph = graph / deg
    np.fill_diagonal(graph, 0.0)
    return graph


# ----------------------------- Core conversion ----------------------------- #

@dataclass
class Columns:
    id_col: str
    var_col: str
    time_col: str
    val_col: str


def read_any(path: Path) -> pd.DataFrame:
    """Read CSV / Parquet / JSONL into a DataFrame."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(pd.json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file type: {path}")


def load_concat(glob_expr: str) -> pd.DataFrame:
    """Load and concatenate a glob of files."""
    paths = sorted(Path().glob(glob_expr))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_expr}")
    dfs = []
    for p in paths:
        df = read_any(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def pick_top_nodes(df: pd.DataFrame, col_var: str, top_n: Optional[int]) -> List[str]:
    """Select the top-N variables by observation count; else return all variables."""
    counts = df[col_var].value_counts()
    if top_n is None or top_n <= 0 or top_n >= len(counts):
        return list(counts.index)
    return list(counts.index[:top_n])


def build_global_grid(times: List[pd.Timestamp]) -> np.ndarray:
    """Build global sorted unique time grid (as numpy datetime64[ns])."""
    t = pd.to_datetime(times)
    t = np.sort(t.unique())
    return t.astype("datetime64[ns]")


def align_patient_window(
    df_p: pd.DataFrame,
    cols: Columns,
    node_names: List[str],
    zscore: bool,
    min_obs_per_node: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build x, y, mask, deltas tensors for a single patient window.

    Returns:
      x      (N,L,1)   impulses at observed times
      y      (N,L,1)   reconstruction target (same as x)
      mask   (N,L)     1 where observed
      deltas (L,)      Δt in seconds
      grid_str (List[str])  ISO timestamps for reference (meta)
    """
    # Filter nodes with too few observations
    obs_counts = df_p.groupby(cols.var_col)[cols.val_col].count()
    ok_nodes = [n for n in node_names if obs_counts.get(n, 0) >= min_obs_per_node]
    if len(ok_nodes) == 0:
        return None, None, None, None, []

    # Global time grid = union of all timestamps across chosen nodes
    grid = build_global_grid(df_p[cols.time_col].tolist())
    L = len(grid)
    N = len(ok_nodes)

    # Prepare outputs
    x = np.zeros((N, L, 1), dtype=np.float32)
    y = np.zeros((N, L, 1), dtype=np.float32)
    mask = np.zeros((N, L), dtype=np.float32)

    # Map time→index for grid lookup
    t2i = {int(ts): i for i, ts in enumerate(grid.view("int64"))}

    # Fill per node
    for n_idx, n in enumerate(ok_nodes):
        sub = df_p[df_p[cols.var_col] == n]
        # ensure ts sorted
        t = pd.to_datetime(sub[cols.time_col]).values.astype("datetime64[ns]").astype("int64")
        v = sub[cols.val_col].astype(float).values
        # place impulses
        for ti, vi in zip(t, v):
            i = t2i.get(int(ti))
            if i is None:
                continue
            x[n_idx, i, 0] = float(vi)
            y[n_idx, i, 0] = float(vi)
            mask[n_idx, i] = 1.0

        if zscore:
            # z-score over observed entries only
            m = mask[n_idx] > 0
            if m.any():
                mu = x[n_idx, m, 0].mean()
                sd = x[n_idx, m, 0].std() + 1e-8
                x[n_idx, m, 0] = (x[n_idx, m, 0] - mu) / sd
                y[n_idx, m, 0] = (y[n_idx, m, 0] - mu) / sd

    # deltas in seconds
    if L == 0:
        return None, None, None, None, []
    times_ns = grid.astype("int64")
    deltas = np.zeros((L,), dtype=np.float32)
    deltas[0] = max(1.0, (times_ns[0] - times_ns[0]) / 1e9)
    if L > 1:
        dt = np.diff(times_ns) / 1e9  # seconds
        deltas[1:] = np.maximum(1.0, dt.astype(np.float32))  # clamp to >=1s to avoid zeros

    return x, y, mask, deltas, [pd.Timestamp(ts).isoformat() for ts in grid]


def node_feature_embedding(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Make a tiny feature vector per node from observed samples (mean, std, #obs, sparsity).
    """
    N, L, _ = x.shape
    feats = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        m = mask[i] > 0
        obs = x[i, m, 0]
        cnt = float(m.sum())
        if cnt > 0:
            feats[i, 0] = float(obs.mean())
            feats[i, 1] = float(obs.std())
        feats[i, 2] = cnt
        feats[i, 3] = 1.0 - (cnt / max(1.0, L))
    return feats


def build_static_adjacency(
    node_names: List[str],
    args: argparse.Namespace,
    x: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Decide how to build the adjacency for this sample."""
    if args.adj_npy:
        return load_adjacency_from_npy(node_names, Path(args.adj_npy), normalize=True)

    if args.adj_csv:
        return load_adjacency_from_csv(
            node_names,
            Path(args.adj_csv),
            src_col=args.adj_src_col,
            dst_col=args.adj_dst_col,
            weight_col=args.adj_weight_col,
            symmetric=True,
            normalize=True,
        )

    # Else: build kNN from simple node embeddings
    if args.knn_k > 0:
        feats = node_feature_embedding(x, mask) if x is not None else np.random.randn(len(node_names), 4).astype(np.float32)
        return build_knn_adjacency(feats, k=args.knn_k)

    # Fallback: fully disconnected (model will still learn)
    N = len(node_names)
    return np.zeros((N, N), dtype=np.float32)


def slice_windows(
    df_p: pd.DataFrame,
    cols: Columns,
    node_names_all: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Emit one or more windows for the patient depending on windowing args.
    If no windowing is provided, emit a single sample spanning the full union grid.
    """
    out = []

    # If no sliding windows requested, just do one pass
    if args.window_L <= 0:
        x, y, mask, deltas, grid_str = align_patient_window(
            df_p, cols, node_names_all, args.zscore, args.min_obs_per_node
        )
        if x is None:
            return out
        # Build adjacency once per sample
        node_names = [n for n in node_names_all if n in df_p[cols.var_col].unique()]
        adj = build_static_adjacency(node_names, args, x, mask)
        out.append({"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj, "node_names": node_names, "grid": grid_str})
        return out

    # Sliding windows: we create windows over the *time grid indices*, not timestamps directly.
    # Strategy: build full grid first; then split by index spans.
    x_full, y_full, mask_full, deltas_full, grid_str = align_patient_window(
        df_p, cols, node_names_all, args.zscore, args.min_obs_per_node
    )
    if x_full is None:
        return out

    N, L, _ = x_full.shape
    stride = max(1, args.window_stride)
    W = args.window_L
    for start in range(0, max(1, L - W + 1), stride):
        end = start + W
        if end > L:
            break
        x = x_full[:, start:end, :]
        y = y_full[:, start:end, :]
        mask = mask_full[:, start:end]
        deltas = deltas_full[start:end]
        # Normalize per window (if requested) will already have been applied in align_patient_window
        node_names = [n for n in node_names_all if n in df_p[cols.var_col].unique()]
        adj = build_static_adjacency(node_names, args, x, mask)
        out.append({"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj, "node_names": node_names,
                    "grid": grid_str[start:end]})
    return out


# ----------------------------- Synthetic example ----------------------------- #

def example_patient(N=12, L=48, P=1, Q=1, seed: int = 1337) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, L, P)).astype(np.float32)
    y = x.copy()  # reconstruction
    mask = (rng.random(size=(N, L)) < 0.6).astype(np.float32)
    deltas = (rng.random(size=(L,)) * 0.6 + 0.1).astype(np.float32)
    adj = build_knn_adjacency(rng.normal(size=(N, 4)).astype(np.float32), k=4)
    return {"x": torch.tensor(x), "y": torch.tensor(y), "mask": torch.tensor(mask),
            "deltas": torch.tensor(deltas), "adj": torch.tensor(adj),
            "meta": {"patient_id": "SYNTHETIC", "node_names": [f"v{i}" for i in range(N)]}}


# ----------------------------- CLI & main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data/mimic3_proc", help="Output directory for split .pt files")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train", help="Which split to emit")

    # Input table(s)
    ap.add_argument("--in_glob", type=str, default=None,
                    help="Glob of CSV/Parquet/JSONL files with columns [id,var,time,val] (names configurable).")
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--var_col", default="variable")
    ap.add_argument("--time_col", default="time")
    ap.add_argument("--val_col", default="value")

    # Node selection & normalization
    ap.add_argument("--top_n_nodes", type=int, default=0, help="Pick top-N most frequent variables (0=all).")
    ap.add_argument("--min_obs_per_node", type=int, default=2, help="Drop nodes with < this many observations")
    ap.add_argument("--zscore", action="store_true", help="Z-score per node over observed entries")

    # Sliding windows (optional)
    ap.add_argument("--window_L", type=int, default=0, help="Window length in grid steps (0 = full sequence)")
    ap.add_argument("--window_stride", type=int, default=0, help="Stride in grid steps if window_L > 0")

    # Adjacency options
    ap.add_argument("--adj_csv", type=str, default=None, help="CSV edge list path for static adjacency")
    ap.add_argument("--adj_npy", type=str, default=None, help="NPY matrix path for static adjacency")
    ap.add_argument("--adj_src_col", type=str, default="src")
    ap.add_argument("--adj_dst_col", type=str, default="dst")
    ap.add_argument("--adj_weight_col", type=str, default=None)
    ap.add_argument("--knn_k", type=int, default=8, help="If no adj provided, build kNN over node feature vectors")

    # Misc
    ap.add_argument("--limit_patients", type=int, default=0, help="Debug: cap patients processed")
    ap.add_argument("--synthesize", action="store_true", help="Write a synthetic split instead of real conversion")

    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{args.split}.pt"

    if args.synthesize:
        count = 32 if args.split == "train" else 8
        patients = [example_patient(seed=1337 + i) for i in range(count)]
        torch.save(patients, out_path)
        print(f"[ok] Wrote SYNTHETIC {len(patients)} samples → {out_path}")
        return

    if not args.in_glob:
        raise SystemExit("Provide --in_glob with your pre-extracted tables (CSV/Parquet/JSONL).")

    # Load and standardize columns
    df = load_concat(args.in_glob)
    cols = Columns(id_col=args.id_col, var_col=args.var_col, time_col=args.time_col, val_col=args.val_col)

    # Basic sanity checks / conversions
    if cols.time_col not in df.columns:
        raise ValueError(f"Missing time column '{cols.time_col}' in input.")
    df[cols.time_col] = pd.to_datetime(df[cols.time_col])

    missing = [c for c in [cols.id_col, cols.var_col, cols.val_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Choose node vocabulary for this split
    node_vocab = pick_top_nodes(df, cols.var_col, args.top_n_nodes)
    print(f"[info] Using {len(node_vocab)} variables (top_n_nodes={args.top_n_nodes or 'all'})")

    # Iterate patients
    patients_out: List[Dict[str, Any]] = []
    unique_pids = df[cols.id_col].dropna().unique().tolist()
    if args.limit_patients > 0:
        unique_pids = unique_pids[:args.limit_patients]

    for pi, pid in enumerate(unique_pids, start=1):
        df_p = df[df[cols.id_col] == pid].copy()
        df_p = df_p[df_p[cols.var_col].isin(node_vocab)]
        if df_p.empty:
            continue

        samples = slice_windows(df_p, cols, node_vocab, args)
        if not samples:
            continue

        for s in samples:
            # Assemble final dict with tensors
            x = torch.tensor(s["x"])
            y = torch.tensor(s["y"])
            mask = torch.tensor(s["mask"])
            deltas = torch.tensor(s["deltas"])
            adj = torch.tensor(s["adj"])
            meta = {
                "patient_id": str(pid),
                "node_names": s["node_names"],
                "time_index": s["grid"],
            }
            patients_out.append({"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj, "meta": meta})

        if pi % 50 == 0:
            print(f"[info] processed {pi}/{len(unique_pids)} patients...")

    torch.save(patients_out, out_path)
    print(f"[ok] Wrote {len(patients_out)} samples → {out_path}")


if __name__ == "__main__":
    main()
