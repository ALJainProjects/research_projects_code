#!/usr/bin/env python3
"""
Time-MMD → CEX-TSLM format (JSONL with ts/docs/target).

This script converts *real datasets* into the JSONL schema used by CEX-TSLM:
    { "ts": [[...],[...],...], "docs": ["...","..."], "target": [[...],...], "pos_idx": [int,int,...] }

It also supports a synthetic mode for quick smoke tests:
    --synthesize  → writes small toy train/val/test JSONL splits that conform to the same schema.

Supported inputs
----------------
Time series:
  - CSV or JSONL with a single file per symbol/series (or many files found via --ts_glob)
  - Must include a timestamp column and one or more numeric value columns (multi-d allowed)
  - Example CSV columns: timestamp, open, high, low, close  (choose --ts_value_cols)
  - Example JSONL: {"time": "...", "x": [v1, v2, ...]} or flat columns like {"time": "...", "value": 1.23}

Documents:
  - CSV or JSONL with fields for time and text (configurable column names)
  - Or a directory of .txt files; timestamp is parsed from filename if possible; otherwise sequence order
  - Selects documents within [window_start - doc_lead, window_end + doc_lag]
  - Picks up to --max_docs (closest by time first); optional keyword ranking

Positive indices (pos_idx):
  - By default: time proximity from the last quarter of the input window (recent events)
  - Optionally: boost/rank documents by keyword hits (if --keyword_file provided)
  - The chosen doc indices (within the selected doc list) are returned as pos_idx

Core windowing:
  - Resamples time series to a fixed frequency (--freq) and builds sliding windows:
      ts: length L, target: horizon H
  - Interpolates missing samples
  - Optional z-score normalize (per series / window or globally)

Examples
--------
1) Synthetic quickstart:
    python scripts/convert_time_mmd.py --root_out data/time_mmd_proc --synthesize

2) CSV time-series + JSONL news (daily resample, 96-step lookback, 8-step horizon):
    python scripts/convert_time_mmd.py \
      --root_in data/time_mmd_raw \
      --root_out data/time_mmd_proc \
      --split_train_glob "data/time_mmd_raw/ts/train/*.csv" \
      --split_val_glob   "data/time_mmd_raw/ts/val/*.csv" \
      --split_test_glob  "data/time_mmd_raw/ts/test/*.csv" \
      --docs_glob "data/time_mmd_raw/news/**/*.jsonl" \
      --ts_time_col timestamp \
      --ts_value_cols close,volume \
      --doc_time_col time \
      --doc_text_col text \
      --freq D --L 96 --H 8 --stride 8 --max_docs 24 \
      --doc_lead 3D --doc_lag 1D \
      --zscore

3) JSONL time-series & CSV docs, hourly:
    python scripts/convert_time_mmd.py \
      --split_train_glob "raw/ts/train/*.jsonl" \
      --docs_glob "raw/docs/*.csv" \
      --ts_jsonl_value_field x --ts_time_col time \
      --doc_time_col published_at --doc_text_col body \
      --freq H --L 168 --H 24

Notes
-----
- This script is intentionally flexible because "Time-MMD" datasets in the wild vary.
- You control globs and column mappings; the script assembles windows and aligns docs.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------- Synthetic data ------------------------- #

def synthesize(n=128, L=96, d=4, H=8, max_docs=24, seed: int = 1337):
    """
    Create a toy dataset matching the expected CEX-TSLM schema.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ts = rng.normal(size=(L, d)).tolist()
        target = rng.normal(size=(H, d)).tolist()
        m = rng.integers(4, max_docs)
        docs = [f"doc {i}: some event happened" for i in range(m)]
        # pick 1..L/8 random positions as positives
        pos_cnt = max(1, L // 8)
        pos_idx = list(rng.integers(0, m, size=pos_cnt))
        rows.append({"ts": ts, "docs": docs, "target": target, "pos_idx": pos_idx})
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------- IO helpers ------------------------- #

def parse_timedelta(s: str) -> pd.Timedelta:
    """
    Accepts shortcuts like "3D", "12H", "30min", "0", "".
    """
    if not s:
        return pd.Timedelta(0)
    try:
        return pd.to_timedelta(s)
    except Exception:
        # last-ditch: if just an int, assume days
        try:
            return pd.Timedelta(int(s), unit="D")
        except Exception:
            raise


def read_timeseries_any(path: Path,
                        time_col: str,
                        value_cols: List[str],
                        is_jsonl: bool = False,
                        jsonl_value_field: Optional[str] = None) -> pd.DataFrame:
    """
    Load a time series file (CSV or JSONL) into a DataFrame with DateTimeIndex and numeric columns.

    - CSV: expects columns [time_col, *value_cols]
    - JSONL:
        * if jsonl_value_field is provided and is a list-like numeric field, read from it
        * else expects flat numeric columns with names in value_cols
    """
    if is_jsonl or path.suffix.lower() == ".jsonl":
        recs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = pd.to_datetime(obj[time_col])
                if jsonl_value_field and jsonl_value_field in obj:
                    vals = obj[jsonl_value_field]
                    if not isinstance(vals, (list, tuple)):
                        raise ValueError(f"{path}: jsonl_value_field={jsonl_value_field} is not list-like")
                    recs.append({"__t": t, **{f"x{i}": float(v) for i, v in enumerate(vals)}})
                else:
                    rec = {"__t": t}
                    for col in value_cols:
                        rec[col] = float(obj[col])
                    recs.append(rec)
        df = pd.DataFrame.from_records(recs).set_index("__t").sort_index()
        # if using jsonl_value_field, rename x0.. to canonical names
        if jsonl_value_field:
            # infer dimension count
            dim = len([c for c in df.columns if c.startswith("x")])
            df = df[[f"x{i}" for i in range(dim)]]
            df.columns = [f"v{i}" for i in range(dim)]
        else:
            df = df[value_cols]
        return df
    else:
        df = pd.read_csv(path)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        df = df[value_cols]
        return df


def read_docs_any(paths: List[Path],
                  time_col: Optional[str],
                  text_col: Optional[str]) -> pd.DataFrame:
    """
    Load documents across many files (CSV/JSONL/txt directory) into a single DataFrame:
        index: datetime (if available), columns: ['text', '__src', '__time'].

    If text files have no timestamp, we keep NaT and later fallback to ranking by file order.
    """
    rows = []
    for p in paths:
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    t = pd.to_datetime(obj[time_col]) if time_col and time_col in obj else pd.NaT
                    txt = str(obj[text_col]) if text_col and text_col in obj else json.dumps(obj)
                    rows.append({"__time": t, "text": txt, "__src": str(p)})
        elif p.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(p)
            t = pd.to_datetime(df[time_col]) if time_col and time_col in df.columns else pd.NaT
            if isinstance(t, pd.Series):
                t = t
            else:
                t = pd.Series([t] * len(df))
            if text_col and text_col in df.columns:
                txts = df[text_col].astype(str)
            else:
                # try to concatenate all non-time columns
                txts = df.drop(columns=[c for c in [time_col] if c in df.columns], errors="ignore").astype(str).agg(" ".join, axis=1)
            for ti, tx in zip(t, txts):
                rows.append({"__time": ti, "text": tx, "__src": str(p)})
        else:
            # Treat as plain text file; try to parse time from filename
            # Example filename: 2021-03-03_15-30_news.txt
            m = re.search(r"(\d{4}-\d{2}-\d{2}[ T_]\d{2}[:\-]?\d{2}(?::?\d{2})?)", p.name)
            ti = pd.to_datetime(m.group(1)) if m else pd.NaT
            try:
                txt = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                txt = p.read_text(errors="ignore")
            rows.append({"__time": ti, "text": txt, "__src": str(p)})

    if not rows:
        return pd.DataFrame(columns=["text"]).assign(__time=pd.NaT, __src="")
    df = pd.DataFrame(rows)
    # Ensure columns
    if "__time" not in df.columns:
        df["__time"] = pd.NaT
    if "text" not in df.columns:
        df["text"] = ""
    return df


# ------------------------- Windowing & alignment ------------------------- #

def build_windows(ts_df: pd.DataFrame,
                  freq: str,
                  L: int,
                  H: int,
                  stride: int,
                  zscore: bool) -> List[Tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """
    Resample to fixed freq, interpolate, and emit sliding windows.
    Returns a list of tuples:
        (win_start, win_end, ts(L,d), target(H,d), index_range_for_window)
    """
    # Regularize index
    ts_df = ts_df.sort_index()
    # Use a full range from min..max with given freq
    idx = pd.date_range(ts_df.index.min(), ts_df.index.max(), freq=freq)
    ts_df = ts_df.reindex(idx)
    ts_df = ts_df.interpolate(method="time").ffill().bfill()

    vals = ts_df.values  # (T, d)
    T, d = vals.shape

    out = []
    i = 0
    while i + L + H <= T:
        x = vals[i:i+L]
        y = vals[i+L:i+L+H]
        if zscore:
            # z-score per feature, using window stats
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mean) / std
            y = (y - mean) / std
        start = ts_df.index[i]
        end = ts_df.index[i+L-1]
        out.append((start, end, x, y, ts_df.index[i:i+L+H]))
        i += stride
    return out


def choose_docs_for_window(docs_df: pd.DataFrame,
                           start: pd.Timestamp,
                           end: pd.Timestamp,
                           lead: pd.Timedelta,
                           lag: pd.Timedelta,
                           max_docs: int,
                           keyword_set: Optional[set] = None) -> Tuple[List[str], List[int]]:
    """
    Pick docs in time range [start - lead, end + lag].
    If many, keep up to max_docs by sorting:
      1) keyword hits (if provided)
      2) closest by time to window end (descending proximity)
    Return (docs_text_list, pos_idx) where pos_idx are *indices within the returned doc list*.

    Positive indices:
      - We pick up to ceil(max_docs/6) recent docs as positive (proximity),
        but prioritizing those with keyword hits if keyword_set is provided.
    """
    # Filter by time if available. NaT -> keep but rank lower.
    lo = start - lead
    hi = end + lag
    df = docs_df.copy()
    has_time = df["__time"].notna().values
    in_range = (df["__time"] >= lo) & (df["__time"] <= hi)
    cand = df[in_range | ~has_time].copy()

    if cand.empty:
        return [], []

    # Rank by keyword hits then proximity to end
    def kw_score(txt: str) -> int:
        if not keyword_set:
            return 0
        # simple token contains; you can expand to regex/lemmatization
        return sum(1 for w in keyword_set if w in txt.lower())

    cand["__kw"] = cand["text"].astype(str).str.lower().map(kw_score)

    def prox_score(t: Any) -> float:
        if pd.isna(t):
            return -np.inf
        # the *higher* the score, the *closer* to end
        return -abs((pd.Timestamp(t) - end).total_seconds())

    cand["__prox"] = cand["__time"].map(prox_score)

    # Sort by keyword desc, proximity desc, fallback stable index
    cand = cand.sort_values(by=["__kw", "__prox"], ascending=[False, False], kind="stable")
    cand = cand.head(max_docs)

    docs_list = cand["text"].astype(str).tolist()

    # Positives: take top m by kw/prox among the *most recent quarter* of window
    m = max(1, int(np.ceil(max_docs / 6)))
    pos_candidates = cand.head(m).index.tolist()
    # Map to indices within docs_list
    pos_idx = []
    for idx in pos_candidates:
        # position in cand.head(max_docs)
        pos = (cand.index == idx).argmax()
        pos_idx.append(int(pos))

    # Dedup & sort
    pos_idx = sorted(set(pos_idx))
    return docs_list, pos_idx


# ------------------------- Conversion pipeline ------------------------- #

def collect_paths(glob_expr: Optional[str]) -> List[Path]:
    if not glob_expr:
        return []
    return [Path(p) for p in sorted(Path().glob(glob_expr))]


def convert_split(ts_paths: List[Path],
                  docs_paths: List[Path],
                  args: argparse.Namespace,
                  out_path: Path) -> int:
    """
    Convert one split into JSONL. Returns number of windows written.
    """
    # Preload all docs once (it can be a lot of files; you can shard if needed)
    docs_df = read_docs_any(docs_paths, time_col=args.doc_time_col, text_col=args.doc_text_col)

    keyword_set = None
    if args.keyword_file:
        try:
            kws = Path(args.keyword_file).read_text(encoding="utf-8").splitlines()
            keyword_set = {k.strip().lower() for k in kws if k.strip()}
        except Exception:
            keyword_set = None

    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for ts_file in ts_paths:
            try:
                ts_df = read_timeseries_any(
                    ts_file,
                    time_col=args.ts_time_col,
                    value_cols=[c.strip() for c in args.ts_value_cols.split(",") if c.strip()],
                    is_jsonl=(ts_file.suffix.lower() == ".jsonl"),
                    jsonl_value_field=args.ts_jsonl_value_field,
                )
            except Exception as e:
                print(f"[warn] Skipping {ts_file}: {e}")
                continue

            windows = build_windows(
                ts_df=ts_df,
                freq=args.freq,
                L=args.L,
                H=args.H,
                stride=args.stride,
                zscore=args.zscore,
            )

            for (start, end, x, y, _) in windows:
                docs_list, pos_idx = choose_docs_for_window(
                    docs_df=docs_df,
                    start=start, end=end,
                    lead=parse_timedelta(args.doc_lead),
                    lag=parse_timedelta(args.doc_lag),
                    max_docs=args.max_docs,
                    keyword_set=keyword_set,
                )
                if not docs_list:
                    # still write an example with empty docs (training can handle it)
                    pass

                row = {
                    "ts": x.tolist(),            # shape (L, d)
                    "docs": docs_list,           # list[str]
                    "target": y.tolist(),        # shape (H, d)
                    "pos_idx": pos_idx,          # list[int], subset of range(len(docs_list))
                }
                out_f.write(json.dumps(row) + "\n")
                total += 1

    return total


# ------------------------- CLI ------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_in", type=str, default=None, help="Root containing raw Time-MMD-like data")
    ap.add_argument("--root_out", type=str, default="data/time_mmd_proc", help="Where JSONLs will be written")
    ap.add_argument("--synthesize", action="store_true", help="Create synthetic JSONLs instead of converting real data")

    # Globs for splits (time-series files). If unspecified, we'll try to use root_in/ts/{split}/*.*
    ap.add_argument("--split_train_glob", type=str, default=None)
    ap.add_argument("--split_val_glob", type=str, default=None)
    ap.add_argument("--split_test_glob", type=str, default=None)

    # Docs glob
    ap.add_argument("--docs_glob", type=str, default=None, help="Glob for documents (jsonl/csv/txt). Can be a big **/* glob.")

    # Time-series parsing
    ap.add_argument("--ts_time_col", type=str, default="time")
    ap.add_argument("--ts_value_cols", type=str, default="value", help="Comma-separated numeric columns OR leave blank if using --ts_jsonl_value_field")
    ap.add_argument("--ts_jsonl_value_field", type=str, default=None, help="If JSONL contains a list-like field for values, name it here")

    # Docs parsing
    ap.add_argument("--doc_time_col", type=str, default="time", help="Column name for document timestamp")
    ap.add_argument("--doc_text_col", type=str, default="text", help="Column name for document text")
    ap.add_argument("--keyword_file", type=str, default=None, help="Optional file with one keyword per line to prioritize positives")

    # Windowing / resampling
    ap.add_argument("--freq", type=str, default="D", help="Resample frequency (e.g., D,H,15min)")
    ap.add_argument("--L", type=int, default=96, help="Lookback length")
    ap.add_argument("--H", type=int, default=8, help="Forecast horizon")
    ap.add_argument("--stride", type=int, default=8, help="Sliding window stride")

    # Doc selection window (relative to [window_start, window_end])
    ap.add_argument("--max_docs", type=int, default=24)
    ap.add_argument("--doc_lead", type=str, default="3D", help="How far back before start to include docs (e.g., 3D, 12H)")
    ap.add_argument("--doc_lag", type=str, default="1D", help="How far after end to include docs")

    # Normalization
    ap.add_argument("--zscore", action="store_true", help="Apply z-score normalization per window")

    args = ap.parse_args()

    out_root = Path(args.root_out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.synthesize:
        train = synthesize(n=256, L=args.L, H=args.H)
        val = synthesize(n=64, L=args.L, H=args.H)
        test = synthesize(n=64, L=args.L, H=args.H)
        write_jsonl(train, out_root / "train.jsonl")
        write_jsonl(val, out_root / "val.jsonl")
        write_jsonl(test, out_root / "test.jsonl")
        print(f"[ok] Synthesized Time-MMD-like dataset in {out_root.resolve()}")
        return

    if not args.root_in and not (args.split_train_glob or args.split_val_glob or args.split_test_glob):
        raise SystemExit("Provide --root_in or explicit --split_*_glob globs (or use --synthesize).")

    # Default ts globs if not provided
    def _default_glob(split: str) -> str:
        if not args.root_in:
            return None
        return str(Path(args.root_in) / "ts" / split / "*")

    g_train = args.split_train_glob or _default_glob("train")
    g_val   = args.split_val_glob   or _default_glob("val")
    g_test  = args.split_test_glob  or _default_glob("test")

    ts_train = collect_paths(g_train) if g_train else []
    ts_val   = collect_paths(g_val) if g_val else []
    ts_test  = collect_paths(g_test) if g_test else []

    if args.docs_glob:
        docs = collect_paths(args.docs_glob)
    else:
        # default: try root_in/docs/**/*.*
        if not args.root_in:
            raise SystemExit("No --docs_glob and no --root_in to infer docs; can’t proceed.")
        docs = collect_paths(str(Path(args.root_in) / "docs" / "**" / "*"))

    print(f"[info] TS files: train={len(ts_train)} val={len(ts_val)} test={len(ts_test)} | docs={len(docs)}")

    # Convert each split
    total_train = convert_split(ts_train, docs, args, out_root / "train.jsonl")
    total_val   = convert_split(ts_val,   docs, args, out_root / "val.jsonl")
    total_test  = convert_split(ts_test,  docs, args, out_root / "test.jsonl")

    print(f"[ok] Wrote: train={total_train} windows, val={total_val}, test={total_test} → {out_root.resolve()}")


if __name__ == "__main__":
    main()
