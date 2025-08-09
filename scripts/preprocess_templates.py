# scripts/preprocess_templates.py
#!/usr/bin/env python3
"""
Preprocessing templates generator + one-command dataset prep hub.

What this script does:
  1) Writes reference preprocessing templates for:
       - MIMIC-III  -> CT-GSSN format   (data/mimic3_proc/*.pt)
       - METR-LA    -> CT-GSSN format   (data/metr-la_proc/*.pt)
       - Time-MMD   -> CEX-TSLM format  (data/time_mmd_proc/*.jsonl)
  2) (Optional) Runs the provided reference converters in scripts/:
       - scripts/convert_mimic3.py
       - scripts/convert_metr_la.py
       - scripts/convert_time_mmd.py
     Falls back to *synthetic* outputs when no raw paths are provided.
  3) (Optional) Emits training config snippets into `configs/`
     so you can train with one command per dataset.

Examples:
  # Just generate template folders and docs
  python scripts/preprocess_templates.py

  # Templates for a single dataset
  python scripts/preprocess_templates.py --dataset metr_la

  # Run preprocessing right now (synthetic fallback if no raw paths)
  python scripts/preprocess_templates.py --dataset time_mmd --run

  # Generate configs that hook into train scripts
  python scripts/preprocess_templates.py --emit-configs

  # One-shot: make METR-LA, write configs, and you’re ready to train
  python scripts/preprocess_templates.py --dataset metr_la --run --emit-configs
  python scripts/train_ct_gssn.py --config configs/ct_gssn_metrla.yml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "preprocess_templates"
CONFIGS_DIR = ROOT / "configs"
SCRIPTS_DIR = ROOT / "scripts"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


# ----------------------------
# Template writers (README + schema + snippet)
# ----------------------------

def write_mimic3_template() -> None:
    base = OUT_DIR / "mimic3"
    _write(
        base / "README.md",
        dedent(
            """
            # MIMIC-III Preprocessing Template

            Convert MIMIC-III raw tables into an irregular multivariate time series format
            for CT-GSSN. Variables become graph nodes; each patient becomes one sample.

            ## Expected Output
            `data/mimic3_proc/` with:
              - `train.pt`, `val.pt`, `test.pt`
                Each is a torch-saved list[dict]. Dict keys:
                  x:      tensor float32 [N, L, P]  (observations/impulses)
                  y:      tensor float32 [N, L, Q]  (targets; zeros if N/A)
                  mask:   tensor float32 [N, L]     (1 if observed at t_k else 0)
                  deltas: tensor float32 [L]        (Δt between grid steps)
                  adj:    tensor float32 [N, N] OR [L, N, N] (static or dynamic adjacency)
                  meta:   dict with identifiers / variable names / units

            See `scripts/convert_mimic3.py` for a reference converter. You must supply
            your own ETL of MIMIC-III—this repo does not ship raw MIMIC data.
            """
        ),
    )
    _write(
        base / "schema.json",
        json.dumps(
            {
                "x": "tensor float32 [N, L, P]",
                "y": "tensor float32 [N, L, Q]",
                "mask": "tensor float32 [N, L]",
                "deltas": "tensor float32 [L]",
                "adj": "tensor float32 [N, N] or [L, N, N]",
                "meta": {"patient_id": "str", "variables": "list[str]", "units": "list[str]"},
            },
            indent=2,
        ),
    )
    _write(
        base / "snippet_convert.py",
        dedent(
            """
            import numpy as np
            import torch
            from pathlib import Path
            from sklearn.neighbors import NearestNeighbors

            def build_knn_adjacency(node_feats: np.ndarray, k: int = 8) -> np.ndarray:
                nbrs = NearestNeighbors(n_neighbors=min(k+1, len(node_feats))).fit(node_feats)
                graph = np.zeros((len(node_feats), len(node_feats)), dtype=np.float32)
                for i in range(len(node_feats)):
                    _, idxs = nbrs.kneighbors(node_feats[i:i+1])
                    for j in idxs[0]:
                        if j != i:
                            graph[i, j] = 1.0
                graph = ((graph + graph.T) > 0).astype(np.float32)
                deg = np.sum(graph, axis=1, keepdims=True).clip(min=1.0)
                graph = graph / deg
                return graph

            def example_patient(N=12, L=48, P=4, Q=1):
                x = np.random.randn(N, L, P).astype(np.float32)
                y = np.random.randn(N, L, Q).astype(np.float32)
                mask = (np.random.rand(N, L) < 0.6).astype(np.float32)
                deltas = (np.random.rand(L) * 0.6 + 0.1).astype(np.float32)
                adj = build_knn_adjacency(np.random.randn(N, 16).astype(np.float32), k=4)
                return {"x": torch.tensor(x), "y": torch.tensor(y), "mask": torch.tensor(mask),
                        "deltas": torch.tensor(deltas), "adj": torch.tensor(adj),
                        "meta": {"patient_id": "EXAMPLE"}}

            if __name__ == "__main__":
                out = Path("data/mimic3_proc"); out.mkdir(parents=True, exist_ok=True)
                patients = [example_patient() for _ in range(8)]
                torch.save(patients, out / "train.pt")
                torch.save(patients[:2], out / "val.pt")
                torch.save(patients[:2], out / "test.pt")
                print("Wrote synthetic MIMIC-III-like splits to", out)
            """
        ),
    )


def write_metrla_template() -> None:
    base = OUT_DIR / "metr_la"
    _write(
        base / "README.md",
        dedent(
            """
            # METR-LA Preprocessing Template

            Convert METR-LA traffic speeds + adjacency to CT-GSSN tensors.

            ## Output Folder
            `data/metr-la_proc/`:
              - `train.pt`, `val.pt`, `test.pt`  (torch-saved list[dict])
                Each dict:
                  x: (N, L, 1), y: (N, L, 1), mask: (N, L), deltas: (L,), adj: (N, N), meta: {"sensor_ids": list[str]}

            See `scripts/convert_metr_la.py` for a reference converter.
            """
        ),
    )
    _write(
        base / "schema.json",
        json.dumps(
            {
                "x": "tensor float32 [N, L, 1]",
                "y": "tensor float32 [N, L, 1]",
                "mask": "tensor float32 [N, L]",
                "deltas": "tensor float32 [L]",
                "adj": "tensor float32 [N, N] or [L, N, N]",
                "meta": {"sensor_ids": "list[str]"},
            },
            indent=2,
        ),
    )


def write_timemmd_template() -> None:
    base = OUT_DIR / "time_mmd"
    _write(
        base / "README.md",
        dedent(
            """
            # Time-MMD Preprocessing Template (CEX-TSLM)

            Output:
              data/time_mmd_proc/
                train.jsonl
                val.jsonl
                test.jsonl

            JSONL record schema:
              {
                "ts": [[...], ...],       # L x d_x
                "docs": ["text", ...],    # list of strings
                "target": [[...], ...],   # H x d_x
                "pos_idx": [i0, i1, ...]  # optional, causal-positive doc indices
              }

            See `scripts/convert_time_mmd.py` for a reference converter.
            """
        ),
    )
    _write(
        base / "schema.json",
        json.dumps(
            {
                "ts": "list[list[float]] (L x d_x)",
                "docs": "list[str]",
                "target": "list[list[float]] (H x d_x)",
                "pos_idx": "optional list[int] length ~ L/patch",
            },
            indent=2,
        ),
    )


# ----------------------------
# Config snippet writers
# ----------------------------

def write_config_snippets() -> None:
    """
    Emit ready-to-train configs that inherit your CT-GSSN/CEX-TSLM base.
    If your base filenames differ, tweak the 'inherit' paths below or
    just edit the generated files in configs/.
    """
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # CT-GSSN on METR-LA
    _write(
        CONFIGS_DIR / "ct_gssn_metrla.yml",
        dedent(
            """
            inherit: configs/ct_gssn_base.yml
            data:
              name: metr_la
              root: data/metr-la_proc
              split: train
              val_split: val
              use_dynamic_adj: false
            model:
              input_dim: 1
              out_dim: 1
            train:
              max_epochs: 30
              batch_size: 8
            """
        ),
    )

    # CT-GSSN on MIMIC-III
    _write(
        CONFIGS_DIR / "ct_gssn_mimic3.yml",
        dedent(
            """
            inherit: configs/ct_gssn_base.yml
            data:
              name: mimic3
              root: data/mimic3_proc
              split: train
              val_split: val
              adjacency: knn
              knn_k: 8
            model:
              input_dim: 4
              out_dim: 1
            train:
              max_epochs: 30
              batch_size: 8
            """
        ),
    )

    # CT-GSSN on synthetic (great for a smoke test)
    _write(
        CONFIGS_DIR / "ct_gssn_synth.yml",
        dedent(
            """
            inherit: configs/ct_gssn_base.yml
            data:
              name: synthetic_imts
              num_samples: 256
              N: 12
              L: 48
              input_dim: 4
              out_dim: 1
              obs_prob: 0.5
              seed: 42
            model:
              input_dim: 4
              out_dim: 1
            train:
              max_epochs: 10
              batch_size: 16
            """
        ),
    )

    # CEX-TSLM on Time-MMD
    _write(
        CONFIGS_DIR / "cex_tslm_time_mmd.yml",
        dedent(
            """
            # Inference/training config for CEX-TSLM
            data:
              name: time_mmd
              root: data/time_mmd_proc
              split: train
              val_split: val
              d_x: 4
              horizon: 8
              max_docs: 64
              max_tok: 64
            model:
              ts_in_dim: 4
              d_model: 256
              ts_patch_len: 4
              forecast_h: 8
            train:
              batch_size: 8
              num_workers: 2
              lr: 1.0e-4
              max_epochs: 10
              device: "cuda"
            """
        ),
    )


# ----------------------------
# Optional: run the converters now
# ----------------------------

def run_mimic3(out_root: Path) -> None:
    """
    Calls scripts/convert_mimic3.py to create data/mimic3_proc/*.pt
    The reference converter already synthesizes example patients if no raw ETL is wired.
    """
    print(">> Running MIMIC-III converter...")
    cmd = [sys.executable, str(SCRIPTS_DIR / "convert_mimic3.py"), "--out_root", str(out_root)]
    subprocess.run(cmd, check=True)
    print("   ✓ MIMIC-III processed to", out_root)


def run_metr_la(out_root: Path, speed_npz: str | None, adj_npy: str | None) -> None:
    """
    Calls scripts/convert_metr_la.py; if no paths are given, it synthesizes.
    """
    print(">> Running METR-LA converter...")
    cmd = [sys.executable, str(SCRIPTS_DIR / "convert_metr_la.py"), "--out_root", str(out_root)]
    if speed_npz:
        cmd += ["--speed_npz", speed_npz]
    if adj_npy:
        cmd += ["--adj_npy", adj_npy]
    subprocess.run(cmd, check=True)
    print("   ✓ METR-LA processed to", out_root)


def run_time_mmd(out_root: Path, synthesize: bool, tmmd_root: str | None) -> None:
    """
    Calls scripts/convert_time_mmd.py; if --synthesize, emits a tiny fake dataset.
    """
    print(">> Running Time-MMD converter...")
    cmd = [sys.executable, str(SCRIPTS_DIR / "convert_time_mmd.py"), "--root", str(out_root)]
    if synthesize:
        cmd += ["--synthesize"]
    elif tmmd_root:
        # If your convert_time_mmd.py supports raw root args, you can add them here.
        # (The provided reference script expects --root only and user edits real parsing.)
        pass
    subprocess.run(cmd, check=True)
    print("   ✓ Time-MMD processed to", out_root)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mimic3", "metr_la", "time_mmd", "all"], default="all",
                        help="Which dataset templates (and optional conversion) to handle.")
    parser.add_argument("--run", action="store_true",
                        help="Run the reference converter(s) after writing templates.")
    parser.add_argument("--emit-configs", action="store_true",
                        help="Write ready-to-train config snippets into configs/.")
    # METR-LA optional raw paths
    parser.add_argument("--metr-speed-npz", default=None, help="Path to METR-LA speed.npz")
    parser.add_argument("--metr-adj-npy", default=None, help="Path to METR-LA adjacency .npy")
    # Time-MMD convenience
    parser.add_argument("--time-mmd-synthesize", action="store_true", help="Emit synthetic Time-MMD JSONL")
    parser.add_argument("--time-mmd-root", default=None, help="(If you extend the converter) raw root for Time-MMD")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Write templates
    if args.dataset in ("mimic3", "all"):
        write_mimic3_template()
    if args.dataset in ("metr_la", "all"):
        write_metrla_template()
    if args.dataset in ("time_mmd", "all"):
        write_timemmd_template()
    print(f"Templates written to: {OUT_DIR}")

    # 2) Optionally run converters now
    if args.run:
        if args.dataset in ("mimic3", "all"):
            run_mimic3(ROOT / "data" / "mimic3_proc")
        if args.dataset in ("metr_la", "all"):
            run_metr_la(ROOT / "data" / "metr-la_proc", args.metr_speed_npz, args.metr_adj_npy)
        if args.dataset in ("time_mmd", "all"):
            run_time_mmd(ROOT / "data" / "time_mmd_proc", args.time_mmd_synthesize, args.time_mmd_root)

    # 3) Optionally emit training configs
    if args.emit_configs:
        write_config_snippets()
        print(f"Config snippets written to: {CONFIGS_DIR}")

    print("Done.")


if __name__ == "__main__":
    main()
