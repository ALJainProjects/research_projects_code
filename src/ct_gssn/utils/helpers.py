# src/ct_gssn/utils/helpers.py
"""
Helper utilities: seeding, TB/W&B init, config snapshot, lightweight logging,
checkpoint helpers, and misc QoL functions.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------
# Seeding & determinism
# ---------------------------------------------------------------------
def seed_all(seed: int = 1337, deterministic: bool = False, cuda_deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch. Optionally enable deterministic behaviors.

    Args:
        seed: base seed for RNGs
        deterministic: set PyTorch's deterministic algorithms (may be slower)
        cuda_deterministic: additionally set CUBLAS/CUDA deterministic flags
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # TF32 can slightly change numerics; keep it on unless you need strict reproducibility
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cuda_deterministic:
        # These envs help make CUDA GEMM more deterministic; set before CUDA context creation if possible
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # or ":4096:8"
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


# ---------------------------------------------------------------------
# Run directories & logging
# ---------------------------------------------------------------------
def make_run_dir(root: str | Path, name: str = "run") -> Path:
    """
    Create a unique run directory like: <root>/<name>-YYYYmmdd-HHMMSS

    Returns:
        Path to the created directory.
    """
    root = Path(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = root / f"{name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_tb_writer(log_dir: str | Path, subdir_if_exists: bool = True) -> SummaryWriter:
    """
    Create a SummaryWriter. If the directory already exists and subdir_if_exists=True,
    attach a timestamped subfolder to avoid clobbering previous runs.

    Example:
        runs/ct_gssn/run-20250101-120000
    """
    log_dir = Path(log_dir)
    if log_dir.exists() and any(log_dir.iterdir()) and subdir_if_exists:
        log_dir = make_run_dir(log_dir, name="run")
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


# ---------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------
def maybe_init_wandb(
    use: bool,
    project: str,
    config: dict,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,  # "online", "offline", or None
    resume: Optional[str] = None,  # "allow"|"must"|None
    dir_: Optional[str | Path] = None,
):
    """
    Initialize Weights & Biases if `use` is True. Returns the imported module or None.

    Args:
        use: whether to initialize W&B
        project: W&B project name
        config: dict to log as config
        run_name: optional user-friendly run name
        mode: "online" | "offline" | None (default uses env or online)
        resume: "allow" | "must" | None
        dir_: directory for W&B files
    """
    if not use:
        return None
    try:
        import wandb
        kwargs = {
            "project": project,
            "config": config,
        }
        if run_name:
            kwargs["name"] = run_name
        if mode:
            kwargs["mode"] = mode
        if resume:
            kwargs["resume"] = resume
        if dir_:
            kwargs["dir"] = str(dir_)
        wandb.init(**kwargs)
        # Return module, not run handle, so caller can still use wandb.Plotly etc.
        return wandb
    except Exception as e:
        print(f"[W&B] failed to init: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------
# Config snapshotting
# ---------------------------------------------------------------------
def save_config_snapshot(log_dir: str | Path, cfg: dict, name_no_ext: str = "config_snapshot") -> Path:
    """
    Save configuration to both JSON and YAML (if pyyaml installed).

    Layout:
        <log_dir>/
          config_snapshot.json
          config_snapshot.yml   (best-effort)
          git_meta.json         (if available)

    Returns:
        Path to the JSON file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    json_path = log_dir / f"{name_no_ext}.json"
    json_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    try:
        import yaml  # optional
        yaml_path = log_dir / f"{name_no_ext}.yml"
        yaml.safe_dump(cfg, yaml_path.open("w", encoding="utf-8"), sort_keys=False)
    except Exception:
        pass

    # Also write git metadata if available
    try:
        meta = get_git_metadata()
        (log_dir / "git_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass

    return json_path


def get_git_metadata() -> Dict[str, Any]:
    """
    Collect minimal git metadata for reproducibility.
    """
    import subprocess

    def _run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()

    meta = {}
    try:
        meta["commit"] = _run(["git", "rev-parse", "HEAD"])
        meta["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        meta["is_dirty"] = bool(_run(["git", "status", "--porcelain"]))
        meta["remote"] = _run(["git", "remote", "get-url", "origin"])
    except Exception:
        # running outside a git repo is fine
        pass
    return meta


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a minimal-but-complete checkpoint.
    """
    payload: Dict[str, Any] = {"model": model.state_dict()}
    if optim is not None:
        payload["opt"] = optim.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load checkpoint. Returns the checkpoint dict for any custom fields.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optim is not None and "opt" in ckpt:
        optim.load_state_dict(ckpt["opt"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ---------------------------------------------------------------------
# Misc QoL
# ---------------------------------------------------------------------
def get_device(prefer: str = "cuda") -> torch.device:
    """Pick an available torch.device with a simple preference order."""
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters (optionally only trainable)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_bytes(n: int) -> str:
    """Pretty-print bytes (e.g., GPU memory)."""
    step = 1024.0
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    f = float(n)
    while f >= step and i < len(units) - 1:
        f /= step
        i += 1
    return f"{f:.2f} {units[i]}"


class AverageMeter:
    """Tracks a running average; simple and thread-safe enough for our loops."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.n = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, v: float, k: int = 1):
        self.val = float(v)
        self.sum += float(v) * k
        self.n += k
        self.avg = self.sum / max(1, self.n)


class timeit:
    """Context manager for measuring elapsed time."""

    def __init__(self, msg: str = ""):
        self.msg = msg
        self.t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.t0
        if self.msg:
            print(f"{self.msg}: {self.elapsed:.3f}s")


def safe_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, convert to float32 NumPy."""
    return x.detach().cpu().float().numpy()
