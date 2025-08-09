#!/usr/bin/env python3
"""
Train CT-GSSN on synthetic/MIMIC-III/METR-LA with improved training utilities.

What’s new vs the basic script:
- Mixed precision (AMP) with GradScaler
- Gradient accumulation
- Learning-rate schedulers (cosine or step)
- Early stopping on validation MSE
- Checkpointing: best / last / periodic
- Resume (model, optimizer, scaler, scheduler, global step/epoch)
- Evaluate every K steps (more frequent feedback)
- Optional torch.compile acceleration

Config additions (under train:):
  amp: true
  grad_accum_steps: 1
  scheduler:
    name: "cosine"        # ["none", "cosine", "step"]
    t_max: 50             # cosine param
    eta_min: 1.0e-6
    step_size: 10         # stepLR param
    gamma: 0.5
  eval_every_steps: 200
  early_stop_patience: 10
  save_dir: "runs/ct_gssn"
  ckpt_every: 1000
  resume_from: null
  compile: false

Examples:
---------
python scripts/train_ct_gssn.py --config configs/ct_gssn/mimic_base.yaml
python scripts/train_ct_gssn.py --config configs/ct_gssn/metr_base.yaml

Resuming:
---------
python scripts/train_ct_gssn.py --config ...  (if train.resume_from is set in YAML, it will resume)

Notes:
------
- This script subclasses CTGSSNTrainer to avoid changing your original trainer file.
- Validation metric = MSE (lower is better).
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

from ct_gssn.models.ct_gssn import CTGSSN, CTGSSNConfig
from ct_gssn.trainers.trainer import CTGSSNTrainer, TrainConfig
from ct_gssn.data.irregular_dataset import SyntheticIMTSDataset, collate_imts
from ct_gssn.data.mimic3 import MIMIC3Dataset, collate_mimic3
from ct_gssn.data.metr_la import METRLADataset, collate_metrla
from ct_gssn.utils.helpers import seed_all
from ct_gssn.utils.metrics import mse_metric


# ------------------------- Config & Data Loading ------------------------- #

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # simple inherit
    if "inherit" in cfg:
        base = cfg["inherit"]
        with open(base, "r") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "inherit"})
        cfg = base_cfg
    return cfg


def make_dataloaders(cfg: Dict[str, Any]):
    data = cfg["data"]
    name = data["name"].lower()
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]

    if name in ("synthetic_imts", "synthetic"):
        ds = SyntheticIMTSDataset(
            num_samples=data.get("num_samples", 256),
            N=data.get("N", 12),
            L=data.get("L", 48),
            input_dim=data.get("input_dim", 4),
            out_dim=data.get("out_dim", 1),
            obs_prob=data.get("obs_prob", 0.5),
            seed=data.get("seed", 42),
        )
        dl_tr = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_imts)
        return dl_tr, None

    elif name == "mimic3":
        train = MIMIC3Dataset(root=data["root"], split=data.get("split", "train"),
                              adjacency=data.get("adjacency", "knn"), knn_k=data.get("knn_k", 8))
        val = MIMIC3Dataset(root=data["root"], split=data.get("val_split", "val"),
                            adjacency=data.get("adjacency", "knn"), knn_k=data.get("knn_k", 8))
        dl_tr = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_mimic3)
        dl_va = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_mimic3)
        return dl_tr, dl_va

    elif name in ("metr_la", "metr-la"):
        train = METRLADataset(root=data["root"], split=data.get("split", "train"),
                              use_dynamic_adj=data.get("use_dynamic_adj", False))
        val = METRLADataset(root=data["root"], split=data.get("val_split", "val"),
                            use_dynamic_adj=data.get("use_dynamic_adj", False))
        dl_tr = DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_metrla)
        dl_va = DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_metrla)
        return dl_tr, dl_va

    else:
        raise ValueError(f"Unknown dataset {name}")


# ------------------------- Enhanced Trainer Wrapper ------------------------- #

class EnhancedCTGSSNTrainer(CTGSSNTrainer):
    """
    Extension over the base CTGSSNTrainer adding:
      - AMP autocast + GradScaler
      - Gradient accumulation
      - LR scheduler (cosine/step)
      - Early stopping & checkpointing
      - Eval every K steps
      - Optional torch.compile acceleration
      - Resume support
    """

    def __init__(self, model_cfg: CTGSSNConfig, train_cfg: TrainConfig, extras: Dict[str, Any]):
        super().__init__(model_cfg, train_cfg)

        # ---- Extras & defaults ----
        self.amp = bool(extras.get("amp", torch.cuda.is_available()))
        self.grad_accum_steps = int(extras.get("grad_accum_steps", 1))
        self.eval_every_steps = int(extras.get("eval_every_steps", 0))  # 0 => per-epoch only
        self.early_stop_patience = int(extras.get("early_stop_patience", 0))
        self.save_dir = Path(extras.get("save_dir", self.cfg.log_dir))
        self.ckpt_every = int(extras.get("ckpt_every", 0))
        self.compile_flag = bool(extras.get("compile", False))
        self.resume_from = extras.get("resume_from", None)
        self.best_val = float("inf")
        self.no_improve = 0
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # ---- Optional: torch.compile for speed (PyTorch 2+) ----
        if self.compile_flag and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                print("[compile] Enabled torch.compile for model.")
            except Exception as e:
                print(f"[compile] Skipping torch.compile: {e}")

        # ---- LR scheduler ----
        sch_cfg = extras.get("scheduler", {"name": "none"})
        self.scheduler = self._build_scheduler(sch_cfg)

        # ---- Checkpoint dir ----
        (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        # ---- Resume if requested ----
        if self.resume_from:
            self._try_resume(self.resume_from)

    def _build_scheduler(self, sch_cfg: Dict[str, Any]):
        name = (sch_cfg.get("name") or "none").lower()
        if name == "none":
            return None
        elif name == "cosine":
            t_max = int(sch_cfg.get("t_max", self.cfg.max_epochs))
            eta_min = float(sch_cfg.get("eta_min", 1e-6))
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=t_max, eta_min=eta_min)
        elif name == "step":
            step_size = int(sch_cfg.get("step_size", 10))
            gamma = float(sch_cfg.get("gamma", 0.5))
            return torch.optim.lr_scheduler.StepLR(self.opt, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    def _save_ckpt(self, tag: str):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler else None),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
        }
        path = self.save_dir / "checkpoints" / f"{tag}.pt"
        torch.save(ckpt, path)
        print(f"[ckpt] Saved: {path}")

    def _try_resume(self, path: str):
        p = Path(path)
        if p.is_dir():
            # try common names
            cand = [p / "last.pt", p / "best.pt"]
            p = next((c for c in cand if c.exists()), None)
            if p is None:
                print(f"[resume] No ckpt found in {path}")
                return
        if p is None or not p.exists():
            print(f"[resume] Path not found: {path}")
            return
        ckpt = torch.load(p, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt and ckpt["opt"] is not None:
            self.opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        print(f"[resume] Resumed from {p} at epoch={self.current_epoch}, step={self.global_step}, best_val={self.best_val:.6f}")

    def _step_update(self, loss: torch.Tensor):
        """Gradient accumulation + AMP-aware optimizer step."""
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if (self.global_step + 1) % self.grad_accum_steps == 0:
            if self.amp:
                self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.amp:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def _validate_once(self, loader: DataLoader) -> float:
        """Return validation MSE."""
        if loader is None:
            return float("nan")
        self.model.eval()
        mses = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                deltas = batch["deltas"].to(self.device)
                adj = batch["adj"].to(self.device)

                y_pred, _ = self.model(x, mask, deltas, adj, return_aux=False)
                mses.append(mse_metric(y_pred, y, mask))
        val_mse = sum(mses) / max(1, len(mses))
        return float(val_mse)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Overrides base trainer loop to add:
        - AMP / accumulation / schedulers
        - eval every k steps
        - early stopping + checkpointing
        """
        from ct_gssn.utils.helpers import save_config_snapshot

        save_config_snapshot(str(self.save_dir), {"model": self.model.cfg.__dict__, "train": self.cfg.__dict__})
        self.current_epoch = getattr(self, "current_epoch", 0)

        for epoch in range(self.current_epoch, self.cfg.max_epochs):
            self.current_epoch = epoch
            self.model.train()
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)
                deltas = batch["deltas"].to(self.device)
                adj = batch["adj"].to(self.device)

                # Forward (with AMP)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    target_mask = self._mask_targets(y, mask, self.cfg.mask_ratio)
                    y_pred, aux = self.model(x, mask, deltas, adj, return_aux=True)

                    # Original losses from base trainer
                    lp = self._loss_predict(y_pred, y, target_mask)
                    lg = self._loss_graph(aux, adj)
                    ls = self._loss_stability(aux)

                    loss = lp + self.cfg.lambda_graph * lg + self.cfg.lambda_stable * ls

                # Backward / step (accumulation-aware)
                self._step_update(loss)

                # Logging each 10 steps (same as base)
                if self.global_step % 10 == 0:
                    mse = mse_metric(y_pred, y, mask)
                    self.tb.add_scalar("train/loss_total", float(loss.item()), self.global_step)
                    self.tb.add_scalar("train/loss_predict", float(lp.item()), self.global_step)
                    self.tb.add_scalar("train/loss_graph", float(lg.item()), self.global_step)
                    self.tb.add_scalar("train/loss_stable", float(ls.item()), self.global_step)
                    self.tb.add_scalar("train/mse", float(mse), self.global_step)
                    if self.wandb:
                        self.wandb.log({
                            "loss_total": float(loss.item()),
                            "loss_predict": float(lp.item()),
                            "loss_graph": float(lg.item()),
                            "loss_stable": float(ls.item()),
                            "mse": float(mse),
                            "lr": self.opt.param_groups[0]["lr"],
                        }, step=self.global_step)

                # Periodic eval during epoch
                if self.eval_every_steps and self.global_step % self.eval_every_steps == 0 and val_loader is not None:
                    val_mse = self._validate_once(val_loader)
                    self.tb.add_scalar("val/mse_intermediate", val_mse, self.global_step)
                    if self.wandb:
                        self.wandb.log({"val_mse_intermediate": val_mse}, step=self.global_step)
                    self._maybe_update_best(val_mse)

                # Periodic checkpoint
                if self.ckpt_every and self.global_step % self.ckpt_every == 0:
                    self._save_ckpt("last")

                self.global_step += 1

            # End-of-epoch validation
            if val_loader is not None:
                val_mse = self._validate_once(val_loader)
                self.tb.add_scalar("val/mse", val_mse, self.global_step)
                if self.wandb:
                    self.wandb.log({"val_mse": val_mse, "epoch": epoch}, step=self.global_step)
                self._maybe_update_best(val_mse)

            # Step the scheduler (epoch-based schedulers)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            # Always save "last" at end of epoch
            self._save_ckpt("last")

            # Early stopping
            if self.early_stop_patience > 0 and self.no_improve >= self.early_stop_patience:
                print(f"[early-stop] No improvement for {self.no_improve} validations — stopping.")
                break

    # ---- Helper loss wrappers (reuse base implementation pieces) ----

    def _loss_predict(self, y_pred: torch.Tensor, y_true: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        from ct_gssn.models.ct_gssn import masked_prediction_loss
        return masked_prediction_loss(y_pred, y_true, target_mask)

    def _loss_graph(self, aux: Dict[str, torch.Tensor], adj: torch.Tensor) -> torch.Tensor:
        # Link prediction from c (use last step c from last layer as proxy)
        c_last = aux["c"][-1, :, -1]  # (B,N,C)
        edges = self._sample_edges_for_link_pred(adj)
        logits_pos = self.link_pred(c_last, edges["pos"])
        logits_neg = self.link_pred(c_last, edges["neg"])
        labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=1)
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def _loss_stability(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        from ct_gssn.models.ct_gssn import lyapunov_stability_loss
        return lyapunov_stability_loss(aux["A"])

    def _maybe_update_best(self, val_mse: float):
        improved = val_mse < self.best_val
        if improved:
            self.best_val = val_mse
            self.no_improve = 0
            self._save_ckpt("best")
            print(f"[val] New best MSE={val_mse:.6f}")
        else:
            self.no_improve += 1


# ------------------------------- CLI -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Improved CT-GSSN Trainer")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["train"].get("seed", 1337))

    model_cfg = CTGSSNConfig(**cfg["model"])
    train_cfg = TrainConfig(**cfg["train"])

    # Extras recognized by EnhancedCTGSSNTrainer
    extras = {
        "amp": cfg["train"].get("amp", torch.cuda.is_available()),
        "grad_accum_steps": cfg["train"].get("grad_accum_steps", 1),
        "scheduler": cfg["train"].get("scheduler", {"name": "none"}),
        "eval_every_steps": cfg["train"].get("eval_every_steps", 0),
        "early_stop_patience": cfg["train"].get("early_stop_patience", 0),
        "save_dir": cfg["train"].get("log_dir", "runs/ct_gssn"),
        "ckpt_every": cfg["train"].get("ckpt_every", 0),
        "resume_from": cfg["train"].get("resume_from", None),
        "compile": cfg["train"].get("compile", False),
    }

    # Dataloaders
    tr_loader, va_loader = make_dataloaders(cfg)

    # Trainer
    trainer = EnhancedCTGSSNTrainer(model_cfg, train_cfg, extras)
    trainer.train(tr_loader, va_loader)


if __name__ == "__main__":
    main()
