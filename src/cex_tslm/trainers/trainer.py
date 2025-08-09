"""
Trainer for CEX-TSLM with quality-of-life upgrades.

What you get
------------
- Forecast MSE + optional explanation CE + causal-contrast supervision
- AMP mixed-precision + gradient accumulation
- Cosine LR schedule with warmup
- Optional EMA (exponential moving average) weights for evaluation
- Checkpointing (best + last) and early stopping
- Rich logging to TensorBoard and (optionally) W&B
- Graceful handling of missing 'pos_idx' or 'tgt_tokens' in a batch
- Utilities to inspect attention (entropy) and a tiny qualitative text decode

Typical usage
-------------
from torch.utils.data import DataLoader
from cex_tslm.models.cex_tslm import CEXTSLM, CEXConfig
from cex_tslm.trainers.trainer import CEXTrainer, TrainConfig

model = CEXTSLM(CEXConfig())
trainer = CEXTrainer(model, TrainConfig())
trainer.train(train_loader, val_loader)

Notes
-----
- If you use HF encoders/decoders, collate should supply tokenized tensors.
- 'tgt_tokens' is optional; if present, explanation loss is used with weight alpha_explain.
- 'pos_idx' is optional; if present, causal contrastive is used with weight beta_causal.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from cex_tslm.models.cex_tslm import (
    CEXTSLM, CEXConfig,
    forecast_loss, explanation_loss, causal_contrastive_loss
)
from ct_gssn.utils.helpers import seed_all, get_tb_writer, maybe_init_wandb


# ------------------------------ Configs ------------------------------------ #

@dataclass
class TrainConfig:
    # data
    batch_size: int = 8
    num_workers: int = 2
    # optimization
    lr: float = 2e-4
    weight_decay: float = 1e-5
    max_epochs: int = 10
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    # device/mixed precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    # seeds
    seed: int = 1337
    # losses
    alpha_explain: float = 0.5         # explanation CE weight
    beta_causal: float = 1.0           # causal contrast weight
    tau: float = 0.07                  # InfoNCE temperature
    # scheduler (cosine with warmup)
    warmup_steps: int = 500
    min_lr_scale: float = 0.1          # final LR = lr * min_lr_scale
    # EMA
    ema_decay: float = 0.999
    use_ema: bool = True
    # early stopping
    early_stop_patience: int = 10
    # logging/ckpt
    log_dir: str = "runs/cex_tslm"
    ckpt_dir: str = "checkpoints/cex_tslm"
    save_every: int = 0                # if >0, save 'last' every N steps
    use_wandb: bool = False
    wandb_project: str = "cex_tslm"
    # qualitative decode logging (only if model provides a tokenizer/eos)
    log_text_every: int = 200


# ------------------------------ Helper: EMA -------------------------------- #

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store(self, model: nn.Module):
        self.backup = {k: p.detach().clone() for k, p in model.state_dict().items()}

    def copy_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def restore(self, model: nn.Module):
        if hasattr(self, "backup"):
            model.load_state_dict(self.backup, strict=False)
            del self.backup


# ------------------------------ Trainer ------------------------------------ #

class CEXTrainer:
    """
    Trainer for CEX-TSLM:
      - forecasting MSE
      - explanation LM loss (teacher forcing, if tgt_tokens present)
      - causal contrastive loss supervising cross-attention logits (if pos_idx present)
    """
    def __init__(self, model: CEXTSLM, cfg: TrainConfig):
        seed_all(cfg.seed)
        self.model = model.to(cfg.device)
        self.cfg = cfg

        # Optimizer & scaler
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

        # Scheduler: cosine with warmup (step-based)
        self.tb = get_tb_writer(cfg.log_dir)
        self.wandb = maybe_init_wandb(cfg.use_wandb, cfg.wandb_project, config={**asdict(cfg), **model.cfg.__dict__})

        self.global_step = 0
        self.best_val = float("inf")
        Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

        # EMA
        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.use_ema else None

        # Placeholder for scheduler, set once we know total_steps
        self.scheduler: Optional[LambdaLR] = None

    # ------------------------------- Sched --------------------------------- #
    def _build_scheduler(self, total_steps: int):
        warmup = self.cfg.warmup_steps
        min_lr = self.cfg.min_lr_scale

        def lr_lambda(step):
            if step < warmup:
                return max(1e-6, float(step + 1) / float(max(1, warmup)))
            # cosine from 1.0 -> min_lr
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr + (1 - min_lr) * cosine

        return LambdaLR(self.opt, lr_lambda=lr_lambda)

    # ------------------------------- Losses -------------------------------- #
    def _compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ts = batch["ts"].to(self.cfg.device)                     # (B,L,d)
        target = batch["target"].to(self.cfg.device)             # (B,H,d)
        docs_tokens = batch["docs_tokens"].to(self.cfg.device)   # (B,M,Ltok)

        # optional bits
        pos_idx = batch.get("pos_idx", None)
        if pos_idx is not None:
            pos_idx = pos_idx.to(self.cfg.device)

        tgt_tokens = batch.get("tgt_tokens", None)
        if tgt_tokens is not None:
            tgt_tokens = tgt_tokens.to(self.cfg.device)

        out = self.model(ts, docs_tokens, tgt_tokens=None)  # we call decoder below if needed

        lf = forecast_loss(out["forecast"], target)

        lc = torch.tensor(0.0, device=self.cfg.device)
        if pos_idx is not None and "attn_logits" in out:
            lc = causal_contrastive_loss(out["attn_logits"], pos_idx, tau=self.cfg.tau)

        le = torch.tensor(0.0, device=self.cfg.device)
        if tgt_tokens is not None:
            # teacher-forced decode against targets
            logits = self.model.decoder(tgt_tokens, out["fused"])
            le = explanation_loss(logits, tgt_tokens)

        total = lf + self.cfg.beta_causal * lc + self.cfg.alpha_explain * le
        return {"total": total, "lf": lf, "lc": lc, "le": le}

    # ------------------------------- Train --------------------------------- #
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        # Infer total steps for scheduler
        total_steps = self.cfg.max_epochs * math.ceil(len(train_loader) / max(1, self.cfg.grad_accum_steps))
        self.scheduler = self._build_scheduler(total_steps)

        patience = 0
        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            running = {"total": 0.0, "lf": 0.0, "lc": 0.0, "le": 0.0}
            for it, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(enabled=(self.cfg.amp and self.cfg.device.startswith("cuda"))):
                    losses = self._compute_losses(batch)
                    loss = losses["total"] / max(1, self.cfg.grad_accum_steps)

                self.scaler.scale(loss).backward()

                if (it + 1) % self.cfg.grad_accum_steps == 0:
                    if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    # EMA update
                    if self.ema is not None:
                        self.ema.update(self.model)

                    # Logging
                    if self.global_step % 10 == 0:
                        for k in running.keys():
                            running[k] += float(losses[k].item())
                        avg = {k: v / 10.0 for k, v in running.items()}
                        self.tb.add_scalar("train/loss_total", avg["total"], self.global_step)
                        self.tb.add_scalar("train/loss_forecast", avg["lf"], self.global_step)
                        self.tb.add_scalar("train/loss_causal", avg["lc"], self.global_step)
                        self.tb.add_scalar("train/loss_explain", avg["le"], self.global_step)
                        self.tb.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.global_step)
                        if self.wandb:
                            self.wandb.log(
                                {"loss_total": avg["total"],
                                 "loss_forecast": avg["lf"],
                                 "loss_causal": avg["lc"],
                                 "loss_explain": avg["le"],
                                 "lr": self.opt.param_groups[0]["lr"]},
                                step=self.global_step,
                            )
                        running = {k: 0.0 for k in running}

                    # Periodic text preview (if decoder tokenizer exists and batch has 'docs' for context)
                    if self.cfg.log_text_every and self.global_step % self.cfg.log_text_every == 0:
                        self._log_qualitative_decode(batch)

                    # Optional save-every
                    if self.cfg.save_every and self.global_step % self.cfg.save_every == 0:
                        self._save_ckpt("last_step")

                    self.global_step += 1

            # Validation (EMA weights if enabled)
            val_mse = None
            if val_loader is not None:
                if self.ema is not None:
                    self.ema.store(self.model)
                    self.ema.copy_to(self.model)
                val_mse = self.validate(val_loader, epoch)
                if self.ema is not None:
                    self.ema.restore(self.model)

                # Early stopping & best ckpt
                improved = val_mse < self.best_val - 1e-7
                if improved:
                    self.best_val = val_mse
                    patience = 0
                    self._save_ckpt("best")
                else:
                    patience += 1
                    if patience >= self.cfg.early_stop_patience:
                        print(f"[EarlyStop] No improvement for {patience} evals. Stopping.")
                        break

            # always keep a 'last' checkpoint each epoch
            self._save_ckpt("last")

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        mse_sum, n = 0.0, 0
        attn_entropy_sum = 0.0
        for batch in loader:
            ts = batch["ts"].to(self.cfg.device)
            target = batch["target"].to(self.cfg.device)
            docs_tokens = batch["docs_tokens"].to(self.cfg.device)

            out = self.model(ts, docs_tokens)
            mse_sum += ((out["forecast"] - target) ** 2).mean().item()
            n += 1

            # attention entropy diagnostic (lower may indicate peaky alignments)
            if "attn_logits" in out:
                logits = out["attn_logits"]  # (B,H,Lt,M)
                p = torch.softmax(logits, dim=-1)
                ent = -(p * (p.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
                attn_entropy_sum += ent

        mse = mse_sum / max(1, n)
        self.tb.add_scalar("val/mse", mse, self.global_step)
        if attn_entropy_sum > 0 and n > 0:
            self.tb.add_scalar("val/attn_entropy", attn_entropy_sum / n, self.global_step)

        if self.wandb:
            logs = {"val_mse": mse, "epoch": epoch}
            if attn_entropy_sum > 0 and n > 0:
                logs["val_attn_entropy"] = attn_entropy_sum / n
            self.wandb.log(logs, step=self.global_step)

        print(f"[Val] epoch={epoch} step={self.global_step} MSE={mse:.6f}")
        return mse

    # ------------------------------- Utils --------------------------------- #
    def _save_ckpt(self, tag: str):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": asdict(self.cfg),
            "global_step": self.global_step,
            "best_val": self.best_val,
        }
        if self.ema is not None:
            ckpt["ema"] = {k: v.cpu() for k, v in self.ema.shadow.items()}
        path = Path(self.cfg.ckpt_dir) / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load_ckpt(self, path: str, load_opt: bool = True, load_ema: bool = True):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if load_opt and "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if self.ema is not None and load_ema and "ema" in ckpt:
            self.ema.shadow = {k: v.to(self.cfg.device) for k, v in ckpt["ema"].items()}
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", float("inf"))
        print(f"[CKPT] Loaded from {path} @ step={self.global_step} best={self.best_val:.6f}")

    def _log_qualitative_decode(self, batch: Dict[str, Any]):
        """
        If the model has a GPT-2 style decoder with a tokenizer, do a greedy decode
        for a single sample and log it. This is *just* for qualitative feedback,
        not a rigorous metric.
        """
        dec = getattr(self.model, "decoder", None)
        tok = getattr(dec, "tokenizer", None)
        if dec is None or tok is None:
            return

        self.model.eval()
        try:
            ts = batch["ts"][:1].to(self.cfg.device)
            docs_tokens = batch["docs_tokens"][:1].to(self.cfg.device)
            fused = self.model(ts, docs_tokens)["fused"]  # (1,Lt,D)

            # prime with BOS/eos if tokenizer supports
            bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            input_ids = torch.tensor([[bos]], device=self.cfg.device)

            # Run backbone.generate if it's a HF model; fallback to a few steps of manual greedy
            backbone = getattr(dec, "backbone", None)
            if backbone is not None and hasattr(backbone, "generate"):
                # simple unconditional generate; conditioning is via fused used during training
                gen = backbone.generate(
                    input_ids=input_ids,
                    max_length=40,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )
                text = tok.decode(gen[0], skip_special_tokens=True)
            else:
                # fallback: pass tokens through dec->logits iteratively
                ids = input_ids.clone()
                for _ in range(40):
                    logits = dec(ids, fused)[:, -1, :]  # (1,V)
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                    ids = torch.cat([ids, nxt], dim=1)
                    if nxt.item() == tok.eos_token_id:
                        break
                text = tok.decode(ids[0], skip_special_tokens=True)

            self.tb.add_text("qual/decoded", text, self.global_step)
            if self.wandb:
                self.wandb.log({"qual_decoded": text}, step=self.global_step)
        except Exception as e:
            # don't crash training for logging
            print(f"[qual_decode] skipped: {e}")
        finally:
            self.model.train()
