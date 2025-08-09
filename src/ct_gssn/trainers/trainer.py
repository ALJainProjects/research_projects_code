# src/ct_gssn/trainers/trainer.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ct_gssn.models.ct_gssn import (
    CTGSSN,
    CTGSSNConfig,
    masked_prediction_loss,
    lyapunov_stability_loss,
    GraphLinkPredictor,
)
from ct_gssn.utils.helpers import (
    seed_all,
    get_tb_writer,
    maybe_init_wandb,
    save_config_snapshot,
)
from ct_gssn.utils.metrics import mse_metric, mae_metric  # add MAE to utils if missing
from ct_gssn.utils.plotting import plot_series_plotly


# ---------------------------------------------------------------------
# Training configuration with sensible, opt-in extras
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    # data
    batch_size: int = 16
    num_workers: int = 0

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    # AMP + EMA
    use_amp: bool = True
    ema_decay: float = 0.0  # 0 disables EMA; try 0.999 for large models

    # schedulers
    scheduler: str = "none"  # "none" | "cosine" | "onecycle"
    warmup_steps: int = 0
    max_steps_override: int = 0  # set >0 to override computed steps (e.g., for cosine)

    # losses
    lambda_graph: float = 0.1
    lambda_stable: float = 0.01
    mask_ratio: float = 0.15  # fraction of observed targets masked for prediction

    # early stopping / checkpoints
    early_stop_patience: int = 0  # 0 disables early stopping
    ckpt_dir: str = "checkpoints/ct_gssn"
    save_best_k: int = 1

    # logging
    log_dir: str = "runs/ct_gssn"
    log_every: int = 10
    val_every: int = 1  # epochs
    use_wandb: bool = False
    wandb_project: str = "ct_gssn"
    seed: int = 1337

    # model aux toggles at train time (forward compatibility)
    compute_aux: bool = True  # collects A/c for stability + graph heads
    use_pad_mask: bool = False  # pass pad_mask from loader batch if provided


# ---------------------------------------------------------------------
# Utilities: EMA, schedulers
# ---------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.decay <= 0:
            return
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for k, p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                backup[k] = p.detach().clone()
                p.copy_(self.shadow[k])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for k, buf in backup.items():
            model.state_dict()[k].copy_(buf)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_total: int):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        # cosine with warmup
        def lr_lambda(step):
            if step < cfg.warmup_steps and cfg.warmup_steps > 0:
                return (step + 1) / cfg.warmup_steps
            t = step - cfg.warmup_steps
            T = max(1, steps_total - cfg.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * t / T))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=steps_total, pct_start=max(1, cfg.warmup_steps) / max(1, steps_total)
        )
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class CTGSSNTrainer:
    """
    Full-featured trainer:
      - masked observation prediction loss
      - link prediction loss on c (edge inference proxy)
      - Lyapunov stability regularizer on A
      - AMP mixed precision + grad accumulation + grad clipping
      - LR scheduling (cosine/onecycle) + warmup
      - EMA weights (optional)
      - Early stopping & checkpointing (top-K)
      - TensorBoard and optional Weights & Biases logging
    """

    def __init__(self, model_cfg: CTGSSNConfig, train_cfg: TrainConfig):
        seed_all(train_cfg.seed)
        self.device = train_cfg.device
        # ensure CTGSSN collects aux if we need stability/graph losses
        model_cfg.compute_aux = train_cfg.compute_aux
        self.model = CTGSSN(model_cfg).to(self.device)
        self.link_pred = GraphLinkPredictor(c_dim=model_cfg.c_dim).to(self.device)

        self.opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.link_pred.parameters()),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.cfg = train_cfg
        self.tb = get_tb_writer(train_cfg.log_dir)
        self.wandb = maybe_init_wandb(
            train_cfg.use_wandb, train_cfg.wandb_project, config={**asdict(model_cfg), **asdict(train_cfg)}
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)
        self.ema = EMA(self.model, decay=train_cfg.ema_decay) if train_cfg.ema_decay > 0 else None

        # runtime stats
        self._global_step = 0
        self._best_metrics: List[Tuple[float, str]] = []  # (val_mse, path)

    # --------------------------- IO helpers ---------------------------

    def _save_checkpoint(self, epoch: int, val_mse: float, tag: str = "latest") -> str:
        import os
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        path = f"{self.cfg.ckpt_dir}/ctgssn_epoch{epoch:03d}_{tag}.pt"
        payload = {
            "model": self.model.state_dict(),
            "link_pred": self.link_pred.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self._global_step,
            "val_mse": val_mse,
            "cfg": asdict(self.cfg),
        }
        if self.ema:
            payload["ema"] = {k: v.cpu() for k, v in self.ema.shadow.items()}
        torch.save(payload, path)
        return path

    # Keep top-K checkpoints by val MSE
    def _maybe_keep_topk(self, ckpt_path: str, val_mse: float):
        import os
        self._best_metrics.append((val_mse, ckpt_path))
        self._best_metrics.sort(key=lambda x: x[0])  # lower MSE is better
        while len(self._best_metrics) > max(1, self.cfg.save_best_k):
            _, path_to_rm = self._best_metrics.pop(-1)
            try:
                os.remove(path_to_rm)
            except OSError:
                pass

    # ------------------------ training internals ----------------------

    @staticmethod
    def _prepare_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        return out

    @staticmethod
    def _safe_mask_targets(y: torch.Tensor, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Randomly select a subset of observed positions to supervise. Works per-batch
        even when some items have no observations (rare but possible).
        """
        B, N, L, _ = y.shape
        obs = (mask > 0).reshape(B, -1)  # (B, N*L)
        target_mask = torch.zeros_like(obs, dtype=y.dtype, device=y.device)
        for b in range(B):
            idx = obs[b].nonzero(as_tuple=False).flatten()
            if len(idx) == 0:
                continue
            k = max(1, int(ratio * len(idx)))
            perm = torch.randperm(len(idx), device=y.device)[:k]
            chosen = idx[perm]
            target_mask[b, chosen] = 1.0
        return target_mask.view(B, N, L)

    @staticmethod
    def _sample_edges_for_link_pred(adj: torch.Tensor, num_neg: int = 2) -> Dict[str, torch.Tensor]:
        """
        Build a small edge set per batch for link prediction: positives from adj>0, negatives random.
        adj: (B,N,N)
        """
        B, N, _ = adj.shape
        Epos, Eneg = [], []
        device = adj.device
        for b in range(B):
            pos = (adj[b] > 0).nonzero(as_tuple=False)
            pos = pos[pos[:, 0] != pos[:, 1]]
            if len(pos) == 0:
                pos = torch.tensor([[0, 1]], device=device)
            # negatives
            all_pairs = torch.cartesian_prod(torch.arange(N, device=device), torch.arange(N, device=device))
            all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]
            neg_mask = (adj[b][all_pairs[:, 0], all_pairs[:, 1]] == 0)
            neg = all_pairs[neg_mask]
            if len(neg) == 0:
                neg = torch.tensor([[0, 2]], device=device)
            neg = neg[torch.randperm(len(neg))[:min(len(pos) * num_neg, len(neg))]]
            Epos.append(pos)
            Eneg.append(neg)
        max_pos = max(1, max(map(len, Epos)))
        max_neg = max(1, max(map(len, Eneg)))
        pos_edges = torch.stack([F.pad(p, (0, 0, 0, max_pos - len(p)), value=0) for p in Epos], dim=0)
        neg_edges = torch.stack([F.pad(n, (0, 0, 0, max_neg - len(n)), value=0) for n in Eneg], dim=0)
        return {"pos": pos_edges, "neg": neg_edges}

    # ---------------------------- train loop --------------------------

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        save_config_snapshot(self.cfg.log_dir, {"model": self.model.cfg.__dict__, "train": self.cfg.__dict__})

        # steps scheduling
        steps_per_epoch = max(1, math.ceil(len(train_loader.dataset) / max(1, self.cfg.batch_size)))
        total_steps = self.cfg.max_steps_override or (steps_per_epoch * max(1, self.cfg.max_epochs))
        scheduler = _build_scheduler(self.opt, self.cfg, total_steps)

        best_val = float("inf")
        no_improve = 0

        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            start_t = time.time()
            running = {"loss": 0.0, "lp": 0.0, "lg": 0.0, "ls": 0.0, "mse": 0.0, "mae": 0.0}
            num_batches = 0

            for it, raw in enumerate(train_loader):
                batch = self._prepare_batch(raw, self.device)
                x = batch["x"]          # (B,N,L,P)
                y = batch["y"]          # (B,N,L,Q)
                mask = batch["mask"]    # (B,N,L)
                deltas = batch["deltas"]
                adj = batch["adj"]
                pad_mask = batch.get("pad_mask") if self.cfg.use_pad_mask else None

                target_mask = self._safe_mask_targets(y, mask, self.cfg.mask_ratio)  # (B,N,L)

                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    y_pred, aux = self.model(
                        x, mask, deltas, adj, return_aux=self.cfg.compute_aux, pad_mask=pad_mask
                    )

                    # Losses
                    lp = masked_prediction_loss(y_pred, y, target_mask)

                    if self.cfg.compute_aux:
                        # Link prediction from c (use last step c from last layer as proxy)
                        c_last = aux["c"][-1, :, -1]  # (B,N,C)
                        edges = self._sample_edges_for_link_pred(adj if adj.dim() == 3 else adj[:, -1], num_neg=2)
                        logits_pos = self.link_pred(c_last, edges["pos"])
                        logits_neg = self.link_pred(c_last, edges["neg"])
                        labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=1)
                        logits = torch.cat([logits_pos, logits_neg], dim=1)
                        lg = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                        ls = lyapunov_stability_loss(aux["A"])
                    else:
                        lg = torch.tensor(0.0, device=x.device)
                        ls = torch.tensor(0.0, device=x.device)

                    loss = lp + self.cfg.lambda_graph * lg + self.cfg.lambda_stable * ls
                    loss = loss / max(1, self.cfg.grad_accum_steps)

                self.scaler.scale(loss).backward()

                # step every grad_accum_steps
                if (it + 1) % max(1, self.cfg.grad_accum_steps) == 0:
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)
                    self._global_step += 1

                # metrics (non-accumulated)
                with torch.no_grad():
                    mse = mse_metric(y_pred, y, mask)
                    try:
                        mae = mae_metric(y_pred, y, mask)
                    except Exception:
                        mae = torch.sqrt(mse)  # fallback

                # running stats
                running["loss"] += float(loss.detach()) * max(1, self.cfg.grad_accum_steps)
                running["lp"] += float(lp.detach())
                running["lg"] += float(lg.detach())
                running["ls"] += float(ls.detach())
                running["mse"] += float(mse)
                running["mae"] += float(mae)
                num_batches += 1

                # Logging
                if self._global_step % max(1, self.cfg.log_every) == 0:
                    avg = {k: v / max(1, num_batches) for k, v in running.items()}
                    lr_val = self.opt.param_groups[0]["lr"]
                    self.tb.add_scalar("train/loss_total", avg["loss"], self._global_step)
                    self.tb.add_scalar("train/loss_predict", avg["lp"], self._global_step)
                    self.tb.add_scalar("train/loss_graph", avg["lg"], self._global_step)
                    self.tb.add_scalar("train/loss_stable", avg["ls"], self._global_step)
                    self.tb.add_scalar("train/mse", avg["mse"], self._global_step)
                    self.tb.add_scalar("train/mae", avg["mae"], self._global_step)
                    self.tb.add_scalar("train/lr", lr_val, self._global_step)
                    if self.wandb:
                        self.wandb.log(
                            {
                                "loss_total": avg["loss"],
                                "loss_predict": avg["lp"],
                                "loss_graph": avg["lg"],
                                "loss_stable": avg["ls"],
                                "mse": avg["mse"],
                                "mae": avg["mae"],
                                "lr": lr_val,
                                "epoch": epoch,
                            },
                            step=self._global_step,
                        )

            epoch_time = time.time() - start_t
            self.tb.add_scalar("time/epoch_seconds", epoch_time, self._global_step)

            # quick val
            if val_loader is not None and ((epoch + 1) % max(1, self.cfg.val_every) == 0):
                val_mse, val_mae = self.validate(val_loader, epoch, self._global_step)
                # Save checkpoints (raw)
                ckpt_path = self._save_checkpoint(epoch, val_mse, tag="latest")
                # EMA evaluation (if enabled)
                if self.ema:
                    backup = self.ema.store(self.model)
                    val_mse_ema, val_mae_ema = self.validate(val_loader, epoch, self._global_step, tag="ema")
                    self.ema.restore(self.model, backup)
                    # Track better of raw vs EMA
                    tracked_mse = min(val_mse, val_mse_ema)
                else:
                    tracked_mse = val_mse

                # keep top-K
                self._maybe_keep_topk(ckpt_path, tracked_mse)

                # early stopping
                if tracked_mse + 1e-12 < best_val:
                    best_val = tracked_mse
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.cfg.early_stop_patience > 0 and no_improve >= self.cfg.early_stop_patience:
                        print(f"[EarlyStop] No improvement for {no_improve} evals. Best MSE={best_val:.6f}.")
                        break

    # ------------------------------ eval ------------------------------

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int, global_step: int, tag: str = "val") -> Tuple[float, float]:
        self.model.eval()
        mses, maes = [], []
        for batch in loader:
            b = self._prepare_batch(batch, self.device)
            y_pred, _ = self.model(
                b["x"], b["mask"], b["deltas"], b["adj"], return_aux=False, pad_mask=(b.get("pad_mask") if self.cfg.use_pad_mask else None)
            )
            mses.append(mse_metric(y_pred, b["y"], b["mask"]))
            try:
                maes.append(mae_metric(y_pred, b["y"], b["mask"]))
            except Exception:
                maes.append(math.sqrt(mses[-1]))

        val_mse = float(sum(mses) / max(1, len(mses)))
        val_mae = float(sum(maes) / max(1, len(maes)))

        self.tb.add_scalar(f"{tag}/mse", val_mse, global_step)
        self.tb.add_scalar(f"{tag}/mae", val_mae, global_step)
        if self.wandb:
            self.wandb.log({f"{tag}_mse": val_mse, f"{tag}_mae": val_mae, "epoch": epoch}, step=global_step)

        # Log a tiny plot for the first sample/node (best effort)
        try:
            y_pred_s = y_pred[0, 0, :, 0].detach().cpu()
            y_true_s = b["y"][0, 0, :, 0].detach().cpu()
            fig = plot_series_plotly(y_pred_s, y_true_s)
            self.tb.add_text(f"{tag}/sample_plot", "Sample forecast plotted.", global_step)
            if self.wandb:
                import wandb
                self.wandb.log({f"{tag}_plot": wandb.Plotly(fig)}, step=global_step)
        except Exception:
            pass

        print(f"[{tag}] epoch={epoch}  MSE={val_mse:.6f}  MAE={val_mae:.6f}")
        return val_mse, val_mae
