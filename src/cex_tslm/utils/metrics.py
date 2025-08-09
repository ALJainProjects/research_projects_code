"""
Metrics for CEX-TSLM.

Includes:
- Forecasting metrics: MSE / MAE / RMSE / MAPE / R^2 with optional masking
- Attention diagnostics: entropy and top-k hit-rate vs. provided positive indices
- Text metrics (optional): BLEU / ROUGE-L if packages are installed
- One-call convenience: `compute_cex_metrics(out, batch, ...)`

Notes
-----
- All functions are torch-first and accept tensors on any device.
- For stability, MAPE has an `eps` to avoid div-by-zero.
- Attention metrics expect `attn_logits` of shape (B, H, Lt, M) and `pos_idx` (B, Lt).
"""
from __future__ import annotations
from typing import Dict, Optional
import math
import torch


# ---------------- Forecast Metrics ---------------- #

def _apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    # Broadcast mask to the last dims of x if needed
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return x * mask


def mse_metric(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Mean Squared Error.
    pred, target: (..., d)
    mask:         (...) or None
    """
    diff2 = (pred - target) ** 2
    diff2 = _apply_mask(diff2, mask)
    if mask is None:
        return diff2.mean().item()
    denom = mask.sum().clamp_min(1.0)
    return (diff2.sum() / denom).item()


def mae_metric(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    diff = (pred - target).abs()
    diff = _apply_mask(diff, mask)
    if mask is None:
        return diff.mean().item()
    denom = mask.sum().clamp_min(1.0)
    return (diff.sum() / denom).item()


def rmse_metric(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    return math.sqrt(max(mse_metric(pred, target, mask), 0.0))


def mape_metric(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error in [0, +inf).
    Adds eps to denominator to avoid div-by-zero.
    """
    pct = ((pred - target).abs() / (target.abs() + eps))
    pct = _apply_mask(pct, mask)
    if mask is None:
        return pct.mean().item()
    denom = mask.sum().clamp_min(1.0)
    return (pct.sum() / denom).item()


def r2_score(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Coefficient of determination R^2 over the last dimension.
    """
    if mask is not None:
        # compute masked mean
        masked_target = _apply_mask(target, mask)
        denom = mask.sum().clamp_min(1.0)
        mean_y = masked_target.sum(dim=0, keepdim=True) / denom
    else:
        mean_y = target.mean(dim=0, keepdim=True)

    ss_res = _apply_mask((target - pred) ** 2, mask).sum()
    ss_tot = _apply_mask((target - mean_y) ** 2, mask).sum()
    if ss_tot.item() == 0.0:
        return 0.0
    return (1.0 - (ss_res / ss_tot)).item()


# ---------------- Attention Diagnostics ---------------- #

@torch.no_grad()
def attention_entropy(attn_logits: torch.Tensor) -> float:
    """
    Average token-level entropy across batch/heads/queries.
    attn_logits: (B, H, Lt, M)
    """
    p = torch.softmax(attn_logits, dim=-1).clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # (B,H,Lt)
    return ent.mean().item()


@torch.no_grad()
def attention_topk_hit_rate(attn_logits: torch.Tensor, pos_idx: torch.Tensor, k: int = 1) -> float:
    """
    Fraction of time queries where the positive doc index is in top-k by attention.

    attn_logits: (B, H, Lt, M)
    pos_idx:     (B, Lt)  in [0, M-1]
    """
    # average heads to a single distribution
    sim = attn_logits.mean(dim=1)  # (B, Lt, M)
    topk = sim.topk(k=min(k, sim.size(-1)), dim=-1).indices  # (B,Lt,k)
    pos = pos_idx.unsqueeze(-1)  # (B,Lt,1)
    hit = (topk == pos).any(dim=-1).float()  # (B,Lt)
    return hit.mean().item()


@torch.no_grad()
def attention_kl_to_onehot(attn_logits: torch.Tensor, pos_idx: torch.Tensor, eps: float = 1e-8) -> float:
    """
    KL(OneHot(pos) || p_attn). Lower is better (peaky on the correct doc).
    """
    p = torch.softmax(attn_logits, dim=-1).clamp_min(eps)  # (B,H,Lt,M)
    p = p.mean(dim=1)                                      # (B,Lt,M)
    B, Lt, M = p.shape
    oh = torch.zeros(B, Lt, M, device=p.device)
    pos_clamped = pos_idx.clamp_min(0).clamp_max(M - 1)
    oh.scatter_(-1, pos_clamped.unsqueeze(-1), 1.0)
    # KL(onehot||p) reduces to -log p[pos]
    gather_p = p.gather(-1, pos_clamped.unsqueeze(-1)).clamp_min(eps)
    kl = (-gather_p.log()).mean().item()
    return kl


# ---------------- Optional Text Metrics ---------------- #

def bleu1(reference: str, hypothesis: str) -> float:
    """
    BLEU-1 unigram precision. Returns [0,1].
    Tries sacrebleu/nltk if available; else provides a tiny fallback.
    """
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu([hypothesis], [[reference]], force=True, effective_order=True).score / 100.0
    except Exception:
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method1
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            return float(sentence_bleu([ref_tokens], hyp_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie))
        except Exception:
            # fallback: token overlap
            ref = set(reference.split())
            hyp = set(hypothesis.split())
            return (len(ref & hyp) / max(1, len(hyp)))


def rouge_l(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L (LCS-based F). Tries rouge-score if available; else a minimal LCS impl.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        # tiny LCS-based F1
        a, b = reference.split(), hypothesis.split()
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[n][m]
        prec = lcs / max(1, m)
        rec  = lcs / max(1, n)
        if prec+rec == 0:
            return 0.0
        return 2*prec*rec/(prec+rec)


# ---------------- Convenience Aggregator ---------------- #

@torch.no_grad()
def compute_cex_metrics(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    attn_topk: int = 1,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper to compute a suite of metrics given model output & batch.

    Expects:
      out["forecast"]: (B,H,d)
      out["attn_logits"]: (B,Ht,Lt,M) (optional)
      batch["target"]: (B,H,d)
      batch["pos_idx"]: (B,Lt) (optional)
      mask: optional mask over target (B,H) or (B,H,d)

    Returns a dict of floats.
    """
    pred = out["forecast"]
    target = batch["target"]
    metrics = {
        "mse": mse_metric(pred, target, mask),
        "mae": mae_metric(pred, target, mask),
        "rmse": rmse_metric(pred, target, mask),
        "mape": mape_metric(pred, target, mask),
        "r2": r2_score(pred, target, mask),
    }

    # Attention diagnostics if available
    attn = out.get("attn_logits", None)
    pos_idx = batch.get("pos_idx", None)
    if attn is not None:
        metrics["attn_entropy"] = attention_entropy(attn)
        if pos_idx is not None:
            metrics["attn_top{}_hit".format(attn_topk)] = attention_topk_hit_rate(attn, pos_idx, k=attn_topk)
            metrics["attn_kl_onehot"] = attention_kl_to_onehot(attn, pos_idx)

    return metrics
