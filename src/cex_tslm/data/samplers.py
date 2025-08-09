"""
Simple positive/negative sampler utilities for causal contrastive supervision.

These helpers produce index tensors you can feed into
`causal_contrastive_loss(attn_logits, pos_idx, ...)` from CEX-TSLM.

Key features
------------
- Works with batched inputs.
- Respects `doc_mask` (which docs exist per sample).
- Supports deterministic seeding for reproducibility.
- Can emit multiple hard negatives (and guarantee they differ from positives).
- Can compute the time-attention length from raw ts lengths + patch length.

Typical usage
-------------
# Suppose your collate produced:
#   doc_mask: [B, M] (bool)  — True for real docs
#   ts_len:   [B]     — raw time steps per sample
# If your TS encoder uses patches of length P, attention length is ceil(ts_len / P)

from cex_tslm.utils.sampling import (
    attn_len_from_patch, sample_pos_indices, sample_neg_indices
)

Lt = attn_len_from_patch(ts_len, patch_len=P)
pos_idx = sample_pos_indices(attn_len=Lt.max().item(), M=doc_mask.size(1),
                             batch_size=doc_mask.size(0), doc_mask=doc_mask)

negs = sample_neg_indices(M=doc_mask.size(1), num_neg=4,
                          pos_idx=pos_idx, doc_mask=doc_mask)
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import torch


def _maybe_seed(seed: Optional[int]) -> torch.Generator:
    """
    Create a torch.Generator optionally seeded for reproducibility.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def attn_len_from_patch(ts_len: torch.Tensor, patch_len: int) -> torch.Tensor:
    """
    Convert raw per-sample time lengths to attention lengths used by a patch encoder.

    Args:
        ts_len:    LongTensor [B] raw L per sample
        patch_len: int, e.g., 4

    Returns:
        LongTensor [B] with ceil(ts_len / patch_len)
    """
    if not torch.is_tensor(ts_len):
        ts_len = torch.tensor(ts_len, dtype=torch.long)
    return (ts_len + patch_len - 1) // patch_len


def _uniform_valid_doc_indices(
    B: int,
    M: int,
    doc_mask: Optional[torch.Tensor],
    attn_len: int,
    g: torch.Generator,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Helper: uniformly sample positive doc indices for each (b, t) from the set of valid docs.
    Returns LongTensor [B, attn_len], filled with -1 where no valid docs exist.
    """
    device = device or (doc_mask.device if doc_mask is not None else torch.device("cpu"))
    pos = torch.full((B, attn_len), -1, dtype=torch.long, device=device)

    if doc_mask is None:
        # All docs are valid; sample from [0, M-1]
        if M == 0:
            return pos
        pos = torch.randint(low=0, high=M, size=(B, attn_len), generator=g, device=device)
        return pos

    # With mask: per-sample valid set may differ
    for b in range(B):
        valid = torch.nonzero(doc_mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        choice = valid[torch.randint(low=0, high=valid.numel(), size=(attn_len,), generator=g, device=device)]
        pos[b] = choice
    return pos


def sample_pos_indices(
    attn_len: int,
    M: int,
    batch_size: int = 1,
    *,
    doc_mask: Optional[torch.Tensor] = None,
    strategy: str = "uniform",
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample positive document indices for each query position.

    Args:
        attn_len:   number of time-attention queries per sample (Lt)
        M:          number of docs per sample (padded to batch max)
        batch_size: B
        doc_mask:   optional BoolTensor [B, M]; True for real docs, False for padded
        strategy:   currently only "uniform" is implemented
        seed:       optional int for determinism
        device:     torch device

    Returns:
        LongTensor [B, attn_len] with values in [0, M-1], or -1 if no valid docs.
    """
    if strategy != "uniform":
        raise NotImplementedError(f"strategy='{strategy}' not implemented (use 'uniform').")
    g = _maybe_seed(seed)
    return _uniform_valid_doc_indices(batch_size, M, doc_mask, attn_len, g, device=device)


def sample_neg_indices(
    M: int,
    num_neg: int,
    *,
    pos_idx: torch.Tensor,
    doc_mask: Optional[torch.Tensor] = None,
    allow_repeat: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample *negative* doc indices for contrastive training.

    Args:
        M:          number of docs per sample
        num_neg:    number of negatives per (b, t)
        pos_idx:    LongTensor [B, attn_len] with positives; -1 indicates "no valid doc"
        doc_mask:   optional BoolTensor [B, M] valid-doc mask (False excludes)
        allow_repeat: if True, negatives may repeat; if False, try to avoid repeats
        seed:       optional seed

    Returns:
        LongTensor [B, attn_len, num_neg] with values in [0, M-1], or -1 where impossible.
        Guarantees negatives differ from the corresponding positive when possible.
    """
    assert pos_idx.dim() == 2, "pos_idx must be [B, attn_len]"
    B, Lt = pos_idx.shape
    g = _maybe_seed(seed)

    device = pos_idx.device
    neg = torch.full((B, Lt, num_neg), -1, dtype=torch.long, device=device)

    if M == 0:
        return neg

    # Precompute candidate sets per sample
    if doc_mask is None:
        doc_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    for b in range(B):
        valid = torch.nonzero(doc_mask[b], as_tuple=False).flatten()  # [Mv]
        Mv = valid.numel()
        if Mv == 0:
            continue

        for t in range(Lt):
            p = pos_idx[b, t].item()
            if p < 0 or p >= M:
                # no positive => just sample arbitrary valid docs (still called "negatives")
                if allow_repeat:
                    choice = valid[torch.randint(0, Mv, (num_neg,), generator=g, device=device)]
                    neg[b, t] = choice
                else:
                    if num_neg <= Mv:
                        choice = valid[torch.randperm(Mv, generator=g, device=device)[:num_neg]]
                    else:
                        # Not enough unique candidates, fill what we can
                        choice = valid[torch.randperm(Mv, generator=g, device=device)]
                        rep = valid[torch.randint(0, Mv, (num_neg - Mv,), generator=g, device=device)]
                        choice = torch.cat([choice, rep], dim=0)
                    neg[b, t] = choice
                continue

            # Form candidate pool that excludes positive p
            # (only if p is within the valid set; else fallback to 'valid')
            if doc_mask[b, p]:
                cand = valid[valid != p]
            else:
                cand = valid

            Cv = cand.numel()
            if Cv == 0:
                # Can't exclude positive; fallback to valid set (duplicates allowed)
                cand = valid
                Cv = Mv

            if allow_repeat:
                choice = cand[torch.randint(0, Cv, (num_neg,), generator=g, device=device)]
            else:
                if num_neg <= Cv:
                    choice = cand[torch.randperm(Cv, generator=g, device=device)[:num_neg]]
                else:
                    # Not enough unique negatives; take all unique then fill with random
                    choice = cand[torch.randperm(Cv, generator=g, device=device)]
                    fill = cand[torch.randint(0, Cv, (num_neg - Cv,), generator=g, device=device)]
                    choice = torch.cat([choice, fill], dim=0)
            neg[b, t] = choice

    return neg


def expand_pos_for_heads(
    pos_idx: torch.Tensor,
    n_heads: int,
) -> torch.Tensor:
    """
    Expand [B, Lt] -> [B, H, Lt] to supervise per-head or to broadcast to heads.

    Args:
        pos_idx: [B, Lt]  (values in [-1, M-1])
        n_heads: number of attention heads

    Returns:
        [B, n_heads, Lt]
    """
    assert pos_idx.dim() == 2, "pos_idx must be [B, Lt]"
    return pos_idx.unsqueeze(1).repeat(1, n_heads, 1)


def make_pos_index(
    attn_len: int,
    M: int,
) -> torch.Tensor:
    """
    Backward-compatible convenience wrapper (OLD API).

    Create a dummy positive index tensor of shape (1, attn_len) with values in [0, M-1].
    Returns -1 if M == 0.
    """
    if M <= 0:
        return torch.full((1, attn_len), -1, dtype=torch.long)
    return torch.randint(low=0, high=M, size=(1, attn_len), dtype=torch.long)
