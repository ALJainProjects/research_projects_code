"""
Time-MMD dataset loader + rich collate utilities for CEX-TSLM.

It supports two text tokenization paths:
  1) "simple"   — a tiny, whitespace-based tokenizer (no external deps)
  2) "hf-*"     — HuggingFace tokenizers (e.g., bert-base-uncased, gpt2)

Input files (created by `scripts/convert_time_mmd.py`):
  data/time_mmd_proc/{train,val,test}.jsonl

Each JSON line:
  {
    "ts": [[...], ...],         # (L x d_x) float
    "docs": ["text", ...],      # list[str], variable length
    "target": [[...], ...],     # (H x d_x) float
    "pos_idx": [i0, i1, ...]    # optional positives aligned to time windows
  }

Returned batch dict (common keys):
  - ts:           (B, L, d_x) float (zero-padded to max L in the batch)
  - target:       (B, H, d_x) float (zero-padded to max H in the batch)
  - ts_len:       (B,)        long, original time length per sample
  - doc_mask:     (B, M)      bool, True for real docs (not padded)
  - pos_idx:      (B, Lt)     long, per-time positive doc index (Lt ~= ceil(L/patch_len))
  - attn_len:     (B,)        long, Lt per sample given `patch_len`

For "simple" tokenizer:
  - docs_tokens:  (B, M, Ltok) long, 0 is pad id
  - docs_attn:    (B, M, Ltok) bool, True for non-pad tokens

For "hf" tokenizer (BERT/GPT-2/etc.):
  - input_ids:        (B, M, Ltok) long
  - attention_mask:   (B, M, Ltok) long/bool
  - (optional) token_type_ids if the tokenizer returns it
  - docs_tokens is ALSO provided as an alias of input_ids for convenience

Notes for GPT-2 family:
  - GPT-2 has no default PAD token; we set `pad_token` to `eos_token` if missing,
    so sequences can be batched safely.

You can set `sample_missing_pos=True` to auto-sample positives when a row
doesn't include `pos_idx`.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset


# ------------------------------- Dataset ------------------------------- #

class TimeMMDDataset(Dataset):
    def __init__(self, root: str, split: str = "train", max_docs: int = 32):
        """
        Args:
            root:     directory with {split}.jsonl
            split:    'train'|'val'|'test'
            max_docs: keep at most this many docs per sample (truncate if longer)
        """
        super().__init__()
        self.root = Path(root)
        self.path = self.root / f"{split}.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing {self.path}. Run scripts/convert_time_mmd.py --synthesize to create a toy set."
            )
        self.rows: List[Dict[str, Any]] = [json.loads(l) for l in self.path.open("r", encoding="utf-8")]
        self.max_docs = int(max_docs)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        ts = torch.tensor(r["ts"], dtype=torch.float32)           # (L, d)
        target = torch.tensor(r["target"], dtype=torch.float32)   # (H, d)
        docs: List[str] = list(map(str, r.get("docs", [])))[: self.max_docs]
        pos_idx = r.get("pos_idx", None)  # optional
        return {"ts": ts, "target": target, "docs": docs, "pos_idx": pos_idx}


# ------------------------------- Collate ------------------------------- #

@dataclass
class CollateCfg:
    # Text tokenization mode: "simple" or "hf"
    text_mode: str = "simple"
    # HF tokenizer (required when text_mode="hf")
    hf_tokenizer: Any = None
    # For GPT-2-like tokenizers without PAD token, set pad to eos.
    set_hf_pad_to_eos: bool = True
    # Token-level truncation/padding length
    max_tokens: int = 64
    # (Re-)limit docs per sample at collate time (use None to keep dataset's cap)
    max_docs: Optional[int] = None
    # Whether to return `doc_mask` (which docs are real vs padded)
    return_doc_mask: bool = True
    # Whether to sample positives when missing in the row
    sample_missing_pos: bool = True
    # Patch length used by the TS encoder (for computing Lt)
    patch_len: int = 4
    # Seed for any random sampling (positives)
    seed: Optional[int] = None


def _maybe_seed(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def _simple_tokenize_batch(
    docs_batch: List[List[str]],
    max_tokens: int,
    vocab_size: int = 50_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tiny per-batch tokenizer:
      - lowercases + splits on whitespace
      - builds a one-off vocab per batch; 0 is pad, 1..V-1 are word indices
    Returns:
      input_ids:     (B, M, Ltok) long
      attention_mask:(B, M, Ltok) bool
    """
    from collections import defaultdict

    word2id = {"<PAD>": 0}
    next_id = 1
    encoded: List[List[List[int]]] = []
    attn: List[List[List[int]]] = []

    for docs in docs_batch:
        doc_ids, doc_attn = [], []
        for s in docs:
            ids = []
            for tok in s.lower().split()[:max_tokens]:
                if tok not in word2id:
                    if next_id < vocab_size:
                        word2id[tok] = next_id
                        next_id += 1
                    else:
                        # send unseen to last id
                        word2id[tok] = vocab_size - 1
                ids.append(word2id[tok])
            # pad/truncate
            ids = ids[:max_tokens]
            mask = [1] * len(ids)
            if len(ids) < max_tokens:
                pad = max_tokens - len(ids)
                ids = ids + [0] * pad
                mask = mask + [0] * pad
            doc_ids.append(ids)
            doc_attn.append(mask)
        encoded.append(doc_ids)
        attn.append(doc_attn)

    input_ids = torch.tensor(encoded, dtype=torch.long)
    attention_mask = torch.tensor(attn, dtype=torch.bool)
    return input_ids, attention_mask


def _pad_docs_to_max(batch_docs: List[List[str]], max_docs: Optional[int]) -> Tuple[List[List[str]], torch.Tensor]:
    """
    Pad the list of docs per sample up to M = max(len(docs_i)) or `max_docs` (if provided).
    Returns:
      padded_docs: list of list[str] with right-padding by "" (empty string)
      doc_mask:    BoolTensor (B, M) True if real doc
    """
    B = len(batch_docs)
    M = max(len(d) for d in batch_docs)
    if max_docs is not None:
        M = min(M, int(max_docs))
    padded, mask = [], torch.zeros(B, M, dtype=torch.bool)
    for i, docs in enumerate(batch_docs):
        docs = docs[:M]
        m = len(docs)
        padded_i = docs + [""] * (M - m)
        padded.append(padded_i)
        mask[i, :m] = True
    return padded, mask


def _compute_ts_pad(batch_ts: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-pad time-series to the max L in the batch.

    Returns:
      ts:     (B, L, d)
      ts_len: (B,)
    """
    B = len(batch_ts)
    Lmax = max(t.shape[0] for t in batch_ts)
    d = batch_ts[0].shape[1]
    ts = torch.zeros(B, Lmax, d, dtype=torch.float32)
    ts_len = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(batch_ts):
        Li = t.shape[0]
        ts[i, :Li] = t
        ts_len[i] = Li
    return ts, ts_len


def _compute_target_pad(batch_target: List[torch.Tensor]) -> torch.Tensor:
    """
    Zero-pad targets to max H in the batch.

    Returns:
      target: (B, H, d)
    """
    B = len(batch_target)
    Hmax = max(t.shape[0] for t in batch_target)
    d = batch_target[0].shape[1]
    tgt = torch.zeros(B, Hmax, d, dtype=torch.float32)
    for i, t in enumerate(batch_target):
        Hi = t.shape[0]
        tgt[i, :Hi] = t
    return tgt


def _attn_len_from_patch(ts_len: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Lt = ceil(ts_len / patch_len) per sample."""
    return (ts_len + patch_len - 1) // patch_len


def _sample_pos_indices(
    B: int,
    Lt: int,
    M: int,
    doc_mask: torch.Tensor,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Uniformly sample a positive doc index for each (b,t) among valid docs.
    Returns LongTensor [B, Lt] with -1 when a sample has no valid docs.
    """
    device = device or doc_mask.device
    g = _maybe_seed(seed)
    pos = torch.full((B, Lt), -1, dtype=torch.long, device=device)

    for b in range(B):
        valid = torch.nonzero(doc_mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        choice = valid[torch.randint(0, valid.numel(), (Lt,), generator=g, device=device)]
        pos[b] = choice
    return pos


def collate_time_mmd(
    batch: List[Dict[str, Any]],
    *,
    cfg: CollateCfg = CollateCfg(),
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of Time-MMD rows into model-ready tensors.

    Args:
        batch: list of samples from TimeMMDDataset
        cfg:   CollateCfg with tokenization + padding options (see class docstring)

    Returns:
        A dict with common keys and text-specific tensors as described in the module docstring.
    """
    # 1) Pad TS + targets
    ts, ts_len = _compute_ts_pad([b["ts"] for b in batch])
    target = _compute_target_pad([b["target"] for b in batch])

    # 2) Pad docs to common M (or cfg.max_docs)
    raw_docs = [b["docs"] for b in batch]
    docs, doc_mask = _pad_docs_to_max(raw_docs, cfg.max_docs)

    # 3) Tokenize docs
    if cfg.text_mode == "simple":
        input_ids, attention_mask = _simple_tokenize_batch(docs, max_tokens=cfg.max_tokens)
        token_type_ids = None
    elif cfg.text_mode == "hf":
        assert cfg.hf_tokenizer is not None, "When `text_mode='hf'`, you must pass `hf_tokenizer` in CollateCfg."
        tok = cfg.hf_tokenizer

        # GPT-2 compatibility: set PAD to EOS if missing
        if cfg.set_hf_pad_to_eos and getattr(tok, "pad_token", None) is None:
            if getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token

        # Flatten (B, M) -> (B*M) docs list for a single tokenizer call, then reshape
        flat_docs: List[str] = [d for docs_i in docs for d in docs_i]
        enc = tok(
            flat_docs,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].view(len(batch), doc_mask.size(1), -1)          # (B,M,Ltok)
        attention_mask = enc["attention_mask"].view(len(batch), doc_mask.size(1), -1)  # (B,M,Ltok)
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(len(batch), doc_mask.size(1), -1)
    else:
        raise ValueError(f"Unknown text_mode '{cfg.text_mode}'. Use 'simple' or 'hf'.")

    # 4) Build/pack pos_idx
    #    If a sample provided pos_idx, clip/pad; otherwise optionally sample.
    Lt_per_sample = _attn_len_from_patch(ts_len, cfg.patch_len)  # [B]
    Lt_max = int(Lt_per_sample.max().item())
    B, M = doc_mask.shape
    pos_idx = torch.full((B, Lt_max), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        p = b.get("pos_idx")
        if p is None or len(p) == 0:
            continue
        # Clip to available docs and to this sample's Lt
        Li = int(Lt_per_sample[i].item())
        trimmed = [min(int(v), M - 1) for v in p[:Li]]
        pos_idx[i, :len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)

    if cfg.sample_missing_pos:
        missing = (pos_idx[:, :Lt_max] < 0).any(dim=1)
        if missing.any():
            # sample for ALL (b,t) — will overwrite only the missing ones
            sampled = _sample_pos_indices(B=B, Lt=Lt_max, M=M, doc_mask=doc_mask, seed=cfg.seed, device=pos_idx.device)
            pos_idx[pos_idx < 0] = sampled[pos_idx < 0]

    # 5) Pack outputs
    out: Dict[str, Any] = {
        "ts": ts,                           # (B,L,d)
        "target": target,                   # (B,H,d)
        "ts_len": ts_len,                   # (B,)
        "attn_len": Lt_per_sample,          # (B,)
        "pos_idx": pos_idx,                 # (B,Lt_max)
        "doc_mask": doc_mask,               # (B,M)
    }

    # Text tensors
    out["input_ids"] = input_ids           # (B,M,Ltok)
    out["attention_mask"] = attention_mask # (B,M,Ltok)
    # For convenience/backwards-compat with SimpleTextEncoder pipeline:
    out["docs_tokens"] = input_ids
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids

    return out
