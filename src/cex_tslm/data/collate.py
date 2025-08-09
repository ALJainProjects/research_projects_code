"""
Alternate collate helpers for CEX-TSLM with (or without) Hugging Face tokenizers.

These collators:
- Pad variable-length time series and targets to the max length in the batch.
- Tokenize *lists of documents* per sample with an HF tokenizer (BERT/roberta/etc.).
- Pad the number of docs per sample to the batch max (with a doc-level mask).
- Return attention masks for text, and optional token_type_ids if the tokenizer provides them.
- Carry through optional `pos_idx` fields (used for causal contrast) with padding.

Expected input sample schema (per item in the batch list):
{
    "ts":      torch.FloatTensor [L, d_x]        # required
    "target":  torch.FloatTensor [H, d_x]        # required
    "docs":    List[str]                         # required but can be empty
    "pos_idx": Optional[List[int]]               # optional (length ~ L/patch); can be missing
}

Returns a dict with:
{
    "ts":            FloatTensor [B, Lmax, d_x]
    "ts_len":        LongTensor  [B]                     # original lengths
    "target":        FloatTensor [B, Hmax, d_x]
    "target_len":    LongTensor  [B]
    "docs_tokens":   LongTensor  [B, Mmax, Ttok]
    "docs_attn":     LongTensor  [B, Mmax, Ttok]
    "docs_type_ids": LongTensor  [B, Mmax, Ttok] (maybe; only if tokenizer provides)
    "doc_mask":      BoolTensor  [B, Mmax]              # True for real docs, False for padded docs
    "pos_idx":       LongTensor  [B, Pmax] (if any pos_idx present; padded with -1)
    "pos_mask":      BoolTensor  [B, Pmax] (True for valid pos entries)
}
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch


def _pad_stack_2d(seqs: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad a list of 2D tensors [L_i, D] to [B, L_max, D].
    """
    assert len(seqs) > 0, "Empty batch."
    B = len(seqs)
    L_max = max(s.size(0) for s in seqs)
    D = seqs[0].size(1)
    out = seqs[0].new_full((B, L_max, D), fill_value=pad_value)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def _pad_stack_1d_int(seqs: List[torch.Tensor], pad_value: int = -1) -> torch.Tensor:
    """
    Pad a list of 1D Long tensors [Li] to [B, L_max] with pad_value (default -1).
    """
    assert len(seqs) > 0, "Empty batch."
    B = len(seqs)
    L_max = max(s.size(0) for s in seqs)
    out = torch.full((B, L_max), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_with_hf_tokenizer(
    batch: List[Dict[str, Any]],
    hf_tokenizer,
    max_tokens: int = 64,
    pad_docs_to_batch_max: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of CEX-TSLM samples using a HuggingFace tokenizer for document lists.

    Args:
        batch: list of samples (see schema above).
        hf_tokenizer: a HuggingFace tokenizer (e.g., AutoTokenizer.from_pretrained("bert-base-uncased"))
        max_tokens: per-document max token length for padding/truncation.
        pad_docs_to_batch_max: if True, pad the number of docs per item to the batch max M_max.
                               if False, pad to the per-sample length only (yields ragged list -> not supported).
    Returns:
        Dict[str, torch.Tensor] (see docstring at top of file)
    """
    # ---- Time-series + targets ----
    ts_list = []
    ts_len = []
    tgt_list = []
    tgt_len = []
    docs_per_item = []

    for b in batch:
        ts = b["ts"]
        tgt = b["target"]
        # Ensure tensors and float dtype
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts, dtype=torch.float32)
        if not torch.is_tensor(tgt):
            tgt = torch.tensor(tgt, dtype=torch.float32)
        ts_list.append(ts.float())
        tgt_list.append(tgt.float())
        ts_len.append(ts.size(0))
        tgt_len.append(tgt.size(0))

        # docs: list[str] (possibly empty)
        docs = b.get("docs", [])
        if not isinstance(docs, list):
            raise ValueError("Field 'docs' must be a List[str].")
        docs_per_item.append(docs)

    ts = _pad_stack_2d(ts_list, pad_value=0.0)          # [B, Lmax, d_x]
    target = _pad_stack_2d(tgt_list, pad_value=0.0)     # [B, Hmax, d_x]
    ts_len = torch.tensor(ts_len, dtype=torch.long)     # [B]
    target_len = torch.tensor(tgt_len, dtype=torch.long)

    # ---- Document tokenization ----
    B = len(batch)
    # Max docs per item in this batch
    M_max = max(len(d) for d in docs_per_item)
    M_max = max(M_max, 1)  # avoid zero (keep 1 "pad" doc when none present)

    # Tokenize each sample's docs (individually to keep per-sample doc counts)
    token_ids_list = []
    attn_mask_list = []
    type_ids_list: Optional[List[torch.Tensor]] = []

    for docs in docs_per_item:
        if len(docs) == 0:
            docs = [""]  # stub out a single empty doc so shapes are valid

        enc = hf_tokenizer(
            docs,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        # enc["input_ids"]: [M_i, Ttok]
        token_ids = enc["input_ids"]                       # LongTensor
        attn_mask = enc.get("attention_mask", None)        # LongTensor or None
        type_ids = enc.get("token_type_ids", None)         # LongTensor or None

        # If this sample has fewer docs than batch max, pad docs dimension
        if pad_docs_to_batch_max and token_ids.size(0) < M_max:
            pad_M = M_max - token_ids.size(0)
            pad_tok = torch.zeros(pad_M, token_ids.size(1), dtype=token_ids.dtype)
            token_ids = torch.cat([token_ids, pad_tok], dim=0)

            if attn_mask is not None:
                pad_attn = torch.zeros(pad_M, attn_mask.size(1), dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, pad_attn], dim=0)

            if type_ids is not None:
                pad_type = torch.zeros(pad_M, type_ids.size(1), dtype=type_ids.dtype)
                type_ids = torch.cat([type_ids, pad_type], dim=0)

        token_ids_list.append(token_ids)                   # [M_max, Ttok]
        attn_mask_list.append(attn_mask if attn_mask is not None
                              else torch.ones_like(token_ids, dtype=torch.long))
        if type_ids is not None:
            if type_ids_list == []:
                type_ids_list = []
            type_ids_list.append(type_ids)

    # Stack across batch -> [B, M_max, Ttok]
    docs_tokens = torch.stack(token_ids_list, dim=0)
    docs_attn = torch.stack(attn_mask_list, dim=0)
    # Doc presence mask: True where the doc existed pre-padding
    doc_mask = torch.zeros(B, M_max, dtype=torch.bool)
    for i, docs in enumerate(docs_per_item):
        m_i = max(len(docs), 1)
        doc_mask[i, :m_i] = True

    out: Dict[str, Any] = {
        "ts": ts,
        "ts_len": ts_len,
        "target": target,
        "target_len": target_len,
        "docs_tokens": docs_tokens.long(),
        "docs_attn": docs_attn.long(),
        "doc_mask": doc_mask,
    }

    if isinstance(type_ids_list, list) and len(type_ids_list) == B:
        out["docs_type_ids"] = torch.stack(type_ids_list, dim=0).long()

    # ---- Optional pos_idx passthrough ----
    pos_lists = []
    any_pos = False
    for b in batch:
        pos = b.get("pos_idx", None)
        if pos is None:
            pos_lists.append(torch.empty(0, dtype=torch.long))
        else:
            any_pos = True
            pos_lists.append(torch.tensor(pos, dtype=torch.long))

    if any_pos:
        pos_padded = _pad_stack_1d_int(pos_lists, pad_value=-1)  # [B, Pmax] with -1 padded
        pos_mask = pos_padded.ne(-1)
        out["pos_idx"] = pos_padded
        out["pos_mask"] = pos_mask

    return out


def collate_pre_tokenized(
    batch: List[Dict[str, Any]],
    pad_docs_to_batch_max: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate when samples ALREADY contain pre-tokenized docs under:
      - "docs_tokens": LongTensor [M_i, Ttok]
      - "docs_attn":   LongTensor [M_i, Ttok] (optional; default all-ones)
      - "docs_type_ids": LongTensor [M_i, Ttok] (optional)

    This mirrors `collate_with_hf_tokenizer` but skips tokenization.
    """
    ts_list = []
    ts_len = []
    tgt_list = []
    tgt_len = []
    docs_tokens_list = []
    docs_attn_list = []
    type_ids_list: Optional[List[torch.Tensor]] = []

    for b in batch:
        ts = b["ts"] if torch.is_tensor(b["ts"]) else torch.tensor(b["ts"], dtype=torch.float32)
        tgt = b["target"] if torch.is_tensor(b["target"]) else torch.tensor(b["target"], dtype=torch.float32)
        ts_list.append(ts.float())
        tgt_list.append(tgt.float())
        ts_len.append(ts.size(0))
        tgt_len.append(tgt.size(0))

        tok = b["docs_tokens"].long()
        attn = b.get("docs_attn", torch.ones_like(tok, dtype=torch.long)).long()
        docs_tokens_list.append(tok)
        docs_attn_list.append(attn)

        type_ids = b.get("docs_type_ids", None)
        if type_ids is not None:
            if isinstance(type_ids_list, list):
                type_ids_list.append(type_ids.long())

    ts = _pad_stack_2d(ts_list, pad_value=0.0)
    target = _pad_stack_2d(tgt_list, pad_value=0.0)
    ts_len = torch.tensor(ts_len, dtype=torch.long)
    target_len = torch.tensor(tgt_len, dtype=torch.long)

    B = len(batch)
    M_max = max(x.size(0) for x in docs_tokens_list)
    Ttok = docs_tokens_list[0].size(1)

    # pad docs dimension to M_max
    def _pad_docs(x_list: List[torch.Tensor], fill: int = 0):
        out = torch.full((B, M_max, Ttok), fill_value=fill, dtype=x_list[0].dtype)
        for i, x in enumerate(x_list):
            Mi = x.size(0)
            out[i, :Mi] = x
        return out

    docs_tokens = _pad_docs(docs_tokens_list, fill=0)
    docs_attn = _pad_docs(docs_attn_list, fill=0)

    doc_mask = torch.zeros(B, M_max, dtype=torch.bool)
    for i, x in enumerate(docs_tokens_list):
        doc_mask[i, : x.size(0)] = True

    out: Dict[str, Any] = {
        "ts": ts,
        "ts_len": ts_len,
        "target": target,
        "target_len": target_len,
        "docs_tokens": docs_tokens,
        "docs_attn": docs_attn,
        "doc_mask": doc_mask,
    }

    if isinstance(type_ids_list, list) and len(type_ids_list) == B:
        docs_type_ids = _pad_docs(type_ids_list, fill=0)
        out["docs_type_ids"] = docs_type_ids

    # Optional pos_idx passthrough
    pos_lists = []
    any_pos = False
    for b in batch:
        pos = b.get("pos_idx", None)
        if pos is None:
            pos_lists.append(torch.empty(0, dtype=torch.long))
        else:
            any_pos = True
            pos_lists.append(torch.tensor(pos, dtype=torch.long))
    if any_pos:
        pos_padded = _pad_stack_1d_int(pos_lists, pad_value=-1)
        pos_mask = pos_padded.ne(-1)
        out["pos_idx"] = pos_padded
        out["pos_mask"] = pos_mask

    return out
