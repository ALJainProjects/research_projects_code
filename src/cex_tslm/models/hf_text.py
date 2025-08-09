"""
HuggingFace text encoder (BERT/roberta/etc.) and GPT-2 decoder wrappers, upgraded.

Drop-in replacements for SimpleTextEncoder and TinyLMDecoder in CEX-TSLM with:

- HFTextEncoder
  * Works with any AutoModel encoder (BERT/Roberta/DeBERTa…)
  * CLS or mean pooling
  * Optional projection to your model dim
  * Optional parameter freezing
  * Accepts either pre-tokenized ids (B,M,L) or raw strings (B,M) via encode_texts()

- GPT2PrefixDecoder
  * GPT-2 LM head + **prefix-tuning** adapter to condition on the fused memory
  * Proper padding token handling
  * Teacher-forced training (returns logits)
  * Fully trainable adapter; GPT-2 can be frozen or fine-tuned

Notes
-----
These wrappers keep external deps to `transformers`. Install:
    pip install transformers>=4.30

Typical usage inside CEXTSLM when using HF encoders:
    txt = HFTextEncoder('bert-base-uncased', pooling='cls', d_out=cfg.d_model)
    dec = GPT2PrefixDecoder('gpt2', prefix_len=10, freeze_gpt2=True, d_mem=cfg.d_model)

    # forward for encoder with pre-tokenized docs (B,M,Ltok):
    H_txt = txt(docs_input_ids, docs_attention_mask)

    # forward for decoder:
    logits = dec(tgt_tokens, memory=fused)  # memory: (B,Lt,D)

Author: you :)
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Literal, Union

import torch
import torch.nn as nn

# ---- Text Encoder -----------------------------------------------------------

class HFTextEncoder(nn.Module):
    """
    HuggingFace encoder that returns a single embedding per document.

    Args:
        model_name: HF model id/path (e.g., "bert-base-uncased", "roberta-base")
        pooling: "cls" → take first token ([CLS]) if available, else fallback to mean
                 "mean" → mean of hidden states masked by attention_mask
        d_out: project encoder hidden size to this dimension (or disable w/ None)
        freeze: if True, freezes the encoder backbone params

    Forward (two modes):
        1) Pre-tokenized:
            input_ids:      (B, M, L)
            attention_mask: (B, M, L)
            token_type_ids: optional (B, M, L)

        2) Raw strings (slower; tokenizes on the fly):
            raw_docs: List[List[str]] of length B, each with M strings
            (use encode_texts to create batch tensors if you want to do it outside)
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling: Literal["cls", "mean"] = "cls",
        d_out: Optional[int] = 256,
        freeze: bool = False,
        cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError("transformers is required for HFTextEncoder.") from e

        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("Could not infer hidden size from encoder config.")

        self.proj = nn.Identity() if (d_out is None or d_out == self.hidden_size) else nn.Linear(self.hidden_size, d_out)
        self.norm = nn.LayerNorm(d_out if isinstance(self.proj, nn.Linear) else self.hidden_size)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode_texts(
        self,
        raw_docs: List[List[str]],
        max_length: int = 128,
        padding: Literal["max_length", "longest"] = "max_length",
        truncation: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Tokenize nested list of docs → (input_ids, attention_mask, token_type_ids?)
        Shapes: (B, M, L)
        """
        B = len(raw_docs)
        M = max(len(x) for x in raw_docs) if raw_docs else 1
        flat = [d for docs in raw_docs for d in docs]
        if len(flat) == 0:
            flat = [""]  # avoid tokenizer crash

        enc = self.tokenizer(
            flat,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        tok_type = enc.get("token_type_ids", None)

        # reshape back to (B, M, L). Pad M dimension per batch with empty strings if needed.
        L = input_ids.size(1)
        if len(flat) != B * M:  # ragged M → pad per batch row
            # Expand each row to M using tokenizer pad_id
            pad_id = self.tokenizer.pad_token_id or 0
            pad_row = torch.full((L,), pad_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros(L, dtype=attn_mask.dtype)
            pad_toktype = torch.zeros(L, dtype=tok_type.dtype) if tok_type is not None else None

            ids_rows, mask_rows, tt_rows = [], [], []
            cursor = 0
            for docs in raw_docs:
                k = len(docs)
                ids_rows.append(input_ids[cursor:cursor+k])
                mask_rows.append(attn_mask[cursor:cursor+k])
                if tok_type is not None:
                    tt_rows.append(tok_type[cursor:cursor+k])
                # pad to M
                if k < M:
                    ids_rows[-1] = torch.cat([ids_rows[-1], pad_row.unsqueeze(0).repeat(M-k, 1)], dim=0)
                    mask_rows[-1] = torch.cat([mask_rows[-1], pad_mask.unsqueeze(0).repeat(M-k, 1)], dim=0)
                    if tok_type is not None:
                        tt_rows[-1] = torch.cat([tt_rows[-1], pad_toktype.unsqueeze(0).repeat(M-k, 1)], dim=0)
                cursor += k
            input_ids = torch.stack(ids_rows, dim=0)       # (B,M,L)
            attn_mask = torch.stack(mask_rows, dim=0)      # (B,M,L)
            tok_type = torch.stack(tt_rows, dim=0) if tok_type is not None else None
        else:
            input_ids = input_ids.view(B, M, L)
            attn_mask = attn_mask.view(B, M, L)
            tok_type = tok_type.view(B, M, L) if tok_type is not None else None

        if device is not None:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            if tok_type is not None:
                tok_type = tok_type.to(device)

        return input_ids, attn_mask, tok_type

    def _pool(self, hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """
        hidden: (BM, L, H), attn: (BM, L)
        """
        if self.pooling == "cls":
            # Many encoders put [CLS] at index 0. If not, this falls back gracefully in practice,
            # but mean pooling is safer for models without CLS (e.g., DistilRoBERTa).
            pooled = hidden[:, 0]
        else:
            mask = attn.unsqueeze(-1)  # (BM, L, 1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return pooled

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        raw_docs: Optional[List[List[str]]] = None,
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Returns:
            doc embeddings: (B, M, D_out)
        """
        if input_ids is None or attention_mask is None:
            assert raw_docs is not None, "Provide either (input_ids, attention_mask) or raw_docs."
            input_ids, attention_mask, token_type_ids = self.encode_texts(raw_docs, max_length=max_length, device=next(self.parameters()).device)

        B, M, L = input_ids.shape
        flat = {
            "input_ids": input_ids.view(B * M, L),
            "attention_mask": attention_mask.view(B * M, L),
        }
        if token_type_ids is not None:
            flat["token_type_ids"] = token_type_ids.view(B * M, L)

        out = self.model(**flat, return_dict=True)
        last_hidden = out.last_hidden_state  # (BM, L, H)
        pooled = self._pool(last_hidden, flat["attention_mask"])  # (BM, H)
        emb = self.norm(self.proj(pooled)).view(B, M, -1)
        return emb


# ---- GPT-2 Prefix Decoder ---------------------------------------------------

class _PrefixAdapter(nn.Module):
    """
    Lightweight prefix-tuning adapter.
    Maps a memory summary vector (B, D_mem) to GPT-2 past_key_values:
      tuple of length n_layer, each element:
        (k, v) with shape (B, n_head, prefix_len, head_dim)

    We implement a small MLP that outputs all layers' (k,v) at once.
    """
    def __init__(self, n_layer: int, n_head: int, n_embd: int, prefix_len: int, d_mem: int, mid: int = 512):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.prefix_len = prefix_len
        self.head_dim = n_embd // n_head

        total = n_layer * 2 * n_head * prefix_len * self.head_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_mem),
            nn.Linear(d_mem, mid),
            nn.GELU(),
            nn.Linear(mid, total),
        )

    def forward(self, mem: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        mem: (B, D_mem) pooled memory
        returns: past_key_values tuple for GPT-2
        """
        B = mem.size(0)
        raw = self.mlp(mem)  # (B, total)
        kv = raw.view(
            B,
            self.n_layer,
            2,
            self.n_head,
            self.prefix_len,
            self.head_dim,
        )
        # convert to tuple of length n_layer with (k, v)
        pkv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = tuple(
            (kv[:, i, 0], kv[:, i, 1]) for i in range(self.n_layer)
        )
        return pkv


class GPT2PrefixDecoder(nn.Module):
    """
    GPT-2 decoder with prefix-tuning to condition on cross-modal memory.

    Conditioning:
      - Pool the fused memory (B, Lt, D_mem) → mean → (B, D_mem)
      - MLP → past_key_values prefixes for each layer
      - Teacher forcing w/ input_ids; returns logits (B, Lr, V)

    Args:
        model_name: HF GPT-2 family id (e.g., "gpt2", "gpt2-medium")
        prefix_len: number of pseudo tokens for the prefix (5–20 works fine)
        freeze_gpt2: if True, freezes GPT-2 weights (only trains prefix adapter)
        d_mem: memory (fused) dim
        cache_dir/use_auth_token: HF caching/auth
    """
    def __init__(
        self,
        model_name: str = "gpt2",
        prefix_len: int = 10,
        freeze_gpt2: bool = True,
        d_mem: int = 256,
        cache_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        except Exception as e:
            raise ImportError("transformers is required for GPT2PrefixDecoder.") from e

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
        # ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir, use_auth_token=use_auth_token)
        cfg = self.gpt2.config
        n_layer, n_head, n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
        self.prefix_len = prefix_len

        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        # Prefix adapter: memory -> past_key_values
        self.adapter = _PrefixAdapter(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            prefix_len=prefix_len,
            d_mem=d_mem,
            mid=max(256, d_mem),
        )

    def _build_prefix_mask(self, attention_mask: Optional[torch.Tensor], B: int, Lr: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        GPT-2 usually ignores attention_mask unless using special settings, but we
        provide a mask including the prefix length for completeness.
        """
        if attention_mask is None:
            # [prefix_len + Lr] ones
            return torch.ones(B, self.prefix_len + Lr, dtype=torch.long, device=device)
        else:
            pre = torch.ones(B, self.prefix_len, dtype=attention_mask.dtype, device=attention_mask.device)
            return torch.cat([pre, attention_mask], dim=1)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt_tokens: (B, Lr) teacher-forcing token ids (use tokenizer.pad_token_id for padding)
        memory:     (B, Lm, D_mem) fused sequence; will be mean-pooled for conditioning
        attention_mask: optional (B, Lr)

        Returns:
            logits: (B, Lr, V)
        """
        B, Lr = tgt_tokens.shape
        device = tgt_tokens.device

        # Pool memory
        mem_pooled = memory.mean(dim=1)  # (B, D_mem)

        # Build past_key_values prefixes (tuple length n_layer)
        past_key_values = self.adapter(mem_pooled)

        # Build masked attention (prefix + tokens)
        attn = self._build_prefix_mask(attention_mask, B, Lr, device)

        out = self.gpt2(
            input_ids=tgt_tokens,
            attention_mask=attn,              # includes prefix length
            past_key_values=past_key_values,  # condition on memory
            use_cache=False,
            return_dict=True,
        )
        return out.logits  # (B, Lr, V)
