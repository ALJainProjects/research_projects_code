from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#                      Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (no params).
    Applied to a sequence (B, L, D). Adds position encodings up to max_len.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)


# ============================================================
#                        TS Encoder
# ============================================================

class SimpleTSEncoder(nn.Module):
    """
    Minimal patch-like time-series encoder:
      1) 1D conv patches (kernel=stride=patch_len) from (B, L, d_x) -> (B, L', D)
      2) TransformerEncoder over the patches
      3) Sinusoidal positional encoding

    Output: sequence of hidden states (B, L', D)
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        patch_len: int = 4,
        dropout: float = 0.1,
        use_sinusoid_pos: bool = True,
        max_len: int = 4096,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Conv1d(
            in_channels=in_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=patch_len,
            bias=True,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.use_sinusoid_pos = use_sinusoid_pos
        if use_sinusoid_pos:
            self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        else:
            # learned pos as fallback
            self.pos_table = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_x)
        returns H_ts: (B, L', D)
        """
        x = x.transpose(1, 2)          # (B, d_x, L)
        z = self.proj(x)               # (B, D, L')
        z = z.transpose(1, 2)          # (B, L', D)
        if self.use_sinusoid_pos:
            z = self.pos(z)
        else:
            z = z + self.pos_table[:, : z.size(1)]
        return self.encoder(z)


# ============================================================
#                        Text Encoders
# ============================================================

class SimpleTextEncoder(nn.Module):
    """
    Lightweight text encoder with no external dependencies.
    Embedding + biGRU → last hidden state concat → per-document embedding.

    Input:
      docs_tokens: (B, M, Ltok) int ids, 0 is PAD
    Output:
      H_txt: (B, M, D)
    """
    def __init__(self, vocab_size: int = 5000, d_model: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.gru = nn.GRU(
            d_model, d_model // 2, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, docs_tokens: torch.Tensor) -> torch.Tensor:
        B, M, Ltok = docs_tokens.shape
        x = self.emb(docs_tokens.view(B * M, Ltok))            # (B*M, Ltok, D)
        _, h = self.gru(x)                                     # (2*nl, B*M, D/2)
        h_last = torch.cat([h[-2], h[-1]], dim=-1)             # (B*M, D)
        return self.ln(h_last).view(B, M, -1)                  # (B, M, D)


class HFTextEncoder(nn.Module):
    """
    HuggingFace text encoder wrapper that turns (B, M, Ltok) into per-document embeddings.

    - For BERT-like models: CLS pooling (or mean pooling) from the last hidden state.
    - For GPT-2-like models (no CLS): mean pooling over non-pad tokens.

    Args:
        model_name_or_path: passed to transformers.AutoModel.from_pretrained
        pooling: "cls" | "mean"
        proj_to_d: optional projection to d_model (if HF hidden size != d_model)
        d_model: desired output dim when projecting
        freeze: if True, disables grad for HF encoder

    Forward expects:
        input_ids: (B, M, Ltok)
        attention_mask: (B, M, Ltok) with 1 for real tokens
        token_type_ids: optional (B, M, Ltok)

    Returns:
        H_txt: (B, M, d_out)
    """
    def __init__(
        self,
        model_name_or_path: str,
        pooling: Literal["cls", "mean"] = "cls",
        proj_to_d: bool = True,
        d_model: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoConfig
        except Exception as e:
            raise ImportError(
                "transformers not available. Install with `pip install transformers` to use HFTextEncoder."
            ) from e

        self.pooling = pooling
        self.core = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = getattr(self.core.config, "hidden_size", d_model)
        self.proj = nn.Linear(self.hidden_size, d_model) if proj_to_d and self.hidden_size != d_model else nn.Identity()
        self.ln = nn.LayerNorm(d_model)

        if freeze:
            for p in self.core.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, M, Ltok = input_ids.shape
        # flatten doc dimension for one HF call
        flat_inputs = {
            "input_ids": input_ids.view(B * M, Ltok),
            "attention_mask": attention_mask.view(B * M, Ltok),
        }
        if token_type_ids is not None:
            flat_inputs["token_type_ids"] = token_type_ids.view(B * M, Ltok)

        out = self.core(**flat_inputs, output_hidden_states=False, return_dict=True)
        last_hidden = out.last_hidden_state  # (B*M, Ltok, H)

        if self.pooling == "cls" and hasattr(self.core.config, "model_type") and "bert" in self.core.config.model_type:
            pooled = last_hidden[:, 0]  # (B*M, H) — [CLS]
        else:
            # mean pool over valid tokens
            mask = flat_inputs["attention_mask"].unsqueeze(-1)  # (B*M, Ltok, 1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom

        pooled = self.proj(pooled)            # (B*M, d_out)
        pooled = self.ln(pooled)
        return pooled.view(B, M, -1)          # (B, M, d_out)


# ============================================================
#                     Cross-Modal Fusion
# ============================================================

class CausalCrossAttention(nn.Module):
    """
    Cross-attention: Time-series queries attend to per-document text keys/values.
    Returns fused sequence and pre-softmax logits for causal contrastive loss.

    Inputs:
      H_ts:  (B, Lt, D)
      H_txt: (B, M, D)

    Outputs:
      fused:       (B, Lt, D)
      attn_logits: (B, H, Lt, M)   (H=heads)
    """
    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.view(B, L, self.nhead, self.d_head).transpose(1, 2)  # (B,H,L,dh)

    def forward(self, H_ts: torch.Tensor, H_txt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self._split(self.q_proj(H_ts))             # (B,H,Lt,dh)
        K = self._split(self.k_proj(H_txt))            # (B,H,M,dh)
        V = self._split(self.v_proj(H_txt))            # (B,H,M,dh)

        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_head ** 0.5)  # (B,H,Lt,M)
        attn = torch.softmax(attn_logits, dim=-1)                                   # (B,H,Lt,M)
        ctx = torch.matmul(attn, V)                                                 # (B,H,Lt,dh)
        ctx = ctx.transpose(1, 2).contiguous().view(H_ts.size(0), H_ts.size(1), -1) # (B,Lt,D)
        out = self.out(ctx)
        out = self.ln(self.dropout(out) + H_ts)  # residual + norm
        return out, attn_logits


# ============================================================
#                     Explanation Decoder
# ============================================================

class TinyLMDecoder(nn.Module):
    """
    Tiny autoregressive decoder for rationale generation.
    Teacher-forced training via TransformerDecoder.
    """
    def __init__(self, vocab_size: int = 5000, d_model: int = 256, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt_tokens: (B, Lr) int ids
        memory:     (B, Lm, D) fused representation from cross-modal encoder
        """
        tgt = self.emb(tgt_tokens)
        Lr = tgt.size(1)
        causal_mask = torch.triu(torch.ones(Lr, Lr, device=tgt.device, dtype=torch.bool), diagonal=1)
        out = self.dec(tgt, memory, tgt_mask=causal_mask)
        return self.lm_head(out)  # (B, Lr, V)


class HFDecoderCausalLM(nn.Module):
    """
    Optional HuggingFace causal LM decoder (e.g., GPT-2) conditioned on fused memory
    via cross-attention **keys/values** injection (project memory to KV and append).
    This is a lightweight approximation if you want to use a real LM head.

    NOTE: To keep dependencies minimal, this module concatenates a single BOS token
    to the target and feeds the model in standard causal fashion. For tight coupling
    (true cross-attn), you'd need a model that supports encoder-decoder or adapters.

    Usage:
      - Pass `decoder_inputs` (input_ids/attention_mask) already prepared.
      - This class then only runs the HF model and returns logits.

    If you don't need HF decode, stick with TinyLMDecoder.
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:
            raise ImportError("transformers not available. Install with `pip install transformers`.") from e
        self.core = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.core(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.logits  # (B, Lr, V)


# ============================================================
#                        Model Config
# ============================================================

@dataclass
class CEXConfig:
    # TS encoder
    ts_in_dim: int = 4
    d_model: int = 256
    ts_patch_len: int = 4
    ts_layers: int = 2
    ts_heads: int = 4
    ts_dropout: float = 0.1

    # Text encoder
    text_backbone: Literal["simple", "hf"] = "simple"
    txt_vocab: int = 5000           # only used for 'simple'
    txt_layers: int = 1
    txt_dropout: float = 0.1
    hf_text_model: str = "bert-base-uncased"
    hf_text_pooling: Literal["cls", "mean"] = "cls"
    hf_text_freeze: bool = False

    # Cross-modal fusion
    fusion_heads: int = 4
    fusion_dropout: float = 0.1

    # Forecasting
    forecast_h: int = 8
    forecast_mlp_depth: int = 2

    # Explanation decoder
    decoder_kind: Literal["tiny", "none"] = "tiny"
    dec_layers: int = 2
    dec_heads: int = 4
    dec_dropout: float = 0.1


# ============================================================
#                         Full Model
# ============================================================

class CEXTSLM(nn.Module):
    """
    CEX-TSLM end-to-end:
      - TS encoder (patch + Transformer)
      - Text encoder (Simple biGRU OR HuggingFace model)
      - Causal cross-modal attention (exposes logits for contrastive supervision)
      - Forecast head (multi-step)
      - Optional explanation decoder (TinyLM)

    Forward paths supported:
      • Simple text:
          forward(ts, docs_tokens=Long[B,M,Ltok], tgt_tokens=Long[B,Lr]?)
      • HF text:
          forward(ts, input_ids=Long[B,M,Ltok], attention_mask=Long[B,M,Ltok], token_type_ids=?)

    Returns dict:
      {
        "H_ts": (B, Lt, D),
        "H_txt": (B, M, D),
        "fused": (B, Lt, D),
        "attn_logits": (B, H, Lt, M),
        "forecast": (B, H_forecast, ts_in_dim),
        "logits_explain": (B, Lr, V)  # if decoder active and tgt provided
      }
    """
    def __init__(self, cfg: CEXConfig):
        super().__init__()
        self.cfg = cfg

        # TS encoder
        self.ts_enc = SimpleTSEncoder(
            in_dim=cfg.ts_in_dim,
            d_model=cfg.d_model,
            nhead=cfg.ts_heads,
            num_layers=cfg.ts_layers,
            patch_len=cfg.ts_patch_len,
            dropout=cfg.ts_dropout,
        )

        # Text encoder (simple or HF)
        if cfg.text_backbone == "simple":
            self.txt_enc = SimpleTextEncoder(
                vocab_size=cfg.txt_vocab, d_model=cfg.d_model, num_layers=cfg.txt_layers, dropout=cfg.txt_dropout
            )
            self.uses_hf = False
        else:
            self.txt_enc = HFTextEncoder(
                model_name_or_path=cfg.hf_text_model,
                pooling=cfg.hf_text_pooling,
                proj_to_d=True,
                d_model=cfg.d_model,
                freeze=cfg.hf_text_freeze,
            )
            self.uses_hf = True

        # Cross-modal fusion
        self.causal_fusion = CausalCrossAttention(d_model=cfg.d_model, nhead=cfg.fusion_heads, dropout=cfg.fusion_dropout)

        # Forecast head: pooled fused → MLP → H*ts_in_dim
        layers: List[nn.Module] = [nn.LayerNorm(cfg.d_model)]
        for _ in range(max(0, cfg.forecast_mlp_depth - 1)):
            layers += [nn.Linear(cfg.d_model, cfg.d_model), nn.GELU()]
        layers += [nn.Linear(cfg.d_model, cfg.forecast_h * cfg.ts_in_dim)]
        self.forecast_head = nn.Sequential(*layers)

        # Optional decoder
        if cfg.decoder_kind == "tiny":
            # Tiny LM over a vocab (only for "simple" path by default)
            self.decoder = TinyLMDecoder(
                vocab_size=(cfg.txt_vocab if cfg.text_backbone == "simple" else 50000),
                d_model=cfg.d_model,
                nhead=cfg.dec_heads,
                num_layers=cfg.dec_layers,
                dropout=cfg.dec_dropout,
            )
        else:
            self.decoder = None

    # --------------- helper --------------- #

    def _run_text_encoder(self, **kwargs) -> torch.Tensor:
        """
        Dispatch to correct text encoder variant.
        Simple: expects docs_tokens=(B,M,Ltok)
        HF:     expects input_ids, attention_mask, token_type_ids? (all (B,M,Ltok))
        """
        if not self.uses_hf:
            docs_tokens = kwargs.get("docs_tokens", None)
            assert docs_tokens is not None, "docs_tokens is required for SimpleTextEncoder."
            return self.txt_enc(docs_tokens)  # (B, M, D)
        else:
            input_ids = kwargs.get("input_ids", None)
            attention_mask = kwargs.get("attention_mask", None)
            token_type_ids = kwargs.get("token_type_ids", None)
            assert input_ids is not None and attention_mask is not None, "HFTextEncoder requires input_ids & attention_mask."
            return self.txt_enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # --------------- forward --------------- #

    def forward(
        self,
        ts: torch.Tensor,
        *,
        # Simple text path
        docs_tokens: Optional[torch.Tensor] = None,
        # HF text path
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        # Explanation target (teacher forcing for TinyLMDecoder)
        tgt_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Time-series + text → fused reps + forecast (+ optional explanation logits).
        """
        # Encode modalities
        H_ts = self.ts_enc(ts)                           # (B, Lt, D)
        H_txt = self._run_text_encoder(
            docs_tokens=docs_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )                                                # (B, M, D)

        # Cross-modal fusion
        fused, attn_logits = self.causal_fusion(H_ts, H_txt)  # (B,Lt,D), (B,H,Lt,M)

        # Forecast from pooled fused rep
        pooled = fused.mean(dim=1)                       # (B,D)
        forecast_flat = self.forecast_head(pooled)       # (B, H*dx)
        forecast = forecast_flat.view(ts.size(0), self.cfg.forecast_h, -1)  # (B,H,ts_in_dim)

        out = {
            "H_ts": H_ts,
            "H_txt": H_txt,
            "fused": fused,
            "attn_logits": attn_logits,
            "forecast": forecast,
        }

        # Optional explanation decoder (TinyLMDecoder)
        if self.decoder is not None and tgt_tokens is not None:
            logits = self.decoder(tgt_tokens, fused)     # (B,Lr,V)
            out["logits_explain"] = logits

        return out


# ============================================================
#                           Losses
# ============================================================

def forecast_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared error over the forecast horizon and feature dimension.
    pred, target: (B, H, d)
    """
    return F.mse_loss(pred, target)


def explanation_loss(logits: torch.Tensor, target_tokens: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:
    """
    Standard LM cross-entropy with padding ignored.
    logits:        (B, Lr, V)
    target_tokens: (B, Lr)
    """
    B, Lr, V = logits.shape
    return F.cross_entropy(logits.view(B * Lr, V), target_tokens.view(B * Lr), ignore_index=ignore_index)


def causal_contrastive_loss(
    attn_logits: torch.Tensor,
    pos_idx: torch.Tensor,
    tau: float = 0.07,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    InfoNCE-style loss over documents per time query.

    attn_logits: (B, H, Lt, M)  pre-softmax similarities from cross-attn
    pos_idx:     (B, Lt)        index of positive doc per time position; use -1 for 'no label'
    tau:         temperature
    mask:        optional (B, Lt) bool mask where True => include position, False => skip

    Implementation details:
      - Average heads first → sim (B, Lt, M)
      - For positions with pos_idx == -1, we drop them from the loss.
    """
    B, H, Lt, M = attn_logits.shape
    sim = attn_logits.mean(dim=1) / tau             # (B, Lt, M)

    # valid positions
    valid = (pos_idx >= 0)
    if mask is not None:
        valid = valid & mask.bool()
    if valid.sum() == 0:
        # no supervision available; return zero (no grad)
        return sim.sum() * 0.0

    # Gather positives
    pos_flat = pos_idx.clamp_min(0).unsqueeze(-1)   # (B, Lt, 1)
    pos_scores = sim.gather(-1, pos_flat).squeeze(-1)  # (B, Lt)

    # Log-sum-exp over docs
    denom = torch.logsumexp(sim, dim=-1)            # (B, Lt)

    loss = (-(pos_scores) + denom)[valid]
    return loss.mean()
