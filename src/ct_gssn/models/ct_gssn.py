# src/ct_gssn/models/ct_gssn.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _mlp(din, dhid, dout, act=nn.SiLU, dropout: float = 0.0):
    """
    Two-layer MLP used throughout the module.

    Args:
        din: input dim
        dhid: hidden dim
        dout: output dim
        act: activation module class
        dropout: dropout prob between layers
    """
    return nn.Sequential(
        nn.Linear(din, dhid),
        act(),
        nn.Dropout(dropout),
        nn.Linear(dhid, dout),
    )


# ---------------------------------------------------------------------
# GIM: Graph-based Inter-Series Modulator
# ---------------------------------------------------------------------
class GIM(nn.Module):
    """
    Graph-based Inter-Series Modulator (GIM)

    Message passing GNN that produces conditioning vectors c_i for each node i.

    Paper-ish notation:
      - messages m_{j->i} = MLP([h_i, h_j])
      - c_i = AGG_j m_{j->i}, with AGG in {"mean","sum","max"}

    Extensions (configurable):
      - edge dropout: randomly drop edges during training for regularization
      - learnable edge scaling via a light bilinear gate (optional)
    """
    def __init__(
        self,
        hidden_dim: int,
        msg_dim: int,
        c_dim: int,
        agg: str = "mean",
        dropout: float = 0.0,
        edge_dropout: float = 0.0,
        use_edge_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.msg = _mlp(din=2 * hidden_dim, dhid=msg_dim, dout=msg_dim, dropout=dropout)
        self.to_c = _mlp(din=msg_dim, dhid=msg_dim, dout=c_dim, dropout=dropout)
        assert agg in ("mean", "sum", "max")
        self.agg = agg
        self.edge_dropout = edge_dropout
        self.use_edge_gate = use_edge_gate
        if use_edge_gate:
            # light bilinear to modulate edges using (h_i, h_j)
            self.edge_gate = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (B, N, D) node hidden states
            adj: (B, N, N) adjacency (weighted or binary). Diagonal is ignored.

        Returns:
            c: (B, N, C) conditioning vectors
        """
        B, N, D = h.shape
        # Pairwise [i, j] -> concat(h_i, h_j)
        hi = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
        pair = torch.cat([hi, hj], dim=-1)      # (B, N, N, 2D)

        # Messages
        m = self.msg(pair)                       # (B, N, N, msg_dim)

        # Mask self-loops
        mask = (1 - torch.eye(N, device=h.device, dtype=h.dtype)).view(1, N, N, 1)
        m = m * mask

        # Optional learned edge gating
        if self.use_edge_gate:
            gate = self.edge_gate(hi, hj).sigmoid().unsqueeze(-1)  # (B,N,N,1)
        else:
            gate = 1.0

        # Edge dropout (train-time, keeps expectation via / (1-p))
        A = adj
        if self.training and self.edge_dropout > 0.0:
            keep = (torch.rand_like(A) >= self.edge_dropout).float()
            A = A * keep / (1.0 - self.edge_dropout)

        # Weight messages by adjacency * gate
        m = m * (A.unsqueeze(-1) * gate)         # (B, N, N, msg_dim)

        # Aggregate
        if self.agg == "mean":
            denom = A.sum(dim=-1, keepdim=True).clamp_min(1.0)     # (B,N,1)
            m_agg = m.sum(dim=2) / denom.unsqueeze(-1)             # (B,N,msg_dim)
        elif self.agg == "sum":
            m_agg = m.sum(dim=2)
        else:
            m_agg, _ = m.max(dim=2)

        c = self.to_c(m_agg)                     # (B,N,C)
        return c


# ---------------------------------------------------------------------
# CISP: Continuous-Time Intra-Series Processor
# ---------------------------------------------------------------------
class CISP(nn.Module):
    """
    Neural ODE dynamics parameterized by SSM matrices A_i(t), B_i(t) produced from c_i.

    Base dynamics per node i:
        d h_i / dt = A_i h_i + B_i u_i

    Exact closed-form step (when used):
        h(t+Δ) = exp(AΔ) h(t) + A^{-1} (exp(AΔ) - I) B u

    Extensions:
      - multiple parameterizations for A (free/skew-neg/lowrank/diag/diag+lowrank)
      - multiple solvers: "closed_form" (default), "euler", "rk4"
      - stability bias via subtracting γI
    """
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        c_dim: int,
        a_rank: Optional[int] = None,
        dropout: float = 0.0,
        param_A: str = "free",          # "free" | "skew_neg" | "lowrank_free" | "diag" | "diag_plus_lowrank"
        stability_eps: float = 0.01,    # γ in -γI for stability bias
        solver: str = "closed_form",    # "closed_form" | "euler" | "rk4"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.a_rank = a_rank
        self.param_A = param_A
        self.stability_eps = stability_eps
        self.solver = solver

        # Map conditioning c_i to A_i and B_i
        D = hidden_dim
        if param_A == "skew_neg":
            self.to_A_raw = _mlp(c_dim, max(D, 64), D * D, dropout=dropout)
        elif param_A == "lowrank_free":
            assert a_rank is not None and a_rank > 0, "lowrank_free requires a_rank>0"
            self.to_Au = _mlp(c_dim, max(D, 64), D * a_rank, dropout=dropout)
            self.to_Av = _mlp(c_dim, max(D, 64), D * a_rank, dropout=dropout)
        elif param_A == "diag":
            self.to_Ad = _mlp(c_dim, max(D, 64), D, dropout=dropout)
        elif param_A == "diag_plus_lowrank":
            assert a_rank is not None and a_rank > 0, "diag_plus_lowrank requires a_rank>0"
            self.to_Ad = _mlp(c_dim, max(D, 64), D, dropout=dropout)
            self.to_Au = _mlp(c_dim, max(D, 64), D * a_rank, dropout=dropout)
            self.to_Av = _mlp(c_dim, max(D, 64), D * a_rank, dropout=dropout)
        else:  # "free"
            self.to_A = _mlp(c_dim, max(D, 64), D * D, dropout=dropout)

        self.to_B = _mlp(c_dim, max(D, 64), D * input_dim, dropout=dropout)

        # Optional input projection (allows masking in input space)
        self.u_proj = nn.Linear(input_dim, input_dim)

    def _make_A_B(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        c: (B, N, C)
        returns:
          A: (B, N, D, D)
          B: (B, N, D, P)
        """
        B, N, _ = c.shape
        D, P = self.hidden_dim, self.input_dim
        eye = torch.eye(D, device=c.device, dtype=c.dtype).view(1, 1, D, D)

        if self.param_A == "skew_neg":
            M = self.to_A_raw(c).view(B, N, D, D)
            skew = 0.5 * (M - M.transpose(-1, -2))
            A = skew - self.stability_eps * eye
        elif self.param_A == "lowrank_free":
            r = int(self.a_rank)
            U = self.to_Au(c).view(B, N, D, r)
            V = self.to_Av(c).view(B, N, D, r)
            A = torch.matmul(U, V.transpose(-1, -2)) - self.stability_eps * eye
        elif self.param_A == "diag":
            d = self.to_Ad(c).view(B, N, D)
            A = torch.diag_embed(d) - self.stability_eps * eye
        elif self.param_A == "diag_plus_lowrank":
            r = int(self.a_rank)
            d = self.to_Ad(c).view(B, N, D)
            U = self.to_Au(c).view(B, N, D, r)
            V = self.to_Av(c).view(B, N, D, r)
            A = torch.diag_embed(d) + torch.matmul(U, V.transpose(-1, -2)) - self.stability_eps * eye
        else:
            A = self.to_A(c).view(B, N, D, D) - self.stability_eps * eye

        Bmat = self.to_B(c).view(B, N, D, P)
        return A, Bmat

    @torch.no_grad()
    def _expA(self, A: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Compute exp(A * dt) using torch.linalg.matrix_exp

        A:  (B,N,D,D)
        dt: (B,1,1) or (B,N,1,1)  (we broadcast (B,1,1) to all nodes)
        returns: (B,N,D,D)
        """
        BN, D, _ = A.shape[0] * A.shape[1], A.shape[2], A.shape[3]
        A_ = (A * dt).reshape(BN, D, D)
        expA = torch.linalg.matrix_exp(A_)
        expA = expA.view(A.shape)
        return expA

    def _step_closed_form(self, h, u, A, Bmat, dt, ridge=1e-4):
        """
        Closed-form update using matrix exponential and linear solve.
        """
        Bsz, N, D = h.shape
        I = torch.eye(D, device=h.device, dtype=h.dtype).view(1, 1, D, D)
        expA = self._expA(A, dt.view(Bsz, 1, 1).to(h.dtype))  # (B,N,D,D)
        rhs = torch.matmul(expA - I, Bmat)                    # (B,N,D,P)
        rhs = torch.matmul(rhs, self.u_proj(u).unsqueeze(-1)).squeeze(-1)  # (B,N,D)

        A_ridge = A + ridge * I
        try:
            x = torch.linalg.solve(A_ridge.view(Bsz * N, D, D), rhs.view(Bsz * N, D, 1)).view(Bsz, N, D)
        except RuntimeError:
            x = torch.linalg.lstsq(A_ridge.view(Bsz * N, D, D), rhs.view(Bsz * N, D, 1)).solution.view(Bsz, N, D)

        h_next = torch.matmul(expA, h.unsqueeze(-1)).squeeze(-1) + x
        return h_next, expA

    def _f(self, h, u, A, Bmat):
        """Right-hand side f(h,u) = A h + B u ."""
        Bu = torch.matmul(Bmat, self.u_proj(u).unsqueeze(-1)).squeeze(-1)
        return torch.matmul(A, h.unsqueeze(-1)).squeeze(-1) + Bu

    def _step_euler(self, h, u, A, Bmat, dt):
        f = self._f(h, u, A, Bmat)
        return h + dt.view(-1, 1, 1) * f

    def _step_rk4(self, h, u, A, Bmat, dt):
        dt_ = dt.view(-1, 1, 1)
        k1 = self._f(h, u, A, Bmat)
        k2 = self._f(h + 0.5 * dt_ * k1, u, A, Bmat)
        k3 = self._f(h + 0.5 * dt_ * k2, u, A, Bmat)
        k4 = self._f(h + dt_ * k3, u, A, Bmat)
        return h + (dt_ / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        ridge: float = 1e-4,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        One irregular-time step for all nodes.

        Args:
            h:  (B,N,D)
            u:  (B,N,P)   observed impulse at this time (0 if not observed)
            c:  (B,N,C)   conditioning from GIM
            dt: (B,)      time increment for each batch item (shared across nodes)

        Returns:
            h_next: (B,N,D)
            aux: dict with 'A','B','expA' (when closed_form) and 'c'
        """
        A, Bmat = self._make_A_B(c)  # (B,N,D,D), (B,N,D,P)

        if self.solver == "closed_form":
            h_next, expA = self._step_closed_form(h, u, A, Bmat, dt, ridge=ridge)
            aux = {"A": A, "B": Bmat, "expA": expA, "c": c}
        elif self.solver == "euler":
            h_next = self._step_euler(h, u, A, Bmat, dt)
            aux = {"A": A, "B": Bmat, "c": c}
        elif self.solver == "rk4":
            h_next = self._step_rk4(h, u, A, Bmat, dt)
            aux = {"A": A, "B": Bmat, "c": c}
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return h_next, aux


# ---------------------------------------------------------------------
# One CT-GSSN layer: GIM -> CISP
# ---------------------------------------------------------------------
class CTGSSNLayer(nn.Module):
    """
    One CT-GSSN layer (Figure 1): GIM produces c_i which modulates CISP dynamics.
    """
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        c_dim: int,
        msg_dim: int,
        agg: str = "mean",
        a_rank: Optional[int] = None,
        dropout: float = 0.0,
        param_A: str = "free",
        stability_eps: float = 0.01,
        solver: str = "closed_form",
        edge_dropout: float = 0.0,
        use_edge_gate: bool = False,
        residual_gate: bool = False,
    ):
        super().__init__()
        self.gim = GIM(
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            c_dim=c_dim,
            agg=agg,
            dropout=dropout,
            edge_dropout=edge_dropout,
            use_edge_gate=use_edge_gate,
        )
        self.cisp = CISP(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            c_dim=c_dim,
            a_rank=a_rank,
            dropout=dropout,
            param_A=param_A,
            stability_eps=stability_eps,
            solver=solver,
        )
        self.residual_gate = residual_gate
        if residual_gate:
            self.alpha = nn.Parameter(torch.tensor(0.0))  # learnable gate for residual

    def forward(self, h: torch.Tensor, u: torch.Tensor, adj: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c = self.gim(h, adj)             # (B,N,C)
        h_next, aux = self.cisp(h, u, c, dt)  # (B,N,D)
        if self.residual_gate:
            # h <- (1 - σ(α)) * h_next + σ(α) * h
            gate = torch.sigmoid(self.alpha)
            h_next = (1 - gate) * h_next + gate * h
        return h_next, aux


# ---------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------
class Readout(nn.Module):
    """
    Lightweight per-node readout. If you prefer a pooled readout,
    you can pool over nodes outside the module.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B,N,out_dim)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class CTGSSNConfig:
    # Core
    input_dim: int = 4
    hidden_dim: int = 128
    c_dim: int = 128
    msg_dim: int = 128
    out_dim: int = 1
    num_layers: int = 4
    dropout: float = 0.0
    a_rank: Optional[int] = None
    agg: str = "mean"

    # Dynamics / Stability
    param_A: str = "free"          # "free" | "skew_neg" | "lowrank_free" | "diag" | "diag_plus_lowrank"
    stability_eps: float = 0.01
    solver: str = "closed_form"    # "closed_form" | "euler" | "rk4"

    # GIM regularization
    edge_dropout: float = 0.0
    use_edge_gate: bool = False

    # Layer behavior
    residual_gate: bool = False

    # AUX collection for losses/analysis
    compute_aux: bool = True       # collect A/c for stability loss & link pred


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class CTGSSN(nn.Module):
    """
    Full CT-GSSN stack with step-wise irregular-time integration.

    Inputs for a batch:
      x:        (B, N, L, P)   raw observations (impulses); zeros where no obs
      mask:     (B, N, L)      1 if observed at t_k else 0
      deltas:   (B, L)         Δt_k between t_{k-1} and t_k (first Δt can be >0)
      adj:      (B, N, N) or (B, L, N, N) adjacency (static or per-step dynamic)
      h0:       (B, N, D)      initial hidden state (optional; else zeros)
      pad_mask: (B, L)         optional; 1 where valid timestep, 0 where padded (skip updates)
    """
    def __init__(self, cfg: CTGSSNConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.layers = nn.ModuleList([
            CTGSSNLayer(
                hidden_dim=cfg.hidden_dim,
                input_dim=cfg.input_dim,
                c_dim=cfg.c_dim,
                msg_dim=cfg.msg_dim,
                agg=cfg.agg,
                a_rank=cfg.a_rank,
                dropout=cfg.dropout,
                param_A=cfg.param_A,
                stability_eps=cfg.stability_eps,
                solver=cfg.solver,
                edge_dropout=cfg.edge_dropout,
                use_edge_gate=cfg.use_edge_gate,
                residual_gate=cfg.residual_gate,
            )
            for _ in range(cfg.num_layers)
        ])
        self.readout = Readout(cfg.hidden_dim, cfg.out_dim)

    def _select_adj_t(self, adj: torch.Tensor, t: int) -> torch.Tensor:
        """
        adj can be (B,N,N) or (B,L,N,N). Return (B,N,N) for time t.
        """
        if adj.dim() == 3:
            return adj
        elif adj.dim() == 4:
            return adj[:, t]
        else:
            raise ValueError(f"adj must be (B,N,N) or (B,L,N,N), got shape {tuple(adj.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        deltas: torch.Tensor,
        adj: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, L, P = x.shape
        D = self.cfg.hidden_dim

        if h0 is None:
            h = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        else:
            h = h0

        outs = []
        if self.cfg.compute_aux or return_aux:
            aux_all = {"A": [], "c": []}
        else:
            aux_all = None

        for t in range(L):
            # If padded step, skip dynamics and just repeat previous h/y for everyone in the batch
            if pad_mask is not None:
                valid_b = pad_mask[:, t] > 0  # (B,)
                if valid_b.any():
                    # process only valid batch rows; others carry previous state
                    dt = deltas[:, t]                          # (B,)
                    u_t = x[:, :, t, :] * mask[:, :, t].unsqueeze(-1)  # (B,N,P)
                    u_t = self.input_proj(u_t)
                    adj_t = self._select_adj_t(adj, t)         # (B,N,N)

                    # For invalid rows, zero dt and zero inputs so state remains unchanged
                    dt = dt * valid_b.float()
                    u_t = u_t * valid_b.view(B, 1, 1)
                    adj_t = adj_t  # adjacency can stay as is; dynamics is zeroed by dt/u_t

                else:
                    # everyone is padded, just emit previous output
                    y_hat = self.readout(h)          # (B,N,out_dim)
                    outs.append(y_hat)
                    if aux_all is not None:
                        # append placeholders to keep time alignment
                        L_layers = len(self.layers)
                        dummy_A = torch.zeros(L_layers, B, N, D, D, device=h.device, dtype=h.dtype)
                        dummy_c = torch.zeros(L_layers, B, N, self.cfg.c_dim, device=h.device, dtype=h.dtype)
                        aux_all["A"].append(dummy_A)
                        aux_all["c"].append(dummy_c)
                    continue
            else:
                dt = deltas[:, t]
                u_t = x[:, :, t, :] * mask[:, :, t].unsqueeze(-1)
                u_t = self.input_proj(u_t)
                adj_t = self._select_adj_t(adj, t)

            # progressive multi-layer update at this step
            h_step = h
            if aux_all is not None:
                step_As = []
                step_cs = []

            for layer in self.layers:
                h_step, aux = layer(h_step, u_t, adj_t, dt)
                if aux_all is not None:
                    step_As.append(aux["A"])
                    step_cs.append(aux["c"])

            h = h_step
            y_hat = self.readout(h)          # (B,N,out_dim)
            outs.append(y_hat)

            if aux_all is not None:
                # (L_layers,B,N,D,D) and (L_layers,B,N,C)
                aux_all["A"].append(torch.stack(step_As, dim=0))
                aux_all["c"].append(torch.stack(step_cs, dim=0))

        y_preds = torch.stack(outs, dim=2)  # (B,N,L,out_dim)

        if return_aux and aux_all is not None:
            # Stack on time: (L_layers,B,L,N,*,*)
            aux_all["A"] = torch.stack(aux_all["A"], dim=2)
            aux_all["c"] = torch.stack(aux_all["c"], dim=2)
            return y_preds, aux_all
        return y_preds, {}


# ---------------------------------------------------------------------
# Losses & Aux Heads
# ---------------------------------------------------------------------
def lyapunov_stability_loss(A_seq: torch.Tensor) -> torch.Tensor:
    """
    Stability regularizer:
    Penalize positive eigenvalues of the symmetric part of A.

    Args:
        A_seq: (L_layers, B, L, N, D, D)

    Returns:
        scalar tensor
    """
    S = 0.5 * (A_seq + A_seq.transpose(-1, -2))  # (...,D,D)
    eigs = torch.linalg.eigvalsh(S)              # (...,D)
    penalty = F.relu(eigs).mean()
    return penalty


def masked_prediction_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error only over supervised positions.

    Args:
        y_pred: (B,N,L,1)
        y_true: (B,N,L,1)
        mask:   (B,N,L)  1 for supervised target timesteps/nodes
    """
    diff = (y_pred - y_true).squeeze(-1) ** 2  # (B,N,L)
    loss = (diff * mask).sum() / (mask.sum().clamp_min(1.0))
    return loss


class GraphLinkPredictor(nn.Module):
    """
    Simple link predictor over inferred node embeddings (conditioning c_i).
    We train a bilinear score on c_i to reconstruct masked edges.

    Usage:
        logits_pos = predictor(c, pos_edges)
        logits_neg = predictor(c, neg_edges)
    """
    def __init__(self, c_dim: int):
        super().__init__()
        self.score = nn.Bilinear(c_dim, c_dim, 1, bias=False)

    def forward(self, c: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        c:     (B,N,C)
        edges: (B, E, 2) edge list [i,j]

        returns:
          logits: (B,E)
        """
        B, N, C = c.shape
        i = edges[..., 0].long()
        j = edges[..., 1].long()
        ci = torch.gather(c, 1, i.unsqueeze(-1).expand(-1, -1, C))
        cj = torch.gather(c, 1, j.unsqueeze(-1).expand(-1, -1, C))
        logits = self.score(ci, cj).squeeze(-1)
        return logits
