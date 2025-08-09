# src/ct_gssn/data/irregular_dataset.py
import math
from typing import Dict, Tuple, List, Optional

import torch
from torch.utils.data import Dataset


class SyntheticIMTSDataset(Dataset):
    """
    Synthetic Irregular Multivariate Time Series (IMTS) generator.

    Features vs. the basic version:
      - Graph topologies: 'erdos', 'ring', 'grid' (normalized adjacency).
      - Optional dynamic adjacency over time (edge dropout/refresh).
      - Stable latent linear dynamics with optional tanh nonlinearity.
      - Irregular sampling masks: Bernoulli + optional block missingness.
      - Flexible target modes: 'same' (y_t), 'next' (y_{t+1}), 'multi_h' (horizon H).
      - Optional variable sequence lengths per sample.
      - Returns per-sample metadata and normalization stats.

    Shapes (per sample):
      x:      (N, L, P)         impulses / observed inputs
      y:      (N, L, Q) or (N, L, Q) for 'same'/'next', or (N, L, H*Q) for 'multi_h'
      mask:   (N, L)            observation mask
      deltas: (L,)              irregular step sizes
      adj:    (N, N) or (L, N, N)  static or dynamic row-normalized graph

    Notes:
      - Defaults reproduce your original behavior.
      - If variable_len=True, each sample gets L_i ∈ [min_L, L]; use collate_imts_pad.
    """

    def __init__(
        self,
        num_samples: int = 256,
        N: int = 12,
        L: int = 48,
        input_dim: int = 4,
        out_dim: int = 1,
        obs_prob: float = 0.5,
        seed: int = 42,
        # Latent dynamics
        hidden_dim: int = 64,
        nonlinear: bool = False,
        u_std: float = 0.3,
        process_noise_std: float = 0.02,
        # Graph
        graph_type: str = "erdos",              # 'erdos' | 'ring' | 'grid'
        adj_p: float = 0.2,                      # Erdos p
        dynamic_adj: bool = False,
        dynamic_refresh_prob: float = 0.05,      # prob of refreshing edges per step
        # Irregular timing
        dt_min: float = 0.1,
        dt_max: float = 0.7,
        # Missingness
        block_miss_prob: float = 0.1,            # prob a node starts a missing block at a step
        block_miss_len: int = 4,                 # length of missing block
        # Targets
        target_mode: str = "same",               # 'same' | 'next' | 'multi_h'
        horizon: int = 1,                         # used when target_mode='multi_h'
        # Variable length sequences
        variable_len: bool = False,
        min_L: Optional[int] = None,
        # Return adjacency as dynamic even if static graph_type
        force_dynamic_adj: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.N = N
        self.L = L
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.obs_prob = obs_prob
        self.hidden_dim = hidden_dim
        self.nonlinear = nonlinear
        self.u_std = u_std
        self.process_noise_std = process_noise_std
        self.graph_type = graph_type
        self.adj_p = adj_p
        self.dynamic_adj = dynamic_adj
        self.dynamic_refresh_prob = dynamic_refresh_prob
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.block_miss_prob = block_miss_prob
        self.block_miss_len = max(1, int(block_miss_len))
        self.target_mode = target_mode
        self.horizon = max(1, int(horizon))
        self.variable_len = variable_len
        self.min_L = max(4, int(min_L)) if (variable_len and min_L is not None) else None
        self.force_dynamic_adj = force_dynamic_adj

        # RNG
        self.g = torch.Generator().manual_seed(seed)

        # Build base adjacency (row-normalized, zero diag)
        self.base_adj = self._make_adjacency(graph_type, N, self.g, p=adj_p)

        # Latent linear SSM parameters per node (stable)
        D = hidden_dim
        U = torch.randn(N, D, D, generator=self.g) * 0.1
        # Symmetric negative-ish for stability
        A = (U + U.transpose(-1, -2)) * 0.5 - 0.5 * torch.eye(D)
        self.A = A  # (N, D, D)
        self.B = torch.randn(N, D, input_dim, generator=self.g) * 0.1
        self.C = torch.randn(N, out_dim, D, generator=self.g) * 0.2
        self.h0 = torch.randn(N, D, generator=self.g) * 0.1

        # Precompute node names for metadata
        self.node_names = [f"node_{i}" for i in range(N)]

    # ---- graph helpers ----
    @staticmethod
    def _row_normalize(adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(-1, keepdim=True).clamp_min(1.0)
        return adj / deg

    def _make_adjacency(self, kind: str, N: int, g: torch.Generator, p: float = 0.2) -> torch.Tensor:
        if kind == "erdos":
            adj = (torch.rand((N, N), generator=g) < p).float()
            adj.fill_diagonal_(0.0)
            adj = ((adj + adj.t()) > 0).float()
        elif kind == "ring":
            adj = torch.zeros(N, N)
            for i in range(N):
                adj[i, (i - 1) % N] = 1.0
                adj[i, (i + 1) % N] = 1.0
        elif kind == "grid":
            # Make roughly square grid
            H = int(math.sqrt(N))
            W = (N + H - 1) // H
            adj = torch.zeros(N, N)
            for i in range(N):
                r, c = divmod(i, W)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    j = rr * W + cc
                    if 0 <= rr < H and 0 <= cc < W and j < N:
                        adj[i, j] = 1.0
        else:
            raise ValueError(f"Unknown graph_type '{kind}'")
        return self._row_normalize(adj)

    def _maybe_refresh_adj(self, adj: torch.Tensor) -> torch.Tensor:
        # stochastic edge refresh for dynamic graphs
        if torch.rand((), generator=self.g).item() < self.dynamic_refresh_prob:
            return self._make_adjacency(self.graph_type, self.N, self.g, p=self.adj_p)
        return adj

    # ---- dataset protocol ----
    def __len__(self):
        return self.num_samples

    def _build_mask(self, N: int, L: int) -> torch.Tensor:
        """
        Irregular observation mask:
          - Bernoulli base obs_prob
          - Plus occasional missing blocks for realism
        """
        mask = (torch.rand(N, L, generator=self.g) < self.obs_prob).float()
        if self.block_miss_prob > 0 and self.block_miss_len > 1:
            for n in range(N):
                t = 0
                while t < L:
                    if torch.rand((), generator=self.g) < self.block_miss_prob:
                        t_end = min(L, t + self.block_miss_len)
                        mask[n, t:t_end] = 0.0
                        t = t_end
                    t += 1
        return mask

    def _target_from_states(self, H_lat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project latent states to observation target y per step.
        Returns y with shape:
          - 'same' : (N, L, Q) using state at t
          - 'next' : (N, L, Q) uses state at t+1, last step copies previous
          - 'multi_h' : (N, L, H*Q) concatenated horizons
        """
        N, L, D = H_lat.shape
        Q = self.out_dim
        C = self.C  # (N, Q, D)

        def _proj(h_step):  # (N, D) -> (N, Q)
            return torch.matmul(C, h_step.unsqueeze(-1)).squeeze(-1)

        if self.target_mode == "same":
            y = torch.stack([_proj(H_lat[:, t]) for t in range(L)], dim=1)  # (N, L, Q)
            return y
        elif self.target_mode == "next":
            y = []
            for t in range(L):
                t_src = min(t + 1, L - 1)
                y.append(_proj(H_lat[:, t_src]))
            return torch.stack(y, dim=1)  # (N, L, Q)
        elif self.target_mode == "multi_h":
            H = self.horizon
            Ys = []
            for t in range(L):
                ys_t = []
                for k in range(H):
                    t_src = min(t + k, L - 1)
                    ys_t.append(_proj(H_lat[:, t_src]))  # (N, Q)
                Ys.append(torch.cat(ys_t, dim=-1))  # (N, H*Q)
            return torch.stack(Ys, dim=1)  # (N, L, H*Q)
        else:
            raise ValueError(f"Unknown target_mode '{self.target_mode}'")

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        N, Lmax, P = self.N, self.L, self.input_dim
        D, Q = self.hidden_dim, self.out_dim

        # Possibly sample a shorter sequence length per sample
        if self.variable_len:
            L = int(torch.randint(low=self.min_L or max(4, Lmax // 2), high=Lmax + 1, size=(1,), generator=self.g))
        else:
            L = Lmax

        # Global irregular time grid (monotone)
        deltas = (torch.rand(L, generator=self.g) * (self.dt_max - self.dt_min) + self.dt_min).float()

        # Inputs u and masks
        u = torch.randn(N, L, P, generator=self.g) * self.u_std
        mask = self._build_mask(N, L)

        # Start states
        h = self.h0.clone()
        H_lat = []
        A = self.A  # (N, D, D)
        B = self.B
        adj_t = self.base_adj.clone()

        # Rollout latent dynamics with optional dynamic graph and nonlinearity
        for t in range(L):
            if self.dynamic_adj:
                adj_t = self._maybe_refresh_adj(adj_t)
            expA = torch.linalg.matrix_exp(A * deltas[t])
            neigh = torch.matmul(adj_t, h)  # (N, D)

            # Linear step with control; ridge for stability of solve
            I = torch.eye(D)
            rhs = (expA - I) @ (B @ u[:, t]).squeeze(-1)  # (N, D)
            h_lin = (expA @ h.unsqueeze(-1)).squeeze(-1) + torch.linalg.solve(A + 1e-3 * I, rhs)

            if self.nonlinear:
                h = h_lin + 0.02 * neigh + 0.05 * torch.tanh(h_lin) + self.process_noise_std * torch.randn_like(h_lin, generator=self.g)
            else:
                h = h_lin + 0.02 * neigh + self.process_noise_std * torch.randn_like(h_lin, generator=self.g)

            H_lat.append(h)

        H_lat = torch.stack(H_lat, dim=1)  # (N, L, D)
        y = self._target_from_states(H_lat, u)  # (N, L, Q) or (N, L, H*Q)

        # Observations x are the impulses u (could also be (C @ h) + noise to mimic sensors)
        x = u  # (N, L, P)

        sample: Dict[str, torch.Tensor] = {
            "x": x.float(),                 # (N,L,P)
            "y": y.float(),                 # (N,L,Q) or (N,L,H*Q)
            "mask": mask.float(),           # (N,L)
            "deltas": deltas.float(),       # (L,)
            "adj": adj_t.float() if (self.dynamic_adj or self.force_dynamic_adj) else self.base_adj.float(),  # (N,N)
        }

        # Meta (kept torch/scalar-friendly)
        sample["meta"] = {
            "node_names": self.node_names,
            "graph_type": self.graph_type,
            "dynamic_adj": bool(self.dynamic_adj or self.force_dynamic_adj),
            "target_mode": self.target_mode,
        }

        # Simple normalization stats (could be used by a Normalizer)
        sample["stats"] = {
            "x_mean": x.mean().detach(),
            "x_std": x.std().clamp_min(1e-6).detach(),
            "y_mean": y.mean().detach(),
            "y_std": y.std().clamp_min(1e-6).detach(),
        }

        return sample


# -------------------------
# Collaters
# -------------------------

def collate_imts(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Backward-compatible collate for fixed-shape sequences.
    Use this when all samples have identical N and L (the default).
    """
    x = torch.stack([b["x"] for b in batch], dim=0)            # (B,N,L,P)
    y = torch.stack([b["y"] for b in batch], dim=0)            # (B,N,L,Q or H*Q)
    mask = torch.stack([b["mask"] for b in batch], dim=0)      # (B,N,L)
    deltas = torch.stack([b["deltas"] for b in batch], dim=0)  # (B,L)
    adj = torch.stack([b["adj"] for b in batch], dim=0)        # (B,N,N)
    out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj}
    # pass-through optional dicts if present (not used by CT-GSSN forward)
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    if "stats" in batch[0]:
        out["stats"] = [b["stats"] for b in batch]
    return out


def collate_imts_pad(batch: List[Dict[str, torch.Tensor]], pad_value: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    Ragged → padded collate for variable length sequences.
    Pads to the max L in the batch; produces an additional 'pad_mask' for time steps.

    Returns:
      x:        (B,N,Lmax,P)
      y:        (B,N,Lmax,Q or H*Q)
      mask:     (B,N,Lmax)              observation mask (padded with 0)
      deltas:   (B,Lmax)                (padded with 0)
      adj:      (B,N,N) or (B,Lmax,N,N) if you pass dynamic graphs yourself
      pad_mask: (B,Lmax)                1 where valid, 0 where padded
    """
    N = batch[0]["x"].shape[0]
    P = batch[0]["x"].shape[-1]
    Ls = [b["x"].shape[1] for b in batch]
    Lmax = max(Ls)
    y_dim_last = batch[0]["y"].shape[-1]

    def pad_tensor(t: torch.Tensor, target_len: int, dims=(1,)):
        # t shape (..., L, ...)
        pad_l = target_len - t.shape[dims[0]]
        if pad_l == 0:
            return t
        pad_spec = []
        # Build pad spec from last dim backwards (F.pad uses (last_dim_pad_left, last_dim_pad_right, ...))
        for _ in range(t.ndim):
            pad_spec.extend([0, 0])
        # Set along L-dimension
        # We pad on the right only
        pad_spec[2 * (t.ndim - dims[0] - 1) + 1] = pad_l
        return torch.nn.functional.pad(t, tuple(pad_spec))

    xs, ys, ms, ds, pads = [], [], [], [], []
    adjs = []

    for i, b in enumerate(batch):
        Li = b["x"].shape[1]
        xs.append(pad_tensor(b["x"], Lmax, dims=(1,)))
        ys.append(pad_tensor(b["y"], Lmax, dims=(1,)))
        ms.append(pad_tensor(b["mask"], Lmax, dims=(1,)))
        ds.append(pad_tensor(b["deltas"].unsqueeze(0), Lmax, dims=(1,)).squeeze(0))
        adjs.append(b["adj"])  # assume static per sample; if dynamic, pre-pack (L,N,N)
        pad = torch.zeros(Lmax)
        pad[:Li] = 1.0
        pads.append(pad)

    x = torch.stack(xs, dim=0)                       # (B,N,Lmax,P)
    y = torch.stack(ys, dim=0)                       # (B,N,Lmax,y_last)
    mask = torch.stack(ms, dim=0)                    # (B,N,Lmax)
    deltas = torch.stack(ds, dim=0)                  # (B,Lmax)
    adj = torch.stack(adjs, dim=0)                   # (B,N,N)  (for dynamic, adapt here)
    pad_mask = torch.stack(pads, dim=0)              # (B,Lmax)

    out = {"x": x, "y": y, "mask": mask, "deltas": deltas, "adj": adj, "pad_mask": pad_mask}
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    if "stats" in batch[0]:
        out["stats"] = [b["stats"] for b in batch]
    return out
