"""Per-algorithm state encoder.

Maps a (batched) ``CanonicalState`` into a structured latent shared across all
algorithms:

    z = (dim_tokens [B, D, H], global_vec [B, H])

Pipeline: assemble per-(particle, dim) input channels from the normalized state
-> per-cell MLP -> DeepSets pool over particles -> dimension-tokens -> stacked
covariance-biased self-attention over the D tokens -> read out a global vector.
Because each encoder consumes a different set of fields, encoders are
per-algorithm; but every encoder emits the *same* latent structure, so any
encoder can be paired with any decoder head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cat.models.layers import MLP, DimAttention, ParticlePool
from cat.models.norm import POSITION_LIKE, NormContext
from cat.state.canonical import CanonicalState
from cat.state.schema import Struct, get_spec


@dataclass
class Latent:
    dim_tokens: Tensor  # (B, D, H)
    global_vec: Tensor  # (B, H)


class StateEncoder(nn.Module):
    def __init__(
        self, algo: str, hidden: int = 64, n_layers: int = 2, n_heads: int = 4
    ):
        super().__init__()
        self.algo = algo
        self.hidden = hidden
        spec = get_spec(algo)

        self.ppd_fields = [
            f for f in spec.specific if f.struct is Struct.PER_PARTICLE_PER_DIM
        ]
        self.pd_fields = [f for f in spec.specific if f.struct is Struct.PER_DIM]
        self.has_cov = any(f.struct is Struct.MATRIX for f in spec.specific)
        self.scalar_fields = [f for f in spec.specific if f.struct is Struct.SCALAR]

        # per-cell input channels: positions + PPD specific (ppd) ;
        # values (pp, broadcast) ; best_x + PD specific (pd, broadcast)
        c_ppd = 1 + len(self.ppd_fields)
        c_pp = 1
        c_pd = 1 + len(self.pd_fields)
        self.cell_mlp = MLP([c_ppd + c_pp + c_pd, hidden, hidden], last_act=True)

        self.pool = ParticlePool()
        self.pool_proj = nn.Linear(2 * hidden, hidden)
        self.attn = nn.ModuleList(
            [DimAttention(hidden, n_heads) for _ in range(n_layers)]
        )

        # global scalars: best_y(std), log N, log D, + one per scalar field
        self.global_mlp = MLP(
            [hidden + 3 + len(self.scalar_fields), hidden, hidden], last_act=True
        )

    def forward(self, state: CanonicalState, ctx: NormContext) -> Latent:
        pos = state.positions  # (B, N, D)
        B, N, D = pos.shape

        pos_n = ctx.normalize(pos, Struct.PER_PARTICLE_PER_DIM, position_like=True)
        ppd = [pos_n.unsqueeze(-1)]
        for f in self.ppd_fields:
            t = ctx.normalize(state.specific[f.name], f.struct, f.name in POSITION_LIKE)
            ppd.append(t.unsqueeze(-1))
        ppd = torch.cat(ppd, dim=-1)  # (B, N, D, c_ppd)

        val_n = ctx.normalize(state.values, Struct.PER_PARTICLE, False)  # (B, N)
        pp = val_n.unsqueeze(-1).unsqueeze(2).expand(B, N, D, 1)  # broadcast over D

        best_n = ctx.normalize(state.best_x, Struct.PER_DIM, True)  # (B, D)
        pd_list = [best_n.unsqueeze(-1)]
        for f in self.pd_fields:
            t = ctx.normalize(state.specific[f.name], f.struct, f.name in POSITION_LIKE)
            pd_list.append(t.unsqueeze(-1))
        pd = torch.cat(pd_list, dim=-1)  # (B, D, c_pd)
        pd = pd.unsqueeze(1).expand(B, N, D, pd.shape[-1])  # broadcast over N

        cells = torch.cat([ppd, pp, pd], dim=-1)  # (B, N, D, C)
        cells = self.cell_mlp(cells)  # (B, N, D, H)

        tokens = self.pool_proj(self.pool(cells))  # (B, D, H)

        cov_bias = self._cov_bias(state, ctx, pos_n)
        for layer in self.attn:
            tokens = layer(tokens, cov_bias)

        g = tokens.mean(dim=1)  # (B, H)
        scalars = self._global_scalars(state, ctx, N, D)
        global_vec = self.global_mlp(torch.cat([g, scalars], dim=-1))
        return Latent(dim_tokens=tokens, global_vec=global_vec)

    # ------------------------------------------------------------------ #

    def _cov_bias(self, state, ctx, pos_n: Tensor) -> Tensor:
        # Empirical covariance of normalized positions (B, D, D).
        xc = pos_n - pos_n.mean(dim=1, keepdim=True)
        cov_emp = xc.transpose(1, 2) @ xc / max(pos_n.shape[1], 1)
        if self.has_cov:
            cov_n = ctx.normalize(state.specific["sigma_cov"], Struct.MATRIX, False)
            return cov_emp + cov_n
        return cov_emp

    def _global_scalars(self, state, ctx, N: int, D: int) -> Tensor:
        B = state.positions.shape[0]
        device = state.positions.device
        best_y_std = ((state.best_y - ctx.val_mean) / ctx.val_std).reshape(B, 1)
        log_n = torch.full((B, 1), float(N), device=device).log1p()
        log_d = torch.full((B, 1), float(D), device=device).log1p()
        feats = [best_y_std, log_n, log_d]
        for f in self.scalar_fields:
            s = ctx.normalize(state.specific[f.name], Struct.SCALAR, False).reshape(
                B, 1
            )
            feats.append(s)
        return torch.cat(feats, dim=-1)
