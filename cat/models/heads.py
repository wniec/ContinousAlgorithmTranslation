"""Per-algorithm decoder head.

Consumes the shared latent ``z`` and the *target* shared population (positions +
normalization context) and emits that algorithm's specific fields in raw units:

* PER_DIM (mean, evolution paths): linear read-out per dimension-token.
* PER_PARTICLE_PER_DIM (velocity, personal-best positions): a per-cell MLP whose
  query for particle *i*, dimension *j* is ``concat(dim_token_j, normalized
  position x_ij)`` — so generated velocities are grounded in the actual swarm
  and the head stays equivariant over both particles and coordinates.
* MATRIX (covariance): ``CovFactorHead`` -> PSD by construction.
* SCALAR (sigma): linear read-out from the global vector, decoded in log-space
  (always positive after denormalization).

The head also exposes an **action interface** for the RL trainer:
``action_mean`` returns the flat normalized coordinates an RL policy puts a
Gaussian over, and ``build_state`` rebuilds the raw fields from (possibly
noised) coordinates. The supervised ``forward`` is exactly
``build_state(action_mean(...))``, so both paths share one network.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cat.models.encoder import Latent
from cat.models.layers import MLP, CovFactorHead
from cat.models.norm import POSITION_LIKE, NormContext
from cat.optimizers.DE import ARATE
from cat.state.schema import Struct, get_spec


class StateDecoder(nn.Module):
    def __init__(self, algo: str, hidden: int = 64, cov_rank: int = 4):
        super().__init__()
        self.algo = algo
        self.hidden = hidden
        self.cov_rank = cov_rank
        spec = get_spec(algo)
        self.spec = spec

        self.ppd_fields = [
            f for f in spec.specific if f.struct is Struct.PER_PARTICLE_PER_DIM
        ]
        self.pd_fields = [f for f in spec.specific if f.struct is Struct.PER_DIM]
        self.scalar_fields = [f for f in spec.specific if f.struct is Struct.SCALAR]
        self.set_fields = [f for f in spec.specific if f.struct is Struct.POINT_SET]
        self.cov_field = next(
            (f for f in spec.specific if f.struct is Struct.MATRIX), None
        )

        if self.pd_fields:
            self.pd_head = nn.Linear(hidden, len(self.pd_fields))
        if self.ppd_fields:
            # input: dim-token (H) + normalized position scalar (1)
            self.ppd_head = MLP([hidden + 1, hidden, len(self.ppd_fields)])
        if self.set_fields:
            # input: dim-token (H) + the slot's seed-position coord (1)
            self.set_head = MLP([hidden + 1, hidden, len(self.set_fields)])
        if self.scalar_fields:
            self.scalar_head = nn.Linear(hidden, len(self.scalar_fields))
        if self.cov_field is not None:
            self.cov_head = CovFactorHead(hidden, rank=cov_rank)

        self._small_init()

    @staticmethod
    def _m_cap(n: int) -> int:
        return max(1, int(round(ARATE * n)))

    def _small_init(self):
        """Start near zero so initial decoded fields (in normalized space) are
        small — keeps the first training steps stable."""
        for head in ("pd_head", "ppd_head", "set_head", "scalar_head"):
            mod = getattr(self, head, None)
            if mod is None:
                continue
            last = mod if isinstance(mod, nn.Linear) else mod.net[-1]
            nn.init.zeros_(last.bias)
            last.weight.data.mul_(0.01)

    # ------------------------------------------------------------------ #
    # action coordinate layout                                            #
    # ------------------------------------------------------------------ #

    def _layout(self, D: int, N: int) -> list[tuple[str, tuple[int, ...]]]:
        """Ordered (segment-name, per-sample shape) of the flat action vector."""
        segs: list[tuple[str, tuple[int, ...]]] = []
        if self.pd_fields:
            segs.append(("pd", (D, len(self.pd_fields))))
        if self.ppd_fields:
            segs.append(("ppd", (N, D, len(self.ppd_fields))))
        if self.set_fields:
            segs.append(("set", (self._m_cap(N), D, len(self.set_fields))))
        if self.cov_field is not None:
            segs.append(("cov_L", (D, self.cov_rank)))
            segs.append(("cov_d", (D,)))
        if self.scalar_fields:
            segs.append(("scalar", (len(self.scalar_fields),)))
        return segs

    def action_dim(self, D: int, N: int) -> int:
        total = 0
        for _, shape in self._layout(D, N):
            n = 1
            for s in shape:
                n *= s
            total += n
        return total

    def action_mean(self, z: Latent, positions: Tensor, ctx: NormContext) -> Tensor:
        """Flat normalized action-mean coordinates, shape (B, action_dim)."""
        tokens = z.dim_tokens  # (B, D, H)
        B, D, _ = tokens.shape
        parts: list[Tensor] = []

        if self.pd_fields:
            parts.append(self.pd_head(tokens).reshape(B, -1))  # (B, D*n_pd)

        if self.ppd_fields:
            N = positions.shape[1]
            pos_n = ctx.normalize(positions, Struct.PER_PARTICLE_PER_DIM, True)
            tok = tokens.unsqueeze(1).expand(B, N, D, self.hidden)
            cell_in = torch.cat([tok, pos_n.unsqueeze(-1)], dim=-1)
            parts.append(self.ppd_head(cell_in).reshape(B, -1))  # (B, N*D*n_ppd)

        if self.set_fields:
            N = positions.shape[1]
            M = self._m_cap(N)
            seed = self._seed_positions(positions, M)  # (B, M, D)
            seed_n = ctx.normalize(seed, Struct.POINT_SET, True)
            tok = z.dim_tokens.unsqueeze(1).expand(B, M, D, self.hidden)
            cell_in = torch.cat([tok, seed_n.unsqueeze(-1)], dim=-1)
            parts.append(self.set_head(cell_in).reshape(B, -1))  # (B, M*D*n_set)

        if self.cov_field is not None:
            L, d_raw = self.cov_head.factor(tokens)
            parts.append(L.reshape(B, -1))  # (B, D*rank)
            parts.append(d_raw.reshape(B, -1))  # (B, D)

        if self.scalar_fields:
            parts.append(self.scalar_head(z.global_vec))  # (B, n_scalar)

        return torch.cat(parts, dim=-1)

    @staticmethod
    def _seed_positions(positions: Tensor, m: int) -> Tensor:
        """Seed ``m`` point-set slots deterministically by cycling the population
        (slot i <- population[i mod N]); keeps archive decode reproducible so the
        cycle-consistency loss is well-defined."""
        n = positions.shape[1]
        idx = torch.arange(m, device=positions.device) % n
        return positions[:, idx, :]

    def build_state(
        self, coords: Tensor, ctx: NormContext, n: int
    ) -> dict[str, Tensor]:
        """Rebuild raw specific fields from flat (possibly noised) action coords."""
        B = coords.shape[0]
        D = ctx.center.shape[-1]
        out: dict[str, Tensor] = {}
        cov_L: Tensor | None = None
        off = 0

        for name, shape in self._layout(D, n):
            size = 1
            for s in shape:
                size *= s
            chunk = coords[:, off : off + size].reshape(B, *shape)
            off += size

            if name == "pd":
                for i, f in enumerate(self.pd_fields):
                    out[f.name] = ctx.denormalize(
                        chunk[..., i], f.struct, f.name in POSITION_LIKE
                    )
            elif name == "ppd":
                for i, f in enumerate(self.ppd_fields):
                    out[f.name] = ctx.denormalize(
                        chunk[..., i], f.struct, f.name in POSITION_LIKE
                    )
            elif name == "cov_L":
                cov_L = chunk  # assembled together with the diagonal below
            elif name == "cov_d":
                cov_n = self.cov_head.assemble(cov_L, chunk)
                out[self.cov_field.name] = ctx.denormalize(cov_n, Struct.MATRIX, False)
            elif name == "set":
                for i, f in enumerate(self.set_fields):
                    out[f.name] = ctx.denormalize(
                        chunk[..., i], Struct.POINT_SET, f.name in POSITION_LIKE
                    )
            elif name == "scalar":
                for i, f in enumerate(self.scalar_fields):
                    out[f.name] = ctx.denormalize(chunk[:, i], Struct.SCALAR, False)

        return out

    # ------------------------------------------------------------------ #

    def forward(
        self, z: Latent, positions: Tensor, ctx: NormContext
    ) -> dict[str, Tensor]:
        coords = self.action_mean(z, positions, ctx)
        return self.build_state(coords, ctx, positions.shape[1])
