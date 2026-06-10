"""Per-sample normalization derived from the shared population.

All state fields live in problem coordinates (BBOB domain, often [-5, 5]) with
objective values spanning many orders of magnitude. The model works in a
normalized frame instead: positions are centred on the population mean and
scaled by a single robust scale; displacements (velocities, evolution paths)
are scaled; the covariance scales by ``scale**2``.

Crucially the normalization context is computed from the **shared** fields
(``positions`` / ``values``), which are held fixed throughout a translation
cycle. So the same context normalizes the A-state, the translated B-state, and
the round-tripped A-state — a precondition for the cycle-consistency loss to be
meaningful. The context is per-dim for the centre (permutes with coordinate
axes) and a single scalar for the scale (permutation-invariant), which keeps the
whole pipeline equivariant to coordinate permutations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cat.state.schema import Struct

_EPS = 1e-6
# BBOB lives in [-5, 5]; floor the population scale well above zero so a
# degenerate (collapsed-to-a-point) population can't make normalized quantities
# explode. Clip normalized values to a sane band for the same reason — this
# bounds the encoder inputs, the critic, and the cycle/recon losses without
# touching the raw states fed to the optimizers (denormalize is left exact).
_SCALE_FLOOR = 1e-2
_CLIP = 50.0


@dataclass
class NormContext:
    center: Tensor  # (B, D) per-dim population centre
    scale: Tensor  # (B,)   single robust scale
    val_mean: Tensor  # (B,) value mean (for standardizing objective features)
    val_std: Tensor  # (B,)

    @classmethod
    def from_shared(cls, positions: Tensor, values: Tensor) -> "NormContext":
        # positions (B, N, D), values (B, N)
        center = positions.mean(dim=-2)  # (B, D)
        centered = positions - center.unsqueeze(-2)
        scale = (
            centered.pow(2).mean(dim=(-2, -1)).sqrt().clamp_min(_SCALE_FLOOR)
        )  # (B,)
        val_mean = values.mean(dim=-1)
        val_std = values.std(dim=-1).clamp_min(_EPS)
        return cls(center, scale, val_mean, val_std)

    # -- broadcasting helpers -------------------------------------------- #

    def _c(self, ndim: int) -> Tensor:
        # center reshaped to broadcast against a (..., D) tensor with `ndim` dims
        if ndim == 2:  # (B, D)
            return self.center
        if ndim == 3:  # (B, N, D)
            return self.center.unsqueeze(-2)
        raise ValueError(ndim)

    def _s(self, ndim: int) -> Tensor:
        return self.scale.reshape((-1,) + (1,) * (ndim - 1))

    # -- field (de)normalization ----------------------------------------- #

    def normalize(self, t: Tensor, struct: Struct, position_like: bool) -> Tensor:
        if struct is Struct.MATRIX:  # (B, D, D)
            out = t / self.scale.reshape(-1, 1, 1).pow(2)
        elif struct is Struct.SCALAR:  # (B,) e.g. sigma -> log(sigma / scale)
            out = torch.log((t / self.scale).clamp_min(_EPS))
        elif struct is Struct.PER_PARTICLE:  # value-like feature
            out = (t - self.val_mean.unsqueeze(-1)) / self.val_std.unsqueeze(-1)
        elif position_like:  # PER_DIM (B,D) or PER_PARTICLE_PER_DIM (B,N,D)
            out = (t - self._c(t.dim())) / self._s(t.dim())
        else:  # displacement
            out = t / self._s(t.dim())
        return out.clamp(-_CLIP, _CLIP)

    def denormalize(self, t: Tensor, struct: Struct, position_like: bool) -> Tensor:
        if struct is Struct.MATRIX:
            return t * self.scale.reshape(-1, 1, 1).pow(2)
        if struct is Struct.SCALAR:
            return torch.exp(t) * self.scale
        if struct is Struct.PER_PARTICLE:
            return t * self.val_std.unsqueeze(-1) + self.val_mean.unsqueeze(-1)
        if position_like:
            return t * self._s(t.dim()) + self._c(t.dim())
        return t * self._s(t.dim())


# Position-like canonical fields (centre + scale); everything else is a
# displacement (scale only), a matrix, or a scalar.
POSITION_LIKE = {"positions", "best_x", "pbest_x", "nbest_x", "mean", "archive"}
