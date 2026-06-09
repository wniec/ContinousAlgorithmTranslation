"""Permutation-respecting building blocks for the state translator.

* ``MLP`` — plain feed-forward block.
* ``ParticlePool`` — DeepSets mean+max pooling over the particle axis
  (permutation-invariant over particles).
* ``DimAttention`` — multi-head self-attention over the ``D`` dimension-tokens
  with an additive bias built from a covariance matrix; equivariant to a
  permutation of coordinate axes (permuting the tokens permutes the bias rows
  and columns identically).
* ``CovFactorHead`` — decodes a positive-semidefinite covariance as
  ``L Lᵀ + diag(d)`` from per-dimension tokens, so it is PSD by construction and
  coordinate-equivariant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, sizes: list[int], act=nn.GELU, last_act=False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2 or last_act:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ParticlePool(nn.Module):
    """Pool a (B, N, D, H) cell grid over the particle axis -> (B, D, 2H)."""

    def forward(self, cells: Tensor) -> Tensor:
        mean = cells.mean(dim=1)
        mx = cells.amax(dim=1)
        return torch.cat([mean, mx], dim=-1)


class DimAttention(nn.Module):
    """Self-attention over D dimension-tokens with a covariance attention bias."""

    def __init__(self, hidden: int, n_heads: int = 4):
        super().__init__()
        assert hidden % n_heads == 0
        self.h = hidden
        self.n_heads = n_heads
        self.dk = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        # Learnable scaling of the covariance-derived attention bias.
        self.bias_scale = nn.Parameter(torch.zeros(n_heads))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = MLP([hidden, 2 * hidden, hidden])

    def forward(self, tokens: Tensor, cov_bias: Tensor | None) -> Tensor:
        # tokens (B, D, H); cov_bias (B, D, D) or None
        B, D, _ = tokens.shape
        x = self.norm1(tokens)
        qkv = self.qkv(x).reshape(B, D, 3, self.n_heads, self.dk)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, D, dk)
        logits = (q @ k.transpose(-2, -1)) / (self.dk**0.5)  # (B, heads, D, D)
        if cov_bias is not None:
            logits = logits + self.bias_scale.view(1, -1, 1, 1) * cov_bias.unsqueeze(1)
        attn = F.softmax(logits, dim=-1)
        out = attn @ v  # (B, heads, D, dk)
        out = out.transpose(1, 2).reshape(B, D, self.h)
        tokens = tokens + self.proj(out)
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens


class CovFactorHead(nn.Module):
    """Decode a PSD covariance ``L Lᵀ + diag(softplus(d))`` from dim-tokens.

    ``factor`` exposes the raw factor coordinates ``(L, d_raw)`` so an RL policy
    can place a Gaussian on them and still obtain a PSD covariance via
    ``assemble``; ``forward`` is the deterministic ``assemble(factor(...))``.
    """

    def __init__(self, hidden: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.to_factor = nn.Linear(hidden, rank)  # row of L per dimension
        self.to_diag = nn.Linear(hidden, 1)

    def factor(self, dim_tokens: Tensor) -> tuple[Tensor, Tensor]:
        L = self.to_factor(dim_tokens)  # (B, D, rank)
        d_raw = self.to_diag(dim_tokens).squeeze(-1)  # (B, D), pre-softplus
        return L, d_raw

    @staticmethod
    def assemble(L: Tensor, d_raw: Tensor) -> Tensor:
        diag = F.softplus(d_raw) + 1e-4
        return L @ L.transpose(-2, -1) + torch.diag_embed(diag)

    def forward(self, dim_tokens: Tensor) -> Tensor:
        # dim_tokens (B, D, H) -> covariance (B, D, D), normalized frame
        return self.assemble(*self.factor(dim_tokens))
