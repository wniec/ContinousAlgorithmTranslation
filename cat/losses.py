"""Training objectives for the state translator.

Three terms, all evaluated in the normalized frame (via ``NormContext``) so
fields of very different raw scale contribute comparably:

* **cycle-consistency** — the user's rule: with the population held fixed,
  ``A -> B -> A`` must change A's own algorithm-specific parameters as little as
  possible. ``field_distance(round_trip_A, A)``.
* **reconstruction** — ``Encoder_X -> Head_X`` must reproduce X (an autoencoder
  anchor that stops the latent from collapsing).
* **utility proxy** — the *translated* state must be a good warm-start for the
  target optimizer, so the cycle cannot be solved by a useless near-identity
  mapping. CMA-ES target: elites are high-likelihood under the decoded search
  distribution. PSO target: decoded velocities point down-hill toward the best.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cat.models.norm import POSITION_LIKE, NormContext
from cat.state.canonical import CanonicalState
from cat.state.schema import Struct, get_spec


def field_distance(
    pred: CanonicalState, target: CanonicalState, ctx: NormContext
) -> Tensor:
    """Weighted MSE between two states' specific fields, in normalized space."""
    spec = get_spec(pred.algo)
    total = pred.positions.new_zeros(())
    wsum = 0.0
    for f in spec.specific:
        pos_like = f.name in POSITION_LIKE
        p = ctx.normalize(pred.specific[f.name], f.struct, pos_like)
        t = ctx.normalize(target.specific[f.name], f.struct, pos_like)
        total = total + f.weight * torch.mean((p - t) ** 2)
        wsum += f.weight
    return total / max(wsum, 1e-8)


# --------------------------------------------------------------------------- #
# Utility proxies                                                              #
# --------------------------------------------------------------------------- #


def _elite_indices(values: Tensor, k: int) -> Tensor:
    return torch.topk(values, k, dim=-1, largest=False).indices  # (B, k)


def cmaes_utility(mid: CanonicalState, ctx: NormContext) -> Tensor:
    """Negative log-likelihood of the elite population under the decoded
    Gaussian search distribution N(mean, Sigma), evaluated in the normalized
    position frame where Sigma_n = Sigma / scale**2."""
    B, N, D = mid.positions.shape
    k = max(1, N // 2)
    idx = _elite_indices(mid.values, k)  # (B, k)

    pos_n = ctx.normalize(mid.positions, Struct.PER_PARTICLE_PER_DIM, True)  # (B,N,D)
    elite = torch.gather(pos_n, 1, idx.unsqueeze(-1).expand(B, k, D))  # (B,k,D)

    mean_n = ctx.normalize(mid.specific["mean"], Struct.PER_DIM, True)  # (B,D)
    cov_n = ctx.normalize(mid.specific["sigma_cov"], Struct.MATRIX, False)  # (B,D,D)

    cov = cov_n + 1e-3 * torch.eye(D, device=pos_n.device)
    L = torch.linalg.cholesky(cov)  # (B,D,D)

    diff = elite - mean_n.unsqueeze(1)  # (B,k,D)
    sol = torch.linalg.solve_triangular(L, diff.transpose(1, 2), upper=False)  # (B,D,k)
    maha = (sol**2).sum(dim=1)  # (B,k)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (B,)
    nll = 0.5 * (maha.mean(dim=-1) + logdet)  # (B,)
    return nll.mean()


def pso_utility(mid: CanonicalState, ctx: NormContext) -> Tensor:
    """Encourage decoded velocities to point down-hill (toward best_x) and
    personal bests to sit near the swarm. Cosine alignment in normalized space."""
    pos_n = ctx.normalize(mid.positions, Struct.PER_PARTICLE_PER_DIM, True)  # (B,N,D)
    best_n = ctx.normalize(mid.best_x, Struct.PER_DIM, True)  # (B,D)
    vel_n = ctx.normalize(mid.specific["velocity"], Struct.PER_PARTICLE_PER_DIM, False)

    descent = best_n.unsqueeze(1) - pos_n  # (B,N,D) toward best
    cos = torch.nn.functional.cosine_similarity(vel_n, descent, dim=-1, eps=1e-6)
    align = -cos.mean()  # minimize -> maximize alignment

    # personal-best memory should stay near the population (not run off to infinity)
    pbest_n = ctx.normalize(mid.specific["pbest_x"], Struct.PER_PARTICLE_PER_DIM, True)
    anchor = torch.mean((pbest_n - pos_n) ** 2)
    return align + 0.1 * anchor


_UTILITY = {"CMAES": cmaes_utility, "PSO": pso_utility}


def utility_loss(mid: CanonicalState, ctx: NormContext) -> Tensor:
    fn = _UTILITY.get(mid.algo)
    if fn is None:
        return mid.positions.new_zeros(())
    return fn(mid, ctx)


# --------------------------------------------------------------------------- #
# Combined objective for one batch                                            #
# --------------------------------------------------------------------------- #


def batch_losses(pair, state: CanonicalState, ctx: NormContext) -> dict[str, Tensor]:
    """Run cycle + reconstruction + utility for one homogeneous batch."""
    mid, back = pair.cycle(state, ctx)
    recon = pair.reconstruct(state, ctx)
    return {
        "cycle": field_distance(back, state, ctx),
        "recon": field_distance(recon, state, ctx),
        "utility": utility_loss(mid, ctx),
    }
