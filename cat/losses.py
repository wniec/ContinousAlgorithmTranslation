"""Training objectives for the state translator.

Three terms, all evaluated in the normalized frame (via ``NormContext``) so
fields of very different raw scale contribute comparably:

* **cycle-consistency** — the user's rule: with the population held fixed,
  ``A -> B -> A`` must change A's own algorithm-specific parameters as little as
  possible. Two interchangeable ways to score the round trip (picked via
  ``batch_losses(..., cycle_mode=...)``):
    - ``"field"`` (default) — ``field_distance(round_trip_A, A)``, comparing the
      fully decoded specific fields.
    - ``"latent"`` — ``latent_cycle_loss``, comparing ``Encoder_A(A)`` to
      ``Encoder_A(round_trip_A)`` directly, without ever looking at decoded
      fields. Cheaper to game by a decoder that reproduces fields the loss
      never inspects, but scores exactly the representation the translator
      actually routes through, and needs no per-field weighting.
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

from cat.models.encoder import Latent
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


def latent_distance(a: Latent, b: Latent) -> Tensor:
    """MSE between two latents (dim-tokens + global vector), averaged evenly.

    Unlike ``field_distance`` there is no per-field weighting to worry about:
    both tensors already live in the same learned hidden space."""
    tok = torch.mean((a.dim_tokens - b.dim_tokens) ** 2)
    glob = torch.mean((a.global_vec - b.global_vec) ** 2)
    return 0.5 * (tok + glob)


def latent_cycle_loss(
    pair, state: CanonicalState, back: CanonicalState, ctx: NormContext
) -> Tensor:
    """Cycle-consistency scored in latent space: ``Encoder_A(A)`` vs.
    ``Encoder_A(A -> B -> A)``. ``back`` is the already-decoded round trip (e.g.
    from ``pair.cycle``); this only adds the two re-encodes and the comparison."""
    z_state = pair.encode(state, ctx)
    z_back = pair.encode(back, ctx)
    return latent_distance(z_state, z_back)


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


def bobyqa_utility(mid: CanonicalState, ctx: NormContext) -> Tensor:
    """The decoded quadratic model (centred on best_x) should *explain the
    landscape*: its predicted increase away from the centre must track the
    population's actual fitnesses. Standardized-MSE between the model's
    predictions and the standardized objective values, in the normalized frame
    — BOBYQA's analogue of ``cmaes_utility`` (elites likely under the decoded
    distribution)."""
    pos_n = ctx.normalize(mid.positions, Struct.PER_PARTICLE_PER_DIM, True)  # (B,N,D)
    best_n = ctx.normalize(mid.best_x, Struct.PER_DIM, True)  # (B,D)
    g = ctx.normalize(mid.specific["grad"], Struct.PER_DIM, False)  # (B,D)
    H = ctx.normalize(mid.specific["hessian"], Struct.MATRIX, False)  # (B,D,D)

    d = pos_n - best_n.unsqueeze(1)  # (B,N,D) displacement from the model centre
    lin = (d * g.unsqueeze(1)).sum(-1)  # (B,N)
    quad = 0.5 * torch.einsum("bnd,bde,bne->bn", d, H, d)  # (B,N)
    pred = lin + quad  # model value relative to the centre

    val_n = ctx.normalize(mid.values, Struct.PER_PARTICLE, False)  # (B,N) standardized
    pred = pred - pred.mean(dim=1, keepdim=True)
    pred = pred / pred.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.mean((pred - val_n) ** 2)


_UTILITY = {"CMAES": cmaes_utility, "PSO": pso_utility, "BOBYQA": bobyqa_utility}


def utility_loss(mid: CanonicalState, ctx: NormContext) -> Tensor:
    fn = _UTILITY.get(mid.algo)
    if fn is None:
        return mid.positions.new_zeros(())
    return fn(mid, ctx)


# --------------------------------------------------------------------------- #
# Combined objective for one batch                                            #
# --------------------------------------------------------------------------- #


def batch_losses(
    pair,
    state: CanonicalState,
    ctx: NormContext,
    cycle_mode: str = "field",
    w_cycle: float = 1.0,
) -> dict[str, Tensor]:
    """Run cycle + reconstruction + utility for one homogeneous batch.

    ``cycle_mode`` picks how the ``A -> B -> A`` round trip is scored: "field"
    (default, decoded specific fields) or "latent" (re-encoded latents only).
    ``w_cycle == 0.0`` skips the round trip's ``B -> A`` leg (and, for "latent",
    the re-encodes) entirely rather than computing it just to multiply by zero
    — ``mid`` (needed by the utility proxy) is still produced via the cheaper
    ``A -> B`` leg alone.
    """
    tgt = pair.other(state.algo)
    mid = pair.translate(state, tgt, ctx)
    recon = pair.reconstruct(state, ctx)
    if w_cycle == 0.0:
        cycle = state.positions.new_zeros(())
    elif cycle_mode == "field":
        back = pair.translate(mid, state.algo, ctx)
        cycle = field_distance(back, state, ctx)
    elif cycle_mode == "latent":
        back = pair.translate(mid, state.algo, ctx)
        cycle = latent_cycle_loss(pair, state, back, ctx)
    else:
        raise ValueError(f"unknown cycle_mode {cycle_mode!r}")
    return {
        "cycle": cycle,
        "recon": field_distance(recon, state, ctx),
        "utility": utility_loss(mid, ctx),
    }
