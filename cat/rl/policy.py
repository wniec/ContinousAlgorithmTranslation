"""Actor-critic policy for PPO.

The **actor** is the existing set-equivariant ``TranslatorPair``: it encodes the
source optimizer's state and decodes the *mean* action (the target's specific
fields, in normalized coordinates). A diagonal Gaussian with a learnable
per-segment log-std (see ``ActorCritic.log_std``) turns that mean into a
stochastic policy; the covariance action is sampled in factor space so it stays
PSD. The **critic** is a small MLP on the
encoder's global latent plus the env context vector.

The same network thus serves supervised translation (deterministic mean) and RL
(sampled actions), with the heads' ``action_mean`` / ``build_state`` interface
(cat/models/heads.py) doing the packing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from cat.data.dataset import collate
from cat.models.encoder import Latent, StateEncoder
from cat.models.layers import MLP
from cat.models.norm import NormContext
from cat.models.translator import TranslatorPair
from cat.state.canonical import (
    CanonicalState,
    from_canonical,
    resample_population,
    to_canonical,
)

PROGRESS_DIM = 2
# The critic's side-input: progress scalars only. (Only the critic sees these;
# the actor/translator stays a function of the optimizer state alone.)
CONTEXT_DIM = PROGRESS_DIM


@dataclass
class Step:
    """Everything produced for one acted transition (used by the PPO buffer)."""

    src_state: CanonicalState  # un-batched, on cpu
    target_algo: str
    action: Tensor  # (A,) sampled normalized action coords
    log_prob: float
    value: float
    native: dict  # target warm-start dict to pass to env.step


def _ctx(state: CanonicalState) -> NormContext:
    return NormContext.from_shared(state.positions, state.values)


class ActorCritic(nn.Module):
    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        hidden: int = 64,
        n_layers: int = 2,
        cov_rank: int = 4,
        log_std_init: float = -1.0,
        resample_seed: int | None = None,
    ):
        super().__init__()
        # Drives the (independent of torch's RNG) population-resizing jitter in
        # _assemble_native, so it's reproducible across runs given a seed.
        self._resample_rng = np.random.default_rng(resample_seed)
        self.translator = TranslatorPair(
            algo_a, algo_b, hidden=hidden, n_layers=n_layers, cov_rank=cov_rank
        )
        # The critic has its OWN encoder per source algorithm. Sharing the
        # actor's encoder makes the critic chase a representation that the
        # policy + cycle losses constantly reshape (a moving target) and lets a
        # diverging value loss corrupt the actor; a separate encoder decouples
        # the two and keeps the value gradient off the actor.
        self.critic_encoders = nn.ModuleDict(
            {a: StateEncoder(a, hidden, n_layers) for a in (algo_a, algo_b)}
        )
        self.critic = MLP([hidden + CONTEXT_DIM, hidden, 1])
        # Exploration std = a per-segment *base* (below) plus a per-dimension,
        # population-conditioned residual (the std heads that follow). The base
        # is one log-std per action segment per target algorithm: a single
        # scalar would force one exploration scale onto fields of very different
        # natural magnitude (CMA-ES's sigma vs. its covariance factor vs. PSO's
        # velocities); per-*element* static params are impossible (the action
        # dim varies with D and N), so the segment is the finest shape-invariant
        # static granularity.
        self.log_std = nn.ParameterDict(
            {
                a: nn.Parameter(
                    torch.full(
                        (len(self.translator.decoders[a].segment_names()),),
                        float(log_std_init),
                    )
                )
                for a in (algo_a, algo_b)
            }
        )
        # Per-dimension, coordinate-equivariant std residual. Each D-indexed
        # segment (grad / velocities / archive / covariance factor+diagonal)
        # gets its *own* log-std per dimension-token, read from the encoder
        # latent — so the exploration scale actually adapts to the population
        # (and permutes with the coordinate axes, keeping the policy
        # equivariant). Scalar segments (e.g. BOBYQA's radius) read one residual
        # from the global vector. Zero-initialised, so training starts exactly
        # at the per-segment base (``exp(log_std)``) and *learns* the modulation.
        self.std_token_head = nn.ModuleDict()
        self.std_global_head = nn.ModuleDict()
        for a in (algo_a, algo_b):
            names = self.translator.decoders[a].segment_names()
            n_dim_seg = sum(1 for nm in names if nm != "scalar")
            head = nn.Linear(hidden, n_dim_seg)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.std_token_head[a] = head
            if any(nm == "scalar" for nm in names):
                g = nn.Linear(hidden, 1)
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
                self.std_global_head[a] = g
        self.algo_a = algo_a
        self.algo_b = algo_b

    # Where the ``D`` (dimension-token) axis sits inside each segment's
    # per-sample shape, so a per-dimension std broadcasts onto it and over the
    # remaining (particle / rank / field) axes.
    _D_AXIS = {"pd": 0, "cov_L": 0, "cov_d": 0, "ppd": 1, "set": 1}
    _STD_DELTA = 1.0  # residual half-range in log-space: std in (e^-1, e^1)*base

    @staticmethod
    def _broadcast_dim(v: Tensor, shape: tuple[int, ...], d_axis: int) -> Tensor:
        """Place a per-dimension vector ``v`` (B, D) onto axis ``d_axis`` of a
        segment shaped ``(B, *shape)`` and broadcast over the other axes."""
        B, D = v.shape
        view = [1] * len(shape)
        view[d_axis] = D
        return v.reshape(B, *view).expand(B, *shape)

    def action_std(self, target_algo: str, z: "Latent", n: int) -> Tensor:
        """Per-coordinate std, shape ``(B, action_dim(D, n))``.

        Each segment's std is its learnable base plus a bounded, population-
        conditioned residual: per dimension-token for the D-indexed segments
        (read from ``z.dim_tokens``), one global residual for scalar segments
        (from ``z.global_vec``). Following the decoder's own segment layout and
        flattening keeps the ordering aligned with ``action_mean``."""
        dec = self.translator.decoders[target_algo]
        B, D, _ = z.dim_tokens.shape
        base = self.log_std[target_algo]
        names = dec.segment_names()
        base_idx = {nm: i for i, nm in enumerate(names)}
        dim_col = {nm: k for k, nm in enumerate(nm for nm in names if nm != "scalar")}

        tok_delta = self.std_token_head[target_algo](z.dim_tokens)  # (B, D, n_dim_seg)

        parts: list[Tensor] = []
        for name, shape in dec._layout(D, n):
            size = 1
            for s in shape:
                size *= s
            if name == "scalar":
                g = self.std_global_head[target_algo](z.global_vec)[:, 0]  # (B,)
                val = base[base_idx[name]] + self._STD_DELTA * torch.tanh(g)
                chunk = val.reshape(B, 1).expand(B, size)
            else:
                per_dim = base[base_idx[name]] + self._STD_DELTA * torch.tanh(
                    tok_delta[:, :, dim_col[name]]
                )  # (B, D)
                chunk = self._broadcast_dim(per_dim, shape, self._D_AXIS[name])
                chunk = chunk.reshape(B, size)
            parts.append(chunk)
        return torch.cat(parts, dim=-1).exp()  # (B, action_dim)

    def _value(self, source_algo: str, batch, ctx, context: Tensor) -> Tensor:
        zc = self.critic_encoders[source_algo](batch, ctx)
        return self.critic(torch.cat([zc.global_vec, context], dim=-1)).squeeze(-1)

    # ------------------------------------------------------------------ #
    # rollout                                                             #
    # ------------------------------------------------------------------ #

    def obs_to_state(self, obs: dict, device="cpu") -> CanonicalState:
        return to_canonical(obs["source_algo"], obs["native"], device=device)

    @torch.no_grad()
    def act(
        self, obs: dict, device="cpu", deterministic: bool = False, context=None
    ) -> Step:
        src = self.obs_to_state(obs, device)
        target = obs["target_algo"]
        batch = collate([src])
        ctx = _ctx(batch)
        # The critic side-input is the progress context. `context` (when given
        # by the PPO loop) is the running-normalized version; otherwise raw.
        if context is None:
            context = obs["context"]
        context = torch.as_tensor(context, dtype=torch.float32, device=device)

        z = self.translator.encode(batch, ctx)
        mean = self.translator.decoders[target].action_mean(z, batch.positions, ctx)
        std = self.action_std(target, z, batch.n)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self._value(obs["source_algo"], batch, ctx, context.unsqueeze(0))

        native = self._assemble_native(action, target, batch, ctx, obs.get("target_n"))
        return Step(
            # Keep the buffered state on CPU so a long rollout doesn't accumulate
            # GPU tensors; it is moved back to the device during the update.
            src_state=src.to("cpu"),
            target_algo=target,
            action=action.squeeze(0).cpu(),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            native=native,
        )

    def _assemble_native(
        self, action: Tensor, target: str, batch, ctx, n_target: int | None = None
    ) -> dict:
        specific = self.translator.decoders[target].build_state(action, ctx, batch.n)
        tgt_state = CanonicalState(
            algo=target,
            positions=batch.positions,
            values=batch.values,
            best_x=batch.best_x,
            best_y=batch.best_y,
            specific=specific,
        ).index(0)
        # The action itself is always decoded onto the *source's* own
        # population (see act()/evaluate_actions — this keeps log-prob/PPO
        # ratio computations independent of population resizing); only the
        # assembled warm-start payload is resized to the target's own N.
        if n_target is not None and n_target != tgt_state.n:
            tgt_state = resample_population(tgt_state, n_target, self._resample_rng)
        return from_canonical(tgt_state)

    # ------------------------------------------------------------------ #
    # PPO update                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_actions(
        self,
        states: list[CanonicalState],
        source_algo: str,
        target_algo: str,
        actions: Tensor,
        contexts: Tensor,
    ):
        """Recompute log-prob / entropy / value for a homogeneous group.

        Returns (log_prob, entropy, value, batch, ctx, std) — the batched state
        and ctx are reused by the cycle-consistency auxiliary loss; ``std`` is
        the effective (population-conditioned) per-coordinate std, for logging.
        """
        batch = collate(states)
        ctx = _ctx(batch)
        z = self.translator.encode(batch, ctx)
        mean = self.translator.decoders[target_algo].action_mean(
            z, batch.positions, ctx
        )
        std = self.action_std(target_algo, z, batch.n)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self._value(source_algo, batch, ctx, contexts)
        return log_prob, entropy, value, batch, ctx, std

    def cycle_drift(
        self, batch: CanonicalState, ctx: NormContext, cycle_mode: str = "field"
    ) -> Tensor:
        """Mean A->B->A drift for a batch (the cycle-consistency penalty); see
        ``TranslatorPair.cycle_drift`` (shared with TD3's ``TD3Actor``)."""
        return self.translator.cycle_drift(batch, ctx, cycle_mode)
