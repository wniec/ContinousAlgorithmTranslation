"""Actor-critic policy for PPO.

The **actor** is the existing set-equivariant ``TranslatorPair``: it encodes the
source optimizer's state and decodes the *mean* action (the target's specific
fields, in normalized coordinates). A diagonal Gaussian with a learnable global
log-std turns that mean into a stochastic policy; the covariance action is
sampled in factor space so it stays PSD. The **critic** is a small MLP on the
encoder's global latent plus the env context vector.

The same network thus serves supervised translation (deterministic mean) and RL
(sampled actions), with the heads' ``action_mean`` / ``build_state`` interface
(cat/models/heads.py) doing the packing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from cat.data.dataset import collate
from cat.models.encoder import StateEncoder
from cat.models.layers import MLP
from cat.models.norm import NormContext
from cat.models.translator import TranslatorPair
from cat.state.canonical import CanonicalState, from_canonical, to_canonical

CONTEXT_DIM = 2


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
    ):
        super().__init__()
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
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))
        self.algo_a = algo_a
        self.algo_b = algo_b

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
        # `context` overrides obs["context"] with the (running-)normalized vector.
        ctx_vec = obs["context"] if context is None else context
        context = torch.as_tensor(ctx_vec, dtype=torch.float32, device=device)

        z = self.translator.encode(batch, ctx)
        mean = self.translator.decoders[target].action_mean(z, batch.positions, ctx)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self._value(obs["source_algo"], batch, ctx, context.unsqueeze(0))

        native = self._assemble_native(action, target, batch, ctx)
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

    def _assemble_native(self, action: Tensor, target: str, batch, ctx) -> dict:
        specific = self.translator.decoders[target].build_state(action, ctx, batch.n)
        tgt_state = CanonicalState(
            algo=target,
            positions=batch.positions,
            values=batch.values,
            best_x=batch.best_x,
            best_y=batch.best_y,
            specific=specific,
        )
        return from_canonical(tgt_state.index(0))

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

        Returns (log_prob, entropy, value, batch, ctx) — the batched state and
        ctx are reused by the cycle-consistency auxiliary loss.
        """
        batch = collate(states)
        ctx = _ctx(batch)
        z = self.translator.encode(batch, ctx)
        mean = self.translator.decoders[target_algo].action_mean(
            z, batch.positions, ctx
        )
        std = self.log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self._value(source_algo, batch, ctx, contexts)
        return log_prob, entropy, value, batch, ctx

    def cycle_drift(self, batch: CanonicalState, ctx: NormContext) -> Tensor:
        """Mean A->B->A field drift for a batch (the cycle-consistency penalty)."""
        from cat.losses import field_distance

        _, back = self.translator.cycle(batch, ctx)
        return field_distance(back, batch, ctx)
