"""Bidirectional translator between two optimizers.

Holds an encoder and a decoder head per algorithm. Translating A->B means
``Encoder_A`` then ``Head_B``; the produced B-state *reuses A's shared
population* (positions / values / best), since those are the fixed
"circumstances" of the switch — only the algorithm-specific fields are decoded.

This makes the round trip well-defined: ``translate(A->B)`` then
``translate(B->A)`` returns an A-state on the same population, and the
cycle-consistency loss compares only the algorithm-specific fields.
"""

from __future__ import annotations

from torch import nn

from cat.models.encoder import Latent, StateEncoder
from cat.models.heads import StateDecoder
from cat.models.norm import NormContext
from cat.state.canonical import CanonicalState


def _with_specific(src: CanonicalState, algo: str, specific: dict) -> CanonicalState:
    """A new state: src's shared fields + freshly decoded specific fields."""
    return CanonicalState(
        algo=algo,
        positions=src.positions,
        values=src.values,
        best_x=src.best_x,
        best_y=src.best_y,
        specific=specific,
    )


class TranslatorPair(nn.Module):
    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        cov_rank: int = 4,
    ):
        super().__init__()
        self.algo_a = algo_a
        self.algo_b = algo_b
        self.encoders = nn.ModuleDict(
            {
                algo_a: StateEncoder(algo_a, hidden, n_layers, n_heads),
                algo_b: StateEncoder(algo_b, hidden, n_layers, n_heads),
            }
        )
        self.decoders = nn.ModuleDict(
            {
                algo_a: StateDecoder(algo_a, hidden, cov_rank),
                algo_b: StateDecoder(algo_b, hidden, cov_rank),
            }
        )

    # -- primitives ------------------------------------------------------- #

    def encode(self, state: CanonicalState, ctx: NormContext) -> Latent:
        return self.encoders[state.algo](state, ctx)

    def decode(
        self, z: Latent, target_algo: str, shared: CanonicalState, ctx: NormContext
    ) -> CanonicalState:
        specific = self.decoders[target_algo](z, shared.positions, ctx)
        return _with_specific(shared, target_algo, specific)

    def translate(
        self, state: CanonicalState, target_algo: str, ctx: NormContext
    ) -> CanonicalState:
        z = self.encode(state, ctx)
        return self.decode(z, target_algo, state, ctx)

    def reconstruct(self, state: CanonicalState, ctx: NormContext) -> CanonicalState:
        return self.translate(state, state.algo, ctx)

    def other(self, algo: str) -> str:
        return self.algo_b if algo == self.algo_a else self.algo_a

    def cycle(self, state: CanonicalState, ctx: NormContext):
        """A -> B -> A. Returns (mid_state_B, round_trip_state_A)."""
        tgt = self.other(state.algo)
        mid = self.translate(state, tgt, ctx)
        back = self.translate(mid, state.algo, ctx)
        return mid, back

    def cycle_drift(
        self, state: CanonicalState, ctx: NormContext, cycle_mode: str = "field"
    ):
        """Mean A->B->A drift for a batch (the cycle-consistency penalty), scored
        either on decoded fields ("field", default) or re-encoded latents
        ("latent") — see ``cat.losses`` for the rationale behind each. Shared by
        both RL trainers (PPO's ``ActorCritic``, TD3's ``TD3Actor``), which each
        wrap one ``TranslatorPair`` as their actor."""
        from cat.losses import field_distance, latent_cycle_loss

        _, back = self.cycle(state, ctx)
        if cycle_mode == "field":
            return field_distance(back, state, ctx)
        elif cycle_mode == "latent":
            return latent_cycle_loss(self, state, back, ctx)
        else:
            raise ValueError(f"unknown cycle_mode {cycle_mode!r}")
