"""Declarative description of each optimizer's internal state.

Every optimizer's warm-start dict (``get_data()`` + a few attributes such as
``sigma``) splits into two parts:

* **shared** fields, common to all population-based optimizers and carried
  verbatim across a switch (these are "the circumstances"): the population
  ``positions`` / ``values`` and the best solution found so far. They are *not*
  translated.
* **specific** fields, the algorithm's own machinery that the neural translator
  learns to map between algorithms — PSO velocities and personal bests, CMA-ES
  covariance / step-size / evolution paths.

A ``StateSpec`` declares the specific fields of one optimizer: their canonical
name, tensor *structure* (how they scale with population size ``N`` and problem
dimension ``D``), a positivity constraint where relevant, and the native
warm-start key they map to. Adding a new optimizer to the translation study is
just adding one ``StateSpec`` (plus a decoder head; see cat/models/heads.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Struct(Enum):
    """How a field's shape relates to population size N and dimension D."""

    PER_PARTICLE_PER_DIM = "ppd"  # (N, D) e.g. velocities, personal-best positions
    PER_PARTICLE = "pp"  # (N,)   e.g. personal-best values
    PER_DIM = "pd"  # (D,)   e.g. distribution mean, evolution paths
    MATRIX = "mat"  # (D, D) e.g. covariance matrix
    SCALAR = "scalar"  # ()     e.g. step-size sigma


class Constraint(Enum):
    FREE = "free"
    POSITIVE = "positive"  # decoded through softplus; compared in log-space


@dataclass(frozen=True)
class Field:
    name: str  # canonical field name
    struct: Struct
    native_key: str  # key in the optimizer's warm-start / attribute dict
    constraint: Constraint = Constraint.FREE
    weight: float = 1.0  # relative weight in cycle / reconstruction loss


@dataclass(frozen=True)
class StateSpec:
    algo: str  # optimizer registry name (matches cat.optimizers.PORTFOLIO)
    specific: tuple[Field, ...]

    def field(self, name: str) -> Field:
        for f in self.specific:
            if f.name == name:
                return f
        raise KeyError(f"{self.algo} has no specific field {name!r}")

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.specific]


# Shared fields are identical across all optimizers and carried verbatim.
SHARED_FIELDS: tuple[Field, ...] = (
    Field("positions", Struct.PER_PARTICLE_PER_DIM, "x"),
    Field("values", Struct.PER_PARTICLE, "y"),
    Field("best_x", Struct.PER_DIM, "best_x"),
    Field("best_y", Struct.SCALAR, "best_y"),
)


# --------------------------------------------------------------------------- #
# Per-optimizer specifications                                                 #
# --------------------------------------------------------------------------- #

# Only the *geometric* PSO machinery is translated: the velocities (the
# momentum term, PSO's analogue of CMA-ES's search distribution) and the
# personal-best positions (the swarm's memory). The personal-best *values*
# (p_y) and neighbour-best positions (n_x) are objective-valued / derivable, so
# set_data fills them from defaults rather than the translator decoding raw
# fitness values.
PSO_SPEC = StateSpec(
    algo="PSO",
    specific=(
        Field("velocity", Struct.PER_PARTICLE_PER_DIM, "v", weight=1.0),
        Field("pbest_x", Struct.PER_PARTICLE_PER_DIM, "p_x", weight=1.0),
    ),
)

# CMA-ES's step-size ``sigma`` and shape matrix ``C`` are redundant up to a
# scalar — only the full search covariance ``Sigma = sigma**2 * C`` is
# observable in the sampled population. We translate that single object
# (``sigma_cov``); ``from_canonical`` splits it back into the standard CMA-ES
# pair (det(C)=1, sigma = det(Sigma)**(1/2D)). This avoids the ill-posed,
# numerically explosive separate normalization of an entangled (C, sigma).
CMAES_SPEC = StateSpec(
    algo="CMAES",
    specific=(
        Field("mean", Struct.PER_DIM, "mean", weight=1.0),
        Field("sigma_cov", Struct.MATRIX, "cm", weight=1.0),
        Field("p_c", Struct.PER_DIM, "p_c", weight=0.5),
        Field("p_s", Struct.PER_DIM, "p_s", weight=0.5),
    ),
)


_SPECS: dict[str, StateSpec] = {
    "PSO": PSO_SPEC,
    "SPSO": PSO_SPEC,
    "SPSOL": PSO_SPEC,
    "CPSO": PSO_SPEC,
    "CMAES": CMAES_SPEC,
}


def get_spec(algo: str) -> StateSpec:
    if algo not in _SPECS:
        raise KeyError(
            f"No StateSpec for optimizer {algo!r}. "
            f"Known: {sorted(set(_SPECS))}. Add one in cat/state/schema.py."
        )
    return _SPECS[algo]
