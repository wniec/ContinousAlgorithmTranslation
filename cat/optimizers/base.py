"""Base sub-optimizer.

Vendored from DynamicAlgorithmSelection2 (das/optimizers/base.py). Every
optimizer extends ``SubOptimizer``, which adds:
  - x/y population history tracking
  - target_fe early stopping (run for exactly one checkpoint interval)
  - set_data / get_data interface for warm-starting between optimizer switches

The ``get_data()`` / ``set_data()`` dicts are exactly the algorithm-specific
state that this project learns to translate between optimizers.
"""

import time
from typing import Any

import numpy as np
from pypop7.optimizers.core import Optimizer as _Pypop7Base, Terminations


class SubOptimizer(_Pypop7Base):
    """Base class for all portfolio optimizers.

    Warm-starting contract
    ----------------------
    After running, call ``get_data()`` to get the population state dict.
    Before running, call ``set_data(**state)`` with that dict to warm-start.
    Subclasses override ``set_data`` / ``get_data`` to add algorithm-specific
    keys. Unknown keys in ``set_data`` are silently ignored so different
    optimizer types can hand off to each other without type checks.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.best_so_far_y: float = options.get("best_so_far_y", float("inf"))
        self.best_so_far_x: np.ndarray | None = None
        self.worst_so_far_y: float = -np.inf
        self.worst_so_far_x: np.ndarray | None = None

        self.x_history: list[np.ndarray] = []
        self.y_history: list[float] = []
        self.fitness_history: list[tuple[int, float]] = []  # (n_fe, best_y) pairs

        self.target_fe: int = options.get("target_fe", int(1e9))
        self._warm_start: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # pypop7 hooks                                                         #
    # ------------------------------------------------------------------ #

    def _evaluate_fitness(self, x: np.ndarray, args=None) -> float:
        t0 = time.time()
        y = (
            self.fitness_function(x)
            if args is None
            else self.fitness_function(x, args=args)
        )
        self.time_function_evaluations += time.time() - t0
        self.n_function_evaluations += 1
        y_val = float(y)

        if y_val < self.best_so_far_y:
            self.best_so_far_x = np.copy(x)
            self.best_so_far_y = y_val
            self.fitness_history.append((self.n_function_evaluations, y_val))
        if y_val > self.worst_so_far_y:
            self.worst_so_far_x = np.copy(x)
            self.worst_so_far_y = y_val

        self.x_history.append(np.copy(x))
        self.y_history.append(y_val)
        return y_val

    def _check_terminations(self) -> bool:
        terminated = super()._check_terminations()
        if not terminated and self.n_function_evaluations >= self.target_fe:
            self.termination_signal = Terminations.MAX_FUNCTION_EVALUATIONS
            terminated = True
        return terminated

    def _collect(self, fitness: list) -> dict:
        result = super()._collect(fitness)
        result.update(
            {
                "x_history": np.array(self.x_history, dtype=np.float32),
                "y_history": np.array(self.y_history, dtype=np.float32),
                "fitness_history": self.fitness_history,
                "best_so_far_x": self.best_so_far_x,
                "best_so_far_y": self.best_so_far_y,
                "worst_so_far_x": self.worst_so_far_x,
                "worst_so_far_y": self.worst_so_far_y,
            }
        )
        return result

    # ------------------------------------------------------------------ #
    # Warm-start interface                                                 #
    # ------------------------------------------------------------------ #

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        """Load population from a previous optimizer run."""
        self._warm_start = {"x": x, "y": y}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        """Return current population state for the next optimizer."""
        return dict(self._warm_start)


def get_checkpoints(
    n_checkpoints: int, max_fe: int, n_individuals: int, cdb: float
) -> np.ndarray:
    """Compute exponentially-spaced checkpoint FE targets.

    cdb == 1.0  -> uniform spacing
    cdb > 1.0   -> early checkpoints are shorter (exponential growth)

    Retained for reference; the translation study uses ``sample_switch_points``
    instead so switch positions are randomized per problem rather than fixed.
    """
    ratios = np.cumprod(np.full(n_checkpoints, cdb))
    ratios = np.cumsum(ratios / ratios.sum())
    checkpoints = (ratios * max_fe).astype(int)
    checkpoints[-1] = max_fe
    checkpoints[0] = max(checkpoints[0], n_individuals)
    for i in range(1, n_checkpoints):
        checkpoints[i] = max(checkpoints[i - 1] + n_individuals, checkpoints[i])
    return checkpoints


def sample_switch_points(
    n_switches: int,
    max_fe: int,
    n_individuals: int,
    rng: np.random.Generator,
    fe_min: int | None = None,
) -> np.ndarray:
    """Sample ``n_switches`` cumulative FE targets with log-uniform positioning.

    Optimization progress is roughly linear in ``log(FE)`` — early evaluations
    move the objective far more than late ones — so switch points are drawn
    uniformly in log-FE space (equal probability per decade), which places more
    switches early and fewer late. Positions are sampled independently on each
    call (i.e. independently per problem). The interior points are sorted; the
    final target is always ``max_fe`` so the full budget is used.

    Returns a strictly increasing int array of length ``n_switches`` whose last
    entry is ``max_fe`` and whose consecutive gaps are at least
    ``n_individuals`` (so every segment can evaluate at least one population).
    """
    step = max(int(n_individuals), 1)
    fe_min = int(fe_min) if fe_min is not None else step
    fe_min = max(2, min(fe_min, max_fe))
    if n_switches <= 1:
        return np.array([max_fe], dtype=int)

    log_pts = rng.uniform(np.log(fe_min), np.log(max_fe), size=n_switches - 1)
    interior = np.exp(np.sort(log_pts))
    # Leave room for the remaining segments + the final max_fe target.
    interior = np.clip(interior, fe_min, max_fe - step)
    points = np.concatenate([interior, [max_fe]]).astype(int)

    points[0] = max(points[0], fe_min)
    for i in range(1, n_switches):
        points[i] = max(points[i], points[i - 1] + step)
    points[-1] = max(points[-1], max_fe)  # always finish on the full budget
    return points
