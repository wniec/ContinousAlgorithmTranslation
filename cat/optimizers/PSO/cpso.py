"""CPSO: Cooperative Particle Swarm Optimization."""

import numpy as np

from .base import PSO


class CPSO(PSO):
    """Cooperative PSO: optimizes one coordinate at a time.

    Each particle's velocity and position are updated per dimension. The
    trial candidate is evaluated as best_so_far with only the current
    dimension swapped -- all other dimensions stay frozen at their personal
    best value.
    """

    def __init__(self, problem: dict, options: dict):
        options = dict(options)
        options.setdefault("cognition", 1.49)
        options.setdefault("society", 1.49)
        super().__init__(problem, options)
        # CPSO inertia decays from 1.0 to 0.0, indexed per (generation x ndim)
        max_gens = max(
            1,
            int(
                np.ceil(
                    self.max_function_evaluations
                    / (self.n_individuals * self.ndim_problem)
                )
            ),
        )
        self._w = 1.0 - (np.arange(max_gens) + 1.0) / max_gens

    def iterate(self, v, x, y, p_x, p_y, n_x):
        w = self._w[min(self._n_generations, len(self._w) - 1)]
        for j in range(self.ndim_problem):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x
                n_x[i, j] = p_x[np.argmin(p_y), j]
                cog = self.rng_optimization.uniform()
                soc = self.rng_optimization.uniform()
                v[i, j] = (
                    w * v[i, j]
                    + self.cognition * cog * (p_x[i, j] - x[i, j])
                    + self.society * soc * (n_x[i, j] - x[i, j])
                )
                v[i, j] = np.clip(v[i, j], self._min_v[j], self._max_v[j])
                x[i, j] += v[i, j]
                candidate = (
                    np.copy(self.best_so_far_x)
                    if self.best_so_far_x is not None
                    else np.copy(x[i])
                )
                candidate[j] = x[i, j]
                y[i] = self._evaluate_fitness(candidate)
                if y[i] < p_y[i]:
                    p_x[i, j], p_y[i] = x[i, j], y[i]
        self._n_generations += 1
        self._warm_start = {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y, "n_x": n_x}
        return v, x, y, p_x, p_y, n_x
