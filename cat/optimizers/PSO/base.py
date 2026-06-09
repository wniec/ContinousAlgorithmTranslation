"""PSO base class shared across SPSO, SPSOL, and CPSO variants.

Vendored from DynamicAlgorithmSelection2 (das/optimizers/PSO/base.py).
Warm-start state keys: v (velocities), x, y, p_x (personal best), p_y, n_x.
"""

import numpy as np

from cat.optimizers.base import SubOptimizer


class PSO(SubOptimizer):
    """PSO base with inertia-weight decay (0.9 -> 0.4 over the run)."""

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        if self.n_individuals is None:
            self.n_individuals = 20

        self.cognition: float = options.get("cognition", 2.0)
        self.society: float = options.get("society", 2.0)
        self.max_ratio_v: float = options.get("max_ratio_v", 0.2)
        self.is_bound: bool = options.get("is_bound", False)

        self._max_v = self.max_ratio_v * (self.upper_boundary - self.lower_boundary)
        self._min_v = -self._max_v

        max_gens = max(
            1, int(np.ceil(self.max_function_evaluations / self.n_individuals))
        )
        self._w = 0.9 - 0.5 * (np.arange(max_gens) + 1.0) / max_gens  # 0.9 -> 0.4
        self._n_generations = 0
        self._shape = (self.n_individuals, self.ndim_problem)

    def initialize(self, v=None, x=None, y=None, p_x=None, p_y=None, n_x=None):
        needs_eval = y is None
        v = (
            v
            if v is not None
            else self.rng_initialization.uniform(self._min_v, self._max_v, self._shape)
        )
        x = (
            x
            if x is not None
            else self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary, self._shape
            )
        )
        y = y if y is not None else np.empty(self.n_individuals)
        p_x = p_x if p_x is not None else np.copy(x)
        p_y = p_y if p_y is not None else np.copy(y)
        n_x = n_x if n_x is not None else np.copy(x)
        if needs_eval:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y, n_x
                y[i] = self._evaluate_fitness(x[i])
            p_y = np.copy(y)
        return v, x, y, p_x, p_y, n_x

    def _social_guide(self, i: int, p_x, p_y, n_x) -> np.ndarray:
        return p_x[np.argmin(p_y)]

    def iterate(self, v, x, y, p_x, p_y, n_x):
        w = self._w[min(self._n_generations, len(self._w) - 1)]
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y, n_x
            guide = self._social_guide(i, p_x, p_y, n_x)
            cog = self.rng_optimization.uniform(size=self.ndim_problem)
            soc = self.rng_optimization.uniform(size=self.ndim_problem)
            v[i] = (
                w * v[i]
                + self.cognition * cog * (p_x[i] - x[i])
                + self.society * soc * (guide - x[i])
            )
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]
            if self.is_bound:
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = np.copy(x[i]), y[i]
        self._n_generations += 1
        self._warm_start = {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y, "n_x": n_x}
        return v, x, y, p_x, p_y, n_x

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        ws = self._warm_start
        v, x, y, p_x, p_y, n_x = self.initialize(
            ws.get("v"),
            ws.get("x"),
            ws.get("y"),
            ws.get("p_x"),
            ws.get("p_y"),
            ws.get("n_x"),
        )
        while not self.termination_signal:
            v, x, y, p_x, p_y, n_x = self.iterate(v, x, y, p_x, p_y, n_x)
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        if x is None or y is None or len(x) < self.n_individuals:
            self._warm_start = {}
        else:
            idx = np.argsort(y)[: self.n_individuals]
            x_sub = x[idx]
            y_sub = y[idx]
            v = (
                kwargs["v"]
                if kwargs.get("v") is not None
                else self.rng_initialization.uniform(
                    self._min_v, self._max_v, self._shape
                )
            )
            p_x = kwargs["p_x"] if kwargs.get("p_x") is not None else np.copy(x_sub)
            p_y = kwargs["p_y"] if kwargs.get("p_y") is not None else np.copy(y_sub)
            n_x = kwargs["n_x"] if kwargs.get("n_x") is not None else np.copy(x_sub)
            if best_x is not None:
                slot = self.rng_initialization.integers(self.n_individuals)
                p_x[slot] = np.copy(best_x)
                p_y[slot] = float(best_y) if best_y is not None else float("inf")
                n_x[slot] = np.copy(best_x)
            self._warm_start = {
                "v": v,
                "x": x_sub,
                "y": y_sub,
                "p_x": p_x,
                "p_y": p_y,
                "n_x": n_x,
            }
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)
