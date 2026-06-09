"""IPSO: Incremental Particle Swarm Optimization."""

import numpy as np

from cat.optimizers.base import SubOptimizer


class IPSO(SubOptimizer):
    """Incremental PSO (Incremental Social Learning).

    Starts with a single particle and grows the swarm by one new particle per
    generation up to max_n_individuals. Uses a constriction factor instead of
    inertia weight, as in the original algorithm.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.n_individuals = 1
        self.max_n_individuals: int = options.get("max_n_individuals", 1000)
        self.cognition: float = options.get("cognition", 2.05)
        self.society: float = options.get("society", 2.05)
        self.constriction: float = options.get("constriction", 0.729)
        self.max_ratio_v: float = options.get("max_ratio_v", 0.5)
        self._max_v = self.max_ratio_v * (self.upper_boundary - self.lower_boundary)
        self._min_v = -self._max_v
        self._n_generations = 0

    def _init_shape(self):
        return (self.n_individuals, self.ndim_problem)

    def initialize(self, v=None, x=None, y=None, p_x=None, p_y=None):
        needs_eval = y is None
        shape = self._init_shape()
        v = np.zeros(shape) if v is None else v
        x = (
            self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary, shape
            )
            if x is None
            else x
        )
        y = np.empty(self.n_individuals) if y is None else y
        p_x = np.copy(x) if p_x is None else p_x
        p_y = np.copy(y) if p_y is None else p_y
        if needs_eval:
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return v, x, y, p_x, p_y
                y[i] = self._evaluate_fitness(x[i])
            p_y = np.copy(y)
        return v, x, y, p_x, p_y

    def iterate(self, v, x, y, p_x, p_y):
        for i in range(self.n_individuals):
            if self._check_terminations():
                return v, x, y, p_x, p_y
            cog = self.rng_optimization.uniform(size=self.ndim_problem)
            soc = self.rng_optimization.uniform(size=self.ndim_problem)
            best_idx = np.argmin(p_y)
            v[i] = self.constriction * (
                v[i]
                + self.cognition * cog * (p_x[i] - x[i])
                + self.society * soc * (p_x[best_idx] - x[i])
            )
            v[i] = np.clip(v[i], self._min_v, self._max_v)
            x[i] += v[i]
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])
            if y[i] < p_y[i]:
                p_x[i], p_y[i] = np.copy(x[i]), y[i]

        if (
            self.n_individuals < self.max_n_individuals
            and not self._check_terminations()
        ):
            xx = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary)
            model = p_x[np.argmin(p_y)]
            xx += self.rng_optimization.uniform(size=self.ndim_problem) * (model - xx)
            xx = np.clip(xx, self.lower_boundary, self.upper_boundary)
            yy = self._evaluate_fitness(xx)
            v = np.vstack([v, np.zeros((1, self.ndim_problem))])
            x = np.vstack([x, xx[np.newaxis]])
            y = np.hstack([y, yy])
            p_x = np.vstack([p_x, xx[np.newaxis]])
            p_y = np.hstack([p_y, yy])
            self.n_individuals += 1

        self._n_generations += 1
        self._warm_start = {"v": v, "x": x, "y": y, "p_x": p_x, "p_y": p_y}
        return v, x, y, p_x, p_y

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        ws = self._warm_start
        v, x, y, p_x, p_y = self.initialize(
            ws.get("v"), ws.get("x"), ws.get("y"), ws.get("p_x"), ws.get("p_y")
        )
        while not self.termination_signal:
            v, x, y, p_x, p_y = self.iterate(v, x, y, p_x, p_y)
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        self.n_individuals = 1
        if x is not None and y is not None and len(x) >= 1:
            best_idx = int(np.argmin(y))
            x1 = x[best_idx : best_idx + 1]
            y1 = y[best_idx : best_idx + 1]
            v_kwarg = kwargs.get("v")
            v1 = (
                v_kwarg[best_idx : best_idx + 1]
                if v_kwarg is not None
                else np.zeros((1, self.ndim_problem))
            )
            p_x_kwarg = kwargs.get("p_x")
            p_x1 = (
                p_x_kwarg[best_idx : best_idx + 1]
                if p_x_kwarg is not None
                else np.copy(x1)
            )
            p_y_kwarg = kwargs.get("p_y")
            p_y1 = (
                p_y_kwarg[best_idx : best_idx + 1]
                if p_y_kwarg is not None
                else np.copy(y1)
            )
            self._warm_start = {"v": v1, "x": x1, "y": y1, "p_x": p_x1, "p_y": p_y1}
        else:
            self._warm_start = {}
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)
