"""ES base class shared across all Evolution Strategy variants."""

import numpy as np

from cat.optimizers.base import SubOptimizer


class ES(SubOptimizer):
    """ES base: handles mu/lambda selection and basic sigma."""

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        self.sigma: float = options.get("sigma", 0.3)
        if self.n_individuals is None:
            self.n_individuals = 4 + int(3 * np.log(self.ndim_problem))
        self.n_parents: int = options.get("n_parents", self.n_individuals // 2)
        self._n_generations = 0

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result
