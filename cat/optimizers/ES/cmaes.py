"""CMAES: Covariance Matrix Adaptation Evolution Strategy.

Vendored from DynamicAlgorithmSelection2 (das/optimizers/ES/cmaes.py).
"""

import numpy as np

from .base import ES


class CMAES(ES):
    """CMA-ES with full covariance matrix adaptation (Hansen, 2001).

    Warm-start state keys
    ---------------------
    mean, x, p_c, p_s, cm, e_ve, e_va, d, y

    The step-size ``sigma`` is carried as an attribute (not in the warm-start
    dict); ``cm`` is the covariance matrix, ``p_c`` / ``p_s`` the evolution
    paths.
    """

    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options | {"sigma": options.get("sigma", 1.5)})
        n, mu = self.ndim_problem, self.n_parents

        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._w = raw_w / raw_w.sum()
        self._mu_eff = 1.0 / (self._w**2).sum()

        self.c_s = (self._mu_eff + 2.0) / (n + self._mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + 2.0 * max(0.0, np.sqrt((self._mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.c_s
        )
        self._chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        self.c_c = (4.0 + self._mu_eff / n) / (n + 4.0 + 2.0 * self._mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self.c_w = min(
            1.0 - self.c_1,
            2.0
            * (self._mu_eff - 2.0 + 1.0 / self._mu_eff)
            / ((n + 2.0) ** 2 + self._mu_eff),
        )

    def initialize(
        self,
        mean=None,
        x=None,
        p_c=None,
        p_s=None,
        cm=None,
        e_ve=None,
        e_va=None,
        d=None,
        y=None,
    ):
        n, lam = self.ndim_problem, self.n_individuals
        if mean is None:
            mean = self.rng_initialization.uniform(
                self.initial_lower_boundary, self.initial_upper_boundary
            )
        if x is None:
            x = np.empty((lam, n))
        if p_c is None:
            p_c = np.zeros(n)
        if p_s is None:
            p_s = np.zeros(n)
        if cm is None:
            cm = np.eye(n)
        if e_ve is None:
            e_ve = np.eye(n)
        if e_va is None:
            e_va = np.ones(n)
        if d is None:
            d = np.ones(n)
        if y is None:
            y = np.empty(lam)
            for i in range(lam):
                if self._check_terminations():
                    return mean, x, p_c, p_s, cm, e_ve, e_va, d, y
                x[i] = mean + self.sigma * e_ve @ (
                    d * self.rng_optimization.standard_normal(n)
                )
                x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
                y[i] = self._evaluate_fitness(x[i])
        return mean, x, p_c, p_s, cm, e_ve, e_va, d, y

    def iterate(self, mean, x, p_c, p_s, cm, e_ve, e_va, d, y):
        n, lam, mu = self.ndim_problem, self.n_individuals, self.n_parents

        z = self.rng_optimization.standard_normal((lam, n))
        for i in range(lam):
            if self._check_terminations():
                return mean, x, p_c, p_s, cm, e_ve, e_va, d, y
            x[i] = mean + self.sigma * e_ve @ (d * z[i])
            x[i] = np.clip(x[i], self.lower_boundary, self.upper_boundary)
            y[i] = self._evaluate_fitness(x[i])

        order = np.argsort(y)[:mu]
        x_best = x[order]
        z_best = z[order]

        old_mean = mean.copy()
        mean = self._w @ x_best

        p_s = (1.0 - self.c_s) * p_s + np.sqrt(
            self.c_s * (2.0 - self.c_s) * self._mu_eff
        ) * (e_ve @ (d * (self._w @ z_best)))
        h_s = (
            np.linalg.norm(p_s)
            / np.sqrt(1.0 - (1.0 - self.c_s) ** (2.0 * (self._n_generations + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * self._chi_n
        )
        p_c = (1.0 - self.c_c) * p_c + h_s * np.sqrt(
            self.c_c * (2.0 - self.c_c) * self._mu_eff
        ) * (mean - old_mean) / self.sigma

        delta_h = (1.0 - h_s) * self.c_c * (2.0 - self.c_c)
        cm = (
            (1.0 - self.c_1 - self.c_w) * cm
            + self.c_1 * (np.outer(p_c, p_c) + delta_h * cm)
            + self.c_w
            * sum(
                self._w[i] * np.outer(x_best[i] - old_mean, x_best[i] - old_mean)
                for i in range(mu)
            )
            / self.sigma**2
        )

        self.sigma *= np.exp(
            self.c_s / self.d_sigma * (np.linalg.norm(p_s) / self._chi_n - 1.0)
        )
        self.sigma = np.clip(self.sigma, 1e-10, 1e10)

        cm = (cm + cm.T) / 2.0
        e_va2, e_ve = np.linalg.eigh(cm)
        e_va = np.sqrt(np.maximum(e_va2, 1e-20))

        self._n_generations += 1
        self._warm_start = {
            "mean": mean,
            "x": x,
            "p_c": p_c,
            "p_s": p_s,
            "cm": cm,
            "e_ve": e_ve,
            "e_va": e_va,
            "d": e_va,
            "y": y,
        }
        return mean, x, p_c, p_s, cm, e_ve, e_va, e_va, y  # d = e_va

    def optimize(self, fitness_function=None, args=None):
        fitness = super(ES, self).optimize(fitness_function)
        ws = self._warm_start
        mean, x, p_c, p_s, cm, e_ve, e_va, d, y = self.initialize(
            ws.get("mean"),
            ws.get("x"),
            ws.get("p_c"),
            ws.get("p_s"),
            ws.get("cm"),
            ws.get("e_ve"),
            ws.get("e_va"),
            ws.get("d"),
            ws.get("y"),
        )
        while not self.termination_signal:
            mean, x, p_c, p_s, cm, e_ve, e_va, d, y = self.iterate(
                mean, x, p_c, p_s, cm, e_ve, e_va, d, y
            )
        return self._collect(fitness)

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        n = self.ndim_problem
        _shapes = {
            "mean": (n,),
            "p_c": (n,),
            "p_s": (n,),
            "cm": (n, n),
            "e_ve": (n, n),
            "e_va": (n,),
            "d": (n,),
        }

        def _valid(key, val):
            return val is not None and np.asarray(val).shape == _shapes[key]

        self._warm_start = {
            k: kwargs.get(k) if _valid(k, kwargs.get(k)) else None for k in _shapes
        }
        # Normalise x/y to exactly n_individuals rows so iterate() z-indexing is safe.
        # Too few points -> start fresh; too many -> keep best n_individuals.
        if x is not None and y is not None and len(x) >= self.n_individuals:
            idx = np.argsort(y)[: self.n_individuals]
            self._warm_start["x"] = x[idx]
            self._warm_start["y"] = y[idx]
        else:
            self._warm_start["x"] = None
            self._warm_start["y"] = None
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        return dict(self._warm_start)
