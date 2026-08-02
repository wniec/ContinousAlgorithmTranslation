"""BOBYQA: a model-based trust-region derivative-free optimizer.

A compact, from-scratch implementation of the *method* behind Powell's BOBYQA
(Powell, 2009, "The BOBYQA algorithm for bound constrained optimization without
derivatives") — not a bit-exact port of his Fortran. Like the vendored CMA-ES
here, it is a clean ~200-line implementation of the essential algorithm so it
can serve as a translation partner in this project.

How it works
------------
A quadratic surrogate centred on the current best point ``c``::

    m(c + s) = f(c) + g . s + 1/2 s . H s

is fit by interpolation to a set of ``M`` sample points, minimized inside a
trust region ``||s|| <= delta`` (dogleg), and the region is grown/shrunk from
the ratio of actual to predicted reduction. The model is *persistent*: each
step updates ``(g, H)`` by the **minimum-norm change** that re-interpolates the
sample set (Powell's least-Frobenius-norm principle, realized here as a
min-norm least-squares fit of the model *update*). So curvature that the sample
set under-determines is inherited rather than discarded — which is exactly what
lets a *translated* Hessian/gradient warm-start actually steer the early steps.

Warm-start state
----------------
Shared:   ``x`` — the interpolation points ``(M, D)``; ``y`` — their values.
Specific: ``grad`` ``(D,)``, ``hessian`` ``(D, D)``, ``radius`` (scalar delta).

The model is centred on the best interpolation point, so its centre rides along
with ``best_x`` and no separate centre field is translated (mirroring how
CMA-ES's mean rides with the population). Because the surrogate is *refit* from
the carried population every step, a translated Hessian is a warm-start prior,
not a hard constraint: a lossy hand-off (population only) still yields a valid,
if colder, model.
"""

from __future__ import annotations

import numpy as np

from cat.optimizers.base import SubOptimizer


class BOBYQA(SubOptimizer):
    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        n = self.ndim_problem
        # Interpolation set size: Powell's default 2n+1, or the configured pop
        # size (the translation env fixes this per algorithm). Never below n+2
        # so a diagonal-plus-gradient model is at least nominally identifiable.
        if self.n_individuals is None:
            self.n_individuals = 2 * n + 1
        self.n_individuals = max(int(self.n_individuals), min(2 * n + 1, n + 2))

        span = float(np.mean(self.upper_boundary - self.lower_boundary))
        self.rho_beg: float = options.get("rho_beg", 0.2 * span)  # initial delta
        self.rho_end: float = options.get("rho_end", 1e-6 * span)  # min resolution
        self._max_delta: float = float(np.max(self.upper_boundary - self.lower_boundary))
        self.delta: float = self.rho_beg

        # (i, j) upper-triangle index pairs for the off-diagonal Hessian entries.
        self._pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Persistent model (centred on self._center), initialised at setup.
        self._X: np.ndarray | None = None
        self._F: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._ibest: int = 0
        self._g = np.zeros(n)
        self._H = np.zeros((n, n))
        self._n_generations = 0

    # ------------------------------------------------------------------ #
    # quadratic model fitting                                             #
    # ------------------------------------------------------------------ #

    def _design(self, D: np.ndarray) -> np.ndarray:
        """Feature matrix of the quadratic basis for displacements ``D`` (M, n).

        Column order: [1, d_i, 0.5 d_i^2, d_i d_j (i<j)] — so a coefficient
        vector ``w`` decodes as g_i = w_lin[i], H_ii = w_diag[i], H_ij = w_off.
        """
        n, m = self.ndim_problem, D.shape[0]
        cols = [np.ones(m)]
        cols.extend(D[:, i] for i in range(n))
        cols.extend(0.5 * D[:, i] ** 2 for i in range(n))
        cols.extend(D[:, i] * D[:, j] for (i, j) in self._pairs)
        return np.stack(cols, axis=1)

    def _model_values(self, D: np.ndarray, g: np.ndarray, H: np.ndarray) -> np.ndarray:
        """m(c + d) - f(c) for each displacement row of ``D``."""
        return D @ g + 0.5 * np.einsum("md,de,me->m", D, H, D)

    def _refit(self):
        """Update (g, H) by the minimum-norm change that re-interpolates the set.

        Fitting the *residual* r = F - m_prior(D) with min-norm least squares
        (``lstsq``) yields the smallest model change (in gradient/Hessian
        coefficients) consistent with the samples — inheriting under-determined
        curvature from the current model, which for a warm start is the
        translated Hessian.
        """
        n = self.ndim_problem
        # Fit only against points with a finite value (a budget-truncated cold
        # start can leave some interpolation slots unevaluated); a lone finite
        # point leaves the model unchanged.
        finite = np.isfinite(self._F)
        if finite.sum() < 2:
            return
        D = (self._X - self._center)[finite]
        A = self._design(D)
        prior = self._F[self._ibest] + self._model_values(D, self._g, self._H)
        r = self._F[finite] - prior
        dw, *_ = np.linalg.lstsq(A, r, rcond=None)

        dg = dw[1 : 1 + n]
        ddiag = dw[1 + n : 1 + 2 * n]
        doff = dw[1 + 2 * n :]
        dH = np.diag(ddiag).astype(float)
        for k, (i, j) in enumerate(self._pairs):
            dH[i, j] = dH[j, i] = doff[k]
        self._g = self._g + dg
        self._H = 0.5 * (self._H + dH + (self._H + dH).T)

    # ------------------------------------------------------------------ #
    # trust-region subproblem                                            #
    # ------------------------------------------------------------------ #

    def _tr_step(self, g: np.ndarray, H: np.ndarray, delta: float) -> np.ndarray:
        """Approximately minimize g.s + 1/2 s.H s over ||s|| <= delta (dogleg)."""
        n = len(g)
        gnorm = float(np.linalg.norm(g))
        evals = np.linalg.eigvalsh(0.5 * (H + H.T))
        if gnorm < 1e-16:
            # No first-order info: exploit negative curvature if present.
            if evals[0] < -1e-12:
                _, vecs = np.linalg.eigh(0.5 * (H + H.T))
                return delta * vecs[:, 0]
            return np.zeros(n)

        if evals[0] > 1e-12:  # positive definite -> Newton / dogleg
            p_newton = -np.linalg.solve(H, g)
            if np.linalg.norm(p_newton) <= delta:
                return p_newton
            return self._dogleg(g, H, p_newton, delta, gnorm)
        # Indefinite or negative curvature: steepest-descent Cauchy point.
        return self._cauchy(g, H, delta, gnorm)

    @staticmethod
    def _cauchy(g, H, delta, gnorm) -> np.ndarray:
        gHg = float(g @ (H @ g))
        tau = 1.0 if gHg <= 0.0 else min(1.0, gnorm**3 / (delta * gHg))
        return -tau * (delta / gnorm) * g

    def _dogleg(self, g, H, p_newton, delta, gnorm) -> np.ndarray:
        gHg = float(g @ (H @ g))
        p_u = -(float(g @ g) / gHg) * g  # unconstrained min along -g
        if np.linalg.norm(p_u) >= delta:
            return -(delta / gnorm) * g  # Cauchy point on the boundary
        d = p_newton - p_u
        a = float(d @ d)
        b = 2.0 * float(p_u @ d)
        c = float(p_u @ p_u) - delta**2
        disc = max(b * b - 4.0 * a * c, 0.0)
        tau = (-b + np.sqrt(disc)) / (2.0 * a) if a > 1e-16 else 0.0
        return p_u + tau * d

    # ------------------------------------------------------------------ #
    # main loop                                                          #
    # ------------------------------------------------------------------ #

    def iterate(self):
        lb, ub = self.lower_boundary, self.upper_boundary
        c = self._center
        f_c = self._F[self._ibest]

        s = self._tr_step(self._g, self._H, self.delta)
        snorm = float(np.linalg.norm(s))
        if snorm < 1e-12 * max(1.0, float(np.linalg.norm(c))):
            # Flat model / converged inside delta: probe to refresh poisedness.
            s = self.rng_optimization.standard_normal(self.ndim_problem)
            s *= self.delta / max(np.linalg.norm(s), 1e-12)
            snorm = float(np.linalg.norm(s))

        x_new = np.clip(c + s, lb, ub)
        f_new = self._evaluate_fitness(x_new)
        if self._check_terminations():
            self._store()
            return

        pred = -self._model_values((x_new - c)[None], self._g, self._H)[0]
        actual = f_c - f_new
        rho = actual / pred if pred > 1e-16 else (1.0 if actual > 0.0 else -1.0)

        self._insert(x_new, f_new)

        if rho < 0.25:
            self.delta = max(0.5 * self.delta, self.rho_end)
        elif rho > 0.75 and snorm > 0.9 * self.delta:
            self.delta = min(2.0 * self.delta, self._max_delta)

        self._refit()
        self._n_generations += 1
        self._store()

    def _insert(self, x_new: np.ndarray, f_new: float):
        """Drop the interpolation point farthest from the centre (never the
        best) and slot the new one in, then recentre on the new best. Keeps the
        set size — and thus the shared population size — constant."""
        dist = np.linalg.norm(self._X - self._center, axis=1)
        dist[self._ibest] = -1.0  # protect the current best point
        j = int(np.argmax(dist))
        old_center = self._center
        self._X[j] = x_new
        self._F[j] = f_new
        self._ibest = int(np.argmin(self._F))
        self._center = self._X[self._ibest].copy()
        # Re-expand the persistent gradient about the (possibly moved) centre:
        # for a quadratic, g(c_new) = g(c_old) + H (c_new - c_old).
        shift = self._center - old_center
        if np.any(shift):
            self._g = self._g + self._H @ shift

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        self._setup()
        while not self.termination_signal:
            self.iterate()
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    # ------------------------------------------------------------------ #
    # initialization / warm start                                        #
    # ------------------------------------------------------------------ #

    def _initial_points(self, center: np.ndarray) -> np.ndarray:
        """Cold interpolation set: centre + coordinate perturbations (+/- delta),
        with any extra slots filled by uniform draws within the trust region."""
        n, m = self.ndim_problem, self.n_individuals
        pts = np.tile(center, (m, 1)).astype(float)
        for k in range(1, m):
            if k - 1 < 2 * n:
                j = (k - 1) % n
                sign = 1.0 if (k - 1) < n else -1.0
                pts[k, j] += sign * self.delta
            else:
                pts[k] += self.rng_initialization.uniform(-1.0, 1.0, n) * self.delta
        return np.clip(pts, self.lower_boundary, self.upper_boundary)

    def _cold_start(self):
        center = self.rng_initialization.uniform(
            self.initial_lower_boundary, self.initial_upper_boundary
        )
        X = self._initial_points(center)
        F = np.full(len(X), np.inf)
        for k in range(len(X)):
            if self._check_terminations():
                break
            F[k] = self._evaluate_fitness(X[k])
        self._X, self._F = X, F
        self._ibest = int(np.argmin(F))
        self._center = X[self._ibest].copy()
        self._g = np.zeros(self.ndim_problem)
        self._H = np.zeros((self.ndim_problem, self.ndim_problem))

    def _setup(self):
        ws = self._warm_start or {}
        n = self.ndim_problem
        x, y = ws.get("x"), ws.get("y")
        if x is not None and y is not None and len(x) >= 2:
            X = np.asarray(x, dtype=float).reshape(-1, n)
            F = np.asarray(y, dtype=float).reshape(-1)
            self.n_individuals = len(X)
            self._X, self._F = X.copy(), F.copy()
            finite = np.isfinite(F)
            self._ibest = int(np.argmin(np.where(finite, F, np.inf)))
            self._center = X[self._ibest].copy()

            g = ws.get("grad")
            H = ws.get("hessian")
            self._g = (
                np.asarray(g, float).reshape(n)
                if g is not None and np.asarray(g).shape == (n,)
                else np.zeros(n)
            )
            self._H = (
                0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
                if H is not None and np.asarray(H).shape == (n, n)
                else np.zeros((n, n))
            )
            r = ws.get("radius")
            if r is not None and np.isfinite(r) and float(r) > 0.0:
                self.delta = float(np.clip(r, self.rho_end, self._max_delta))
            else:
                spread = float(np.mean(np.std(X, axis=0)))
                self.delta = float(np.clip(spread, self.rho_end, self._max_delta)) or self.rho_beg
        else:
            self._cold_start()

        self._refit()  # anchor the model on the actual sample values
        self._store()

    def _store(self):
        self._warm_start = {
            "x": self._X,
            "y": self._F,
            "grad": self._g,
            "hessian": self._H,
            "radius": float(self.delta),
        }

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        n = self.ndim_problem
        ws: dict = {}
        if x is not None and y is not None and len(x) >= 2:
            ws["x"] = np.asarray(x, dtype=float).reshape(-1, n)
            ws["y"] = np.asarray(y, dtype=float).reshape(-1)

        def _valid(val, shape) -> bool:
            return val is not None and np.asarray(val).shape == shape

        if _valid(kwargs.get("grad"), (n,)):
            ws["grad"] = np.asarray(kwargs["grad"], dtype=float)
        if _valid(kwargs.get("hessian"), (n, n)):
            ws["hessian"] = np.asarray(kwargs["hessian"], dtype=float)
        r = kwargs.get("radius")
        if r is not None and np.isfinite(r) and float(r) > 0.0:
            ws["radius"] = float(r)

        self._warm_start = ws
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        return dict(self._warm_start)
