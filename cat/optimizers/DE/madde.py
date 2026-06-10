"""MadDE: Multiple-adaptation Differential Evolution.

Implemented from the paper:
    Biswas, Saha, De, Cobb, Das, Jalaian (2021), "Improving Differential
    Evolution through Bayesian Hyperparameter Optimization", IEEE CEC 2021
    (`biswas2021.pdf`), Section II.

This is a fresh implementation from the paper's equations (Eq. 2-21 and
Algorithm 1) — not a copy of any other codebase.

MadDE combines:
  * an ensemble of three mutation strategies selected per-individual by adaptive
    probabilities ``p_m`` (Eq. 4-6, 20-21);
  * a probabilistic crossover that mixes binomial and greedy q-best binomial
    (Eq. 10-11);
  * SHADE-style success-history adaptation of F and Cr with a weighted Lehmer
    mean (Eq. 13-18);
  * an external archive of replaced parents (JADE-style) that feeds mutation;
  * linear population-size reduction (LPSR, Eq. 19).

Framework note: when ``n_individuals`` is supplied (as the translation env
always does) the population is held fixed at that size (LPSR disabled) so it
matches the other portfolio optimizers; standalone it uses the paper's
``NP_max = round(NPm·D²)`` with LPSR down to ``NP_min = 4``.
"""

from __future__ import annotations

import numpy as np

from cat.optimizers.base import SubOptimizer

ARATE = 2.30  # archive size multiplier; shared with the translator (archive capacity)
_TERMINAL = -1.0  # the bottom (terminal) value for the Cr memory (paper's ⊥)


def _weighted_lehmer(s: np.ndarray, w: np.ndarray) -> float:
    """Weighted Lehmer mean (Eq. 17): sum(w*s^2) / sum(w*s)."""
    denom = float(np.sum(w * s))
    if denom <= 1e-12:
        return float(np.sum(w * s * s) / (denom + 1e-12))
    return float(np.sum(w * s * s) / denom)


class MADDE(SubOptimizer):
    def __init__(self, problem: dict, options: dict):
        super().__init__(problem, options)
        n = self.ndim_problem

        # Hyperparameters (paper's tuned defaults).
        self.p_qbx = options.get("p_qbx", 0.01)
        self.p = options.get("p", 0.18)
        self.arate = options.get("arate", ARATE)
        self.npm = options.get("npm", 2)
        self.hm = options.get("hm", 10)
        self.f0 = options.get("f0", 0.20)
        self.cr0 = options.get("cr0", 0.20)

        # Population sizing: fixed when n_individuals given, else NPm·D² with LPSR.
        if self.n_individuals is None:
            self.np_max = max(4, int(round(self.npm * n * n)))
            self.np_max = min(self.np_max, max(4, self.max_function_evaluations))
            self.np_min = 4
        else:
            self.np_max = self.np_min = int(self.n_individuals)
        self.n_individuals = self.np_max
        self.lpsr = self.np_max != self.np_min

        self.h = max(1, int(round(self.hm * n)))  # memory size
        self._n_generations = 0

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _archive_capacity(self, np_g: int) -> int:
        return int(round(self.arate * np_g))

    def _sample_f_cr(self, m_f, m_cr):
        k = int(self.rng_optimization.integers(self.h))
        if m_cr[k] == _TERMINAL:
            cr = 0.0
        else:
            cr = float(np.clip(self.rng_optimization.normal(m_cr[k], 0.1), 0.0, 1.0))
        f = -1.0
        while f <= 0.0:
            f = m_f[k] + 0.1 * self.rng_optimization.standard_cauchy()
            if f > 1.0:
                f = 1.0
                break
        return float(f), cr

    def _bound_repair(self, v, x_target):
        lb, ub = self.lower_boundary, self.upper_boundary
        below = v < lb
        above = v > ub
        v = np.where(below, 0.5 * (x_target + lb), v)
        v = np.where(above, 0.5 * (x_target + ub), v)
        return v

    def _pick(self, n_pool: int, exclude: set[int], size: int) -> list[int]:
        """Sample ``size`` distinct indices from [0, n_pool) avoiding ``exclude``."""
        out: list[int] = []
        guard = 0
        while len(out) < size and guard < 100:
            j = int(self.rng_optimization.integers(n_pool))
            if j not in exclude and j not in out:
                out.append(j)
            guard += 1
        while len(out) < size:  # tiny pool fallback: allow repeats
            out.append(int(self.rng_optimization.integers(n_pool)))
        return out

    # ------------------------------------------------------------------ #
    # init / iterate / optimize                                            #
    # ------------------------------------------------------------------ #

    def initialize(self, ws: dict):
        n = self.ndim_problem
        x = ws.get("x")
        y = ws.get("y")
        if x is None or y is None or len(x) < self.n_individuals:
            x = self.rng_initialization.uniform(
                self.initial_lower_boundary,
                self.initial_upper_boundary,
                (self.n_individuals, n),
            )
            y = np.empty(self.n_individuals)
            for i in range(self.n_individuals):
                if self._check_terminations():
                    return x, y, *self._aux_defaults(ws)
                y[i] = self._evaluate_fitness(x[i])
        else:
            idx = np.argsort(y)[: self.n_individuals]
            x, y = np.array(x[idx], dtype=float), np.array(y[idx], dtype=float)
        return x, y, *self._aux_defaults(ws)

    def _aux_defaults(self, ws: dict):
        n = self.ndim_problem
        archive_x = ws.get("archive")
        if archive_x is None:
            archive_x = np.empty((0, n))
            archive_y = np.empty(0)
        else:
            archive_x = np.array(archive_x, dtype=float).reshape(-1, n)
            ay = ws.get("archive_y")
            archive_y = (
                np.array(ay, dtype=float).reshape(-1)
                if ay is not None and len(ay) == len(archive_x)
                else np.full(len(archive_x), np.inf)  # translated archive: no fitness
            )
        m_f = (
            np.array(ws.get("m_f"), dtype=float)
            if ws.get("m_f") is not None
            else np.full(self.h, self.f0)
        )
        m_cr = (
            np.array(ws.get("m_cr"), dtype=float)
            if ws.get("m_cr") is not None
            else np.full(self.h, self.cr0)
        )
        k_mem = int(ws.get("k_mem", 0))
        p_m = (
            np.array(ws.get("p_m"), dtype=float)
            if ws.get("p_m") is not None
            else np.full(3, 1.0 / 3.0)
        )
        return archive_x, archive_y, m_f, m_cr, k_mem, p_m

    def iterate(self, x, y, archive_x, archive_y, m_f, m_cr, k_mem, p_m):
        n = self.ndim_problem
        np_g = len(x)
        fes_ratio = self.n_function_evaluations / max(self.max_function_evaluations, 1)
        q = 2.0 * self.p - self.p * fes_ratio
        fa = 0.5 + 0.5 * fes_ratio

        pa_x = np.vstack([x, archive_x]) if len(archive_x) else x
        order = np.argsort(y)
        p_num = max(2, int(round(self.p * np_g)))
        q_num = max(2, int(round(q * np_g)))
        pbest_pool = order[:p_num]
        qbest_pool = order[:q_num]  # top-q% of the population (Eq. 6 / qBX)

        trial = np.empty_like(x)
        trial_y = np.full(np_g, np.nan)
        used_f = np.zeros(np_g)
        used_cr = np.zeros(np_g)
        used_m = np.zeros(np_g, dtype=int)
        evaluated = np.zeros(np_g, dtype=bool)

        for i in range(np_g):
            if self._check_terminations():
                break
            f_i, cr_i = self._sample_f_cr(m_f, m_cr)
            m = int(self.rng_optimization.choice(3, p=p_m))

            if m == 0:  # DE/current-to-pbest/1 + archive (Eq. 4)
                pbest = x[int(self.rng_optimization.choice(pbest_pool))]
                r1 = self._pick(np_g, {i}, 1)[0]
                r3 = self._pick(len(pa_x), {i, r1}, 1)[0]
                v = x[i] + f_i * (pbest - x[i] + x[r1] - pa_x[r3])
            elif m == 1:  # DE/current-to-rand/1 + archive (Eq. 5)
                r1 = self._pick(np_g, {i}, 1)[0]
                r3 = self._pick(len(pa_x), {i, r1}, 1)[0]
                v = x[i] + f_i * (x[r1] - pa_x[r3])
            else:  # DE/weighted-rand-to-qbest/1 (Eq. 6)
                qbest = x[int(self.rng_optimization.choice(qbest_pool))]
                r1, r2 = self._pick(np_g, {i}, 2)
                v = f_i * x[r1] + f_i * fa * (qbest - x[r2])

            v = self._bound_repair(v, x[i])

            # Crossover (Eq. 10-11): q-best binomial with prob p_qbx, else binomial.
            if self.rng_optimization.random() < self.p_qbx:
                base = x[int(self.rng_optimization.choice(qbest_pool))]
            else:
                base = x[i]
            u = np.array(base, dtype=float)
            j_rand = int(self.rng_optimization.integers(n))
            cross = self.rng_optimization.random(n) <= cr_i
            cross[j_rand] = True
            u[cross] = v[cross]

            trial[i] = u
            used_f[i], used_cr[i], used_m[i] = f_i, cr_i, m
            trial_y[i] = self._evaluate_fitness(u)
            evaluated[i] = True

        # Selection (Eq. 12) + success bookkeeping.
        s_f, s_cr, s_df = [], [], []
        strat_imp = np.zeros(3)
        strat_cnt = np.zeros(3)
        new_x, new_y = x.copy(), y.copy()
        for i in range(np_g):
            if not evaluated[i]:
                continue
            df = y[i] - trial_y[i]
            strat_cnt[used_m[i]] += 1
            if df > 0:
                strat_imp[used_m[i]] += df
            if trial_y[i] < y[i]:  # strict success -> record F, Cr, improvement
                s_f.append(used_f[i])
                s_cr.append(used_cr[i])
                s_df.append(abs(df))
            if trial_y[i] <= y[i]:  # trial replaces parent; old parent -> archive
                archive_x, archive_y = self._archive_add(
                    archive_x, archive_y, x[i], y[i], np_g
                )
                new_x[i], new_y[i] = trial[i], trial_y[i]

        m_f, m_cr, k_mem = self._update_memory(s_f, s_cr, s_df, m_f, m_cr, k_mem)
        p_m = self._update_pm(strat_imp, strat_cnt)

        # Linear population-size reduction (Eq. 19); inactive when fixed-pop.
        if self.lpsr:
            np_next = int(round(self.np_max - (self.np_max - self.np_min) * fes_ratio))
            np_next = max(self.np_min, min(np_g, np_next))
            if np_next < np_g:
                keep = np.argsort(new_y)[:np_next]
                new_x, new_y = new_x[keep], new_y[keep]
            archive_x, archive_y = self._archive_trim(archive_x, archive_y, np_next)

        self._n_generations += 1
        self._warm_start = {
            "x": new_x,
            "y": new_y,
            "archive": archive_x,
            "archive_y": archive_y,
            "m_f": m_f,
            "m_cr": m_cr,
            "k_mem": k_mem,
            "p_m": p_m,
        }
        return new_x, new_y, archive_x, archive_y, m_f, m_cr, k_mem, p_m

    def _archive_add(self, archive_x, archive_y, parent_x, parent_y, np_g):
        archive_x = np.vstack([archive_x, parent_x[None]])
        archive_y = np.concatenate([archive_y, [parent_y]])
        return self._archive_trim(archive_x, archive_y, np_g)

    def _archive_trim(self, archive_x, archive_y, np_g):
        cap = self._archive_capacity(np_g)
        if len(archive_x) > cap:
            keep = self.rng_optimization.choice(len(archive_x), cap, replace=False)
            archive_x, archive_y = archive_x[keep], archive_y[keep]
        return archive_x, archive_y

    def _update_memory(self, s_f, s_cr, s_df, m_f, m_cr, k_mem):
        if len(s_f) == 0:  # no success: leave memory unchanged (standard SHADE)
            return m_f, m_cr, k_mem
        s_f = np.array(s_f)
        s_cr = np.array(s_cr)
        w = np.array(s_df)
        w = w / (w.sum() + 1e-12)
        m_f = m_f.copy()
        m_cr = m_cr.copy()
        m_f[k_mem] = _weighted_lehmer(s_f, w)
        if m_cr[k_mem] == _TERMINAL or s_cr.max() == 0.0:
            m_cr[k_mem] = _TERMINAL
        else:
            m_cr[k_mem] = _weighted_lehmer(s_cr, w)
        k_mem = (k_mem + 1) % self.h
        return m_f, m_cr, k_mem

    def _update_pm(self, strat_imp, strat_cnt):
        delta = np.where(strat_cnt > 0, strat_imp / np.maximum(strat_cnt, 1), 0.0)
        total = delta.sum()
        if total <= 1e-12:
            return np.full(3, 1.0 / 3.0)
        p_m = delta / total
        p_m = np.clip(p_m, 0.1, 0.9)
        return p_m / p_m.sum()

    def optimize(self, fitness_function=None, args=None):
        fitness = super().optimize(fitness_function)
        state = self.initialize(self._warm_start)
        while not self.termination_signal:
            state = self.iterate(*state)
        return self._collect(fitness)

    def _collect(self, fitness):
        result = super()._collect(fitness)
        result["_n_generations"] = self._n_generations
        return result

    # ------------------------------------------------------------------ #
    # warm-start contract                                                  #
    # ------------------------------------------------------------------ #

    def set_data(self, x=None, y=None, best_x=None, best_y=None, **kwargs):
        ws: dict = {}
        if x is not None and y is not None and len(x) >= self.n_individuals:
            ws["x"] = np.asarray(x, dtype=float)
            ws["y"] = np.asarray(y, dtype=float)
        for key in ("archive", "archive_y", "m_f", "m_cr", "p_m"):
            if kwargs.get(key) is not None:
                ws[key] = np.asarray(kwargs[key], dtype=float)
        if kwargs.get("k_mem") is not None:
            ws["k_mem"] = int(kwargs["k_mem"])
        self._warm_start = ws
        if best_x is not None:
            self.best_so_far_x = np.copy(best_x)
        if best_y is not None:
            self.best_so_far_y = float(best_y)

    def get_data(self) -> dict:
        return dict(self._warm_start)
