"""Gymnasium environment: a BBOB run with learned optimizer hand-offs.

One episode optimizes a single BBOB problem (of a fixed dimension) over
``n_switches`` segments whose boundaries are sampled log-uniformly over the FE
budget. The acting optimizer alternates A, B, A, …; at each step the agent
observes the *source* optimizer's internal state and must produce the *target*
optimizer's internal state (the translation), which warm-starts the target for
the next segment.

* **observation** — a dict with the source optimizer's augmented native state,
  the source/target algorithm names, and a small context vector (fraction of
  budget used, normalized best-so-far). The custom PPO policy turns the native
  state into a ``CanonicalState`` and encodes it directly; nothing is flattened.
* **action** — the target's full native warm-start dict (shared population +
  decoded specific fields), assembled by the policy from its decoded state.
* **reward** — clipped best-so-far improvement over the segment, scaled by an
  agent-independent random-probe range fixed at reset (prevents reward hacking,
  mirroring DAS's ``DASEnv``).

The observation/action spaces are declared for completeness but the env is meant
to be driven by the custom loop in ``cat.rl.ppo`` (which reads the structured
objects), not by a generic flat-vector agent.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cat.data.collector import _augment, _has_full_state
from cat.optimizers.base import sample_switch_points
from cat.optimizers.portfolio import PORTFOLIO
from cat.state.canonical import warm_start_optimizer


def scaled_improvement(prev_best: float, new_best: float, rng_range: float) -> float:
    """Best-so-far improvement over a segment, scaled to [0, 1]."""
    improvement = max(0.0, prev_best - new_best)
    return float(np.clip(improvement / max(rng_range, 1e-12), 0.0, 1.0))


class TranslationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        algo_a: str,
        algo_b: str,
        problem_ids: list[str],
        suite,
        *,
        fe_multiplier: int = 2000,
        n_switches: int = 6,
        n_individuals: int | None = 12,
        reward_mode: str = "absolute",
        seed: int = 0,
    ):
        super().__init__()
        if not problem_ids:
            raise ValueError("problem_ids is empty")
        if reward_mode not in ("absolute", "relative"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.algo_a = algo_a
        self.algo_b = algo_b
        self.problem_ids = problem_ids
        self.suite = suite
        self.fe_multiplier = fe_multiplier
        self.n_switches = n_switches
        self.n_individuals = n_individuals
        self.reward_mode = reward_mode
        self._seed = seed
        self.dim = self._parse_dim(problem_ids[0])

        # Declared for completeness; the custom PPO loop reads structured objects.
        self.observation_space = spaces.Dict(
            {"context": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)}
        )
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        self._problem_idx = 0
        self._reset_episode_state()

    @staticmethod
    def _parse_dim(pid: str) -> int:
        return int(pid.split("_d")[-1])

    def _reset_episode_state(self):
        self._problem = None
        self._cfg = None
        self._max_fe = 0
        self._n_fe = 0
        self._best_y = float("inf")
        self._best_x = None
        self._range = 1.0
        self._checkpoints = None
        self._step_idx = 0
        self._src_native = None
        self._src_algo = None
        self._tgt_algo = None

    # ------------------------------------------------------------------ #

    def _make_optimizer(self, algo: str, target_fe: int, seed_rng: int):
        options = {
            "max_function_evaluations": self._max_fe,
            "target_fe": int(target_fe),
            "best_so_far_y": self._best_y
            if self._best_y < float("inf")
            else float("inf"),
            "seed_rng": seed_rng,
            "verbose": False,
        }
        if self.n_individuals is not None:
            options["n_individuals"] = self.n_individuals
        opt = PORTFOLIO[algo](self._cfg, options)
        opt.n_function_evaluations = self._n_fe
        return opt

    def _observation(self) -> dict:
        frac = self._n_fe / max(self._max_fe, 1)
        best = self._best_y if np.isfinite(self._best_y) else 0.0
        return {
            "native": self._src_native,
            "source_algo": self._src_algo,
            "target_algo": self._tgt_algo,
            "context": np.array(
                [frac, np.tanh(best / max(self._range, 1e-9))], dtype=np.float32
            ),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pid = self.problem_ids[self._problem_idx % len(self.problem_ids)]
        self._reset_episode_state()
        self._problem = self.suite.get_problem(pid)
        dim = self._problem.dimension
        self._cfg = {
            "fitness_function": self._problem,
            "ndim_problem": dim,
            "lower_boundary": self._problem.lower_bounds,
            "upper_boundary": self._problem.upper_bounds,
        }
        self._max_fe = self.fe_multiplier * dim
        pop = self.n_individuals if self.n_individuals is not None else 20

        base = (self._seed * 1_000_003 + self._problem_idx) % (2**31)
        sw_rng = np.random.default_rng(base)
        self._problem_idx += 1

        # Agent-independent probe: fixes best/scale before the agent acts.
        n_probe = min(2 * dim + 8, max(self._max_fe // (self.n_switches + 3), dim + 1))
        # The warmup segment must leave room (after the probe) for the source
        # optimizer to complete at least a couple of generations, else it would
        # return an empty state. Bound the earliest switch boundary accordingly.
        fe_min = min(n_probe + 2 * pop, max(self._max_fe - pop, n_probe + pop))
        # n_switches + 1 boundaries: [warmup_end, seg_1_end, ..., seg_n_end].
        self._checkpoints = sample_switch_points(
            self.n_switches + 1, self._max_fe, pop, sw_rng, fe_min=fe_min
        )

        lb, ub = self._problem.lower_bounds, self._problem.upper_bounds
        x_probe = sw_rng.uniform(lb, ub, size=(n_probe, dim))
        y_probe = np.array([self._problem(x) for x in x_probe], dtype=float)
        i_best = int(np.argmin(y_probe))
        self._best_y = float(y_probe[i_best])
        self._best_x = x_probe[i_best]
        self._range = max(float(np.median(y_probe)) - self._best_y, 1e-5)
        self._n_fe = n_probe

        # Warm up the first algorithm (cold) to build a genuine source state.
        warmup = self._make_optimizer(self.algo_a, self._checkpoints[0], base + 1)
        warmup.set_data(best_x=self._best_x, best_y=self._best_y)
        result = warmup.optimize()
        self._update_best(result)
        self._n_fe = result.get("n_function_evaluations", self._n_fe)

        self._src_native = _augment(warmup, dim)
        self._src_algo = self.algo_a
        self._tgt_algo = self.algo_b
        self._step_idx = 0
        return self._observation(), {"problem_id": pid, "dimension": dim}

    def step(self, action_native: dict):
        """``action_native`` is the target's full warm-start dict (shared + specific)."""
        target = self._tgt_algo
        prev_best = self._best_y
        target_fe = int(self._checkpoints[self._step_idx + 1])
        seg_seed = self._seed + self._step_idx + 1

        opt = self._make_optimizer(target, target_fe, seg_seed)
        warm_start_optimizer(opt, action_native)
        result = opt.optimize()
        new_best = result.get("best_so_far_y", prev_best)
        translated_improvement = max(0.0, prev_best - new_best)

        # Compute the reward BEFORE mutating episode state, so the relative
        # baseline is constructed from the same pre-segment best / FE count.
        if self.reward_mode == "relative":
            # Counterfactual: run the same target from the *lossy* default hand-off
            # over the same segment (same seed) and reward how much the
            # translated state beat it. Isolates exactly what the action controls.
            base_improvement = self._baseline_improvement(
                target, target_fe, seg_seed, prev_best
            )
            reward = float(
                np.clip(
                    (translated_improvement - base_improvement)
                    / max(self._range, 1e-12),
                    -1.0,
                    1.0,
                )
            )
        else:
            reward = scaled_improvement(prev_best, new_best, self._range)

        # Now advance the episode: the target's resulting state becomes the
        # next source; roles swap.
        self._update_best(result)
        self._n_fe = result.get("n_function_evaluations", self._n_fe)
        self._src_native = _augment(opt, self.dim)
        self._src_algo = target
        self._tgt_algo = self.algo_a if target == self.algo_b else self.algo_b
        self._step_idx += 1

        terminated = self._step_idx >= self.n_switches
        info = {"best_y": self._best_y, "n_fe": self._n_fe, "reward": reward}
        obs = self._observation()
        # The next observation is valid only if there is another step to take.
        return obs, reward, terminated, False, info

    def _baseline_improvement(self, target, target_fe, seg_seed, prev_best) -> float:
        """Improvement a *lossy* hand-off (shared population only) would achieve
        over the same segment — the counterfactual the relative reward compares
        against. Does not advance the episode (best / FE are left untouched)."""
        n = self._src_native
        shared = {
            "x": n["x"],
            "y": n["y"],
            "best_x": n["best_x"],
            "best_y": n["best_y"],
        }
        opt = self._make_optimizer(target, target_fe, seg_seed)
        opt.set_data(**shared)
        result = opt.optimize()
        base_best = result.get("best_so_far_y", float("inf"))
        return max(0.0, prev_best - base_best)

    def _update_best(self, result: dict):
        new_best = result.get("best_so_far_y", float("inf"))
        if new_best < self._best_y:
            self._best_y = float(new_best)
            self._best_x = result.get("best_so_far_x", self._best_x)

    def source_has_full_state(self) -> bool:
        return _has_full_state(self._src_algo, self._src_native)
