"""Collect realistic mid-optimization states by running BBOB episodes that
switch between two optimizers.

This mirrors the run/switch loop of DAS's ``DASEnv._run_optimizer``: an episode
optimizes one BBOB problem; at each exponentially-spaced checkpoint the
scheduled optimizer runs to its FE target, warm-started from the population
carried over from the previous checkpoint. After each run we *snapshot* the
optimizer's full native state (``get_data()`` + ``sigma`` + best-so-far) as one
training example.

The translation objectives (cycle-consistency, reconstruction, utility proxy)
are all unsupervised on a *single* state, so we just need a varied pool of
genuine A-states and B-states — any switching schedule produces usable data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cat.optimizers.base import sample_switch_points
from cat.optimizers.portfolio import PORTFOLIO
from cat.suite import IOHSuite


@dataclass
class Snapshot:
    algo: str
    native: dict  # augmented native warm-start dict (numpy), ready for to_canonical
    problem_id: str
    dimension: int
    checkpoint: int
    best_y: float


def _augment(opt, dim: int) -> dict:
    """Native warm-start dict plus the attributes canonical conversion needs."""
    native = {
        k: (np.copy(v) if isinstance(v, np.ndarray) else v)
        for k, v in opt.get_data().items()
        if v is not None
    }
    native["sigma"] = getattr(opt, "sigma", None)
    native["best_x"] = (
        np.copy(opt.best_so_far_x) if opt.best_so_far_x is not None else np.zeros(dim)
    )
    native["best_y"] = float(opt.best_so_far_y)
    return native


def _has_full_state(algo: str, native: dict) -> bool:
    """Snapshot is usable only if every native key the spec needs is present."""
    from cat.state.schema import get_spec

    required = ["x", "y"] + [f.native_key for f in get_spec(algo).specific]
    return all(native.get(k) is not None for k in required)


def collect_episode(
    algo_a: str,
    algo_b: str,
    problem_id: str,
    suite: IOHSuite,
    *,
    fe_multiplier: int,
    n_switches: int,
    schedule: str,
    n_individuals: int | None,
    seed: int,
    problem_idx: int,
    n_individuals_a: int | None = None,
    n_individuals_b: int | None = None,
) -> list[Snapshot]:
    """Run one episode on *problem_id*, returning one snapshot per segment.

    The episode is split into ``n_switches`` segments whose boundaries are drawn
    log-uniformly over the FE budget (independently for this problem), so the
    optimizer is switched at randomized, early-biased positions rather than at
    fixed checkpoints. A snapshot of the active optimizer's state is taken at the
    end of each segment.

    ``n_individuals`` is the shared fallback; ``n_individuals_a``/``_b`` let
    each algorithm run with its own population size. A hand-off to a
    differently-sized optimizer isn't translated here (this collector doesn't
    use the neural translator) — the carried-forward population is simply
    smaller than the new optimizer wants, so its own ``set_data``/
    ``initialize`` falls back to a cold, random init, which is a perfectly
    valid diverse snapshot for these single-state, unsupervised losses.
    """
    prob = suite.get_problem(problem_id)
    dim = prob.dimension
    max_fe = fe_multiplier * dim
    n_a = n_individuals_a if n_individuals_a is not None else n_individuals
    n_b = n_individuals_b if n_individuals_b is not None else n_individuals
    pop = max(n_a or 20, n_b or 20)

    rng = np.random.default_rng((seed * 1_000_003 + problem_idx) % (2**31))
    checkpoints = sample_switch_points(n_switches, max_fe, pop, rng)
    cfg = {
        "fitness_function": prob,
        "ndim_problem": dim,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }

    carried: dict = {}
    best_y = float("inf")
    best_x = None
    n_fe = 0
    snapshots: list[Snapshot] = []

    for ck in range(n_switches):
        if schedule == "alternate":
            algo = algo_a if ck % 2 == 0 else algo_b
        else:  # random
            algo = algo_a if rng.random() < 0.5 else algo_b

        options = {
            "max_function_evaluations": max_fe,
            "target_fe": int(checkpoints[ck]),
            "best_so_far_y": best_y if best_y < float("inf") else float("inf"),
            "seed_rng": (seed * 1_000_000 + problem_idx * 1_000 + ck) % (2**31),
            "verbose": False,
        }
        n = n_a if algo == algo_a else n_b
        if n is not None:
            options["n_individuals"] = n

        opt = PORTFOLIO[algo](cfg, options)
        opt.n_function_evaluations = n_fe
        opt.set_data(
            best_x=best_x,
            best_y=best_y if best_y < float("inf") else None,
            **carried,
        )
        result = opt.optimize()

        # Update episode bests + carried population for the next optimizer.
        new_best_y = result.get("best_so_far_y", float("inf"))
        if new_best_y < best_y:
            best_y = new_best_y
            best_x = result.get("best_so_far_x")
        n_fe = result.get("n_function_evaluations", n_fe)

        native = _augment(opt, dim)
        carried = opt.get_data()

        if _has_full_state(algo, native):
            snapshots.append(
                Snapshot(
                    algo=algo,
                    native=native,
                    problem_id=problem_id,
                    dimension=dim,
                    checkpoint=ck,
                    best_y=float(best_y),
                )
            )

    return snapshots


def collect_dataset(
    algo_a: str,
    algo_b: str,
    problem_ids: list[str],
    *,
    fe_multiplier: int = 2000,
    n_switches: int = 8,
    schedule: str = "alternate",
    n_individuals: int | None = None,
    n_individuals_a: int | None = None,
    n_individuals_b: int | None = None,
    seed: int = 42,
    progress: bool = True,
) -> list[Snapshot]:
    """Run an episode per problem id and gather all snapshots."""
    suite = IOHSuite()
    snapshots: list[Snapshot] = []
    iterator = enumerate(problem_ids)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(
            list(iterator), desc=f"collect {algo_a}<->{algo_b}", unit="prob"
        )
    for idx, pid in iterator:
        snapshots.extend(
            collect_episode(
                algo_a,
                algo_b,
                pid,
                suite,
                fe_multiplier=fe_multiplier,
                n_switches=n_switches,
                schedule=schedule,
                n_individuals=n_individuals,
                n_individuals_a=n_individuals_a,
                n_individuals_b=n_individuals_b,
                seed=seed,
                problem_idx=idx,
            )
        )
    return snapshots
