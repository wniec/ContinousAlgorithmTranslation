"""Evaluate a trained translator: does the learned hand-off beat the default?

    python evaluate.py PSO CMAES [options]

For each BBOB **test** problem we warm up the *source* optimizer to build a
genuine internal state, then switch to the *target* optimizer for the remaining
budget under three hand-offs:

* **translated** — the source state is run through the trained translator to
  produce the target's algorithm-specific fields (velocities <-> covariance);
* **lossy**      — the current default: carry only the shared population
  (positions / values / best), discarding all algorithm-specific structure;
* **cold**       — a fresh target optimizer that ignores the source entirely.

We report the best objective value each hand-off reaches. Success = the
translated hand-off is at least as good as lossy on average. We also report the
cycle-consistency drift (how much A -> B -> A perturbs the source state), the
quantity the translator is trained to minimize.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cat.data.collector import _augment, _has_full_state
from cat.data.dataset import collate
from cat.losses import field_distance
from cat.models.norm import NormContext
from cat.optimizers.base import sample_switch_points
from cat.optimizers.portfolio import PORTFOLIO
from cat.state.canonical import (
    CanonicalState,
    from_canonical,
    to_canonical,
    warm_start_optimizer,
)
from cat.suite import IOHSuite
from cat.suite.bbob_splits import get_train_test_split
from cat.train_loop import load_translator


def _problem_cfg(prob):
    return {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _make(algo, cfg, max_fe, target_fe, best_y, n_individuals, seed):
    options = {
        "max_function_evaluations": max_fe,
        "target_fe": target_fe,
        "best_so_far_y": best_y if best_y < float("inf") else float("inf"),
        "seed_rng": seed,
        "verbose": False,
    }
    if n_individuals is not None:
        options["n_individuals"] = n_individuals
    return PORTFOLIO[algo](cfg, options)


def translate_single(pair, state: CanonicalState, target_algo: str) -> CanonicalState:
    batch = collate([state])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    with torch.no_grad():
        out = pair.translate(batch, target_algo, ctx)
    return out.index(0)


def cycle_drift(pair, state: CanonicalState) -> float:
    batch = collate([state])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    with torch.no_grad():
        _, back = pair.cycle(batch, ctx)
        return float(field_distance(back, batch, ctx))


def run_target(algo, cfg, max_fe, n_fe_start, n_individuals, seed, native=None):
    """Run target optimizer for the remaining budget; return final best_y."""
    opt = _make(algo, cfg, max_fe, max_fe, float("inf"), n_individuals, seed)
    opt.n_function_evaluations = n_fe_start
    if native is None:
        opt.set_data()  # cold
    else:
        warm_start_optimizer(opt, native)
    result = opt.optimize()
    return float(result.get("best_so_far_y", float("inf")))


def evaluate_problem(
    pair,
    source,
    target,
    problem_id,
    suite,
    *,
    fe_multiplier,
    n_individuals,
    seed,
    problem_idx,
):
    prob = suite.get_problem(problem_id)
    dim = prob.dimension
    cfg = _problem_cfg(prob)
    max_fe = fe_multiplier * dim
    pop = n_individuals if n_individuals is not None else 20

    base_seed = (seed * 1_000_000 + problem_idx * 1_000) % (2**31)

    # One switch per episode (source -> target), positioned log-uniformly over
    # the FE budget and sampled independently per problem — so the hand-off is
    # tested at randomized, early-biased points rather than a fixed mid-run one.
    sw_rng = np.random.default_rng((seed * 7_919 + problem_idx) % (2**31))
    switch = int(sample_switch_points(2, max_fe, pop, sw_rng)[0])

    # --- warm up the source optimizer to `switch` FEs -------------------- #
    src = _make(source, cfg, max_fe, switch, float("inf"), n_individuals, base_seed)
    src.optimize()
    native_src = _augment(src, dim)
    if not _has_full_state(source, native_src):
        return None
    best_after_warmup = float(src.best_so_far_y)
    n_fe = src.n_function_evaluations

    state = to_canonical(source, native_src)

    # --- three hand-offs over the remaining budget ----------------------- #
    # `translated_native` already bundles the shared population (from the source
    # state's positions) plus the decoded target-specific fields.
    translated_native = from_canonical(translate_single(pair, state, target))
    # The lossy hand-off carries only the shared population — the current default.
    shared = {
        "x": native_src["x"],
        "y": native_src["y"],
        "best_x": native_src["best_x"],
        "best_y": native_src["best_y"],
    }

    results = {
        "translated": run_target(
            target, cfg, max_fe, n_fe, n_individuals, base_seed + 1, translated_native
        ),
        "lossy": run_target(
            target, cfg, max_fe, n_fe, n_individuals, base_seed + 1, shared
        ),
        "cold": run_target(
            target, cfg, max_fe, n_fe, n_individuals, base_seed + 1, None
        ),
    }
    return {
        "problem_id": problem_id,
        "best_after_warmup": best_after_warmup,
        "drift": cycle_drift(pair, state),
        **{f"best_{k}": v for k, v in results.items()},
    }


def summarize(rows, source, target):
    if not rows:
        print("no evaluable problems")
        return
    arr = {
        k: np.array([r[f"best_{k}"] for r in rows])
        for k in ("translated", "lossy", "cold")
    }
    drift = np.array([r["drift"] for r in rows])
    print(f"\n=== {source} -> {target}  ({len(rows)} problems) ===")
    for k in ("translated", "lossy", "cold"):
        print(
            f"  {k:11s} best_y: mean={arr[k].mean():.4g}  median={np.median(arr[k]):.4g}"
        )
    # Lower best_y is better. Win = translated strictly better than lossy.
    wins = int((arr["translated"] < arr["lossy"]).sum())
    ties = int(np.isclose(arr["translated"], arr["lossy"]).sum())
    print(f"  translated beats lossy on {wins}/{len(rows)} problems ({ties} ties)")
    print(f"  mean cycle-consistency drift (A->B->A): {drift.mean():.4g}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))
    p.add_argument(
        "--model",
        default=None,
        help="translator checkpoint (default models/<A>_<B>.pt)",
    )
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--split", choices=["easy", "hard", "random"], default="easy")
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-individuals", type=int, default=None)
    p.add_argument("--max-problems", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model or f"models/{args.algo_a}_{args.algo_b}.pt"
    pair = load_translator(model, device=args.device)
    print(f"loaded translator {pair.algo_a}<->{pair.algo_b} from {model}")

    _, test_ids = get_train_test_split(args.split, args.dims)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(test_ids)
    test_ids = test_ids[: args.max_problems]
    suite = IOHSuite()

    from tqdm import tqdm

    for source, target in ((args.algo_a, args.algo_b), (args.algo_b, args.algo_a)):
        rows = []
        for idx, pid in enumerate(
            tqdm(test_ids, desc=f"{source}->{target}", unit="prob")
        ):
            row = evaluate_problem(
                pair,
                source,
                target,
                pid,
                suite,
                fe_multiplier=args.fe_multiplier,
                n_individuals=args.n_individuals,
                seed=args.seed,
                problem_idx=idx,
            )
            if row is not None:
                rows.append(row)
        summarize(rows, source, target)


if __name__ == "__main__":
    main()
