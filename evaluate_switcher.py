"""Benchmark a translating switcher over the whole BBOB suite.

    python evaluate_switcher.py PSO CMAES --model models/PSO_CMAES_ppo.pt \
        --n-switches 6 --fe-multiplier 2000 --cdb 1.0 --switch-mode random

An *agent* runs each BBOB problem for the full FE budget while **randomly
switching** between the two named algorithms across ``--n-switches + 1``
segments. At every boundary where the algorithm actually changes, the source
optimizer's internal state is handed to the target. We compare five ways of
running that same schedule:

* ``<A>_only`` / ``<B>_only`` — no switching at all, one algorithm for the whole
  budget (the "would switching have helped?" baselines);
* ``cold_switch``           — same schedule, but every hand-off **cold-starts**
  the next optimizer (state discarded — the naive switcher);
* ``lossy_switch``          — same schedule, hand-offs carry only the shared
  population (positions/values/best), discarding algorithm-specific structure
  (the current *default* hand-off);
* ``translated_switch``     — same schedule, hand-offs run through the trained
  translator (velocities <-> covariance, etc.).

All five see the same problem, the same segment seeds, and (the switchers) the
same random algorithm schedule and switch points, so differences are attributable
to the hand-off alone.

The primary metric is **AOCC** (Area Over the Convergence Curve, in [0, 1],
higher = better anytime performance), computed over each run's best-so-far
trajectory exactly as DynamicAlgorithmSelection's ``agents/agent_utils`` — the
precision ``best_y - optimum`` is clipped to [1e-8, 1e8], log10-normalized, and
``1 - that`` is averaged over the FE budget. Methods are ranked per run by AOCC
(mean rank reported, lower = better) with head-to-head AOCC win rates. **ERT**
(Expected Running Time) is also reported per method, pooled over all runs, across
a fixed precision-target ladder (1e-2, 1e-3, ..., 1e-7).

Switch points come from one of two modes (``--switch-mode``), both honoring the
concentration-of-decision-boundaries knob ``--cdb``:

* ``random`` — sampled log-uniformly per problem via ``sample_switch_points``
  (``--cdb`` warps *where in log-FE space* they land: >1 earlier, <1 later);
* ``cdb``    — deterministic exponentially-spaced checkpoints via
  ``get_checkpoints`` (``--cdb`` is the growth ratio: 1.0 = uniform spacing,
  >1 = shorter early segments).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cat.data.collector import _augment, _has_full_state
from cat.data.dataset import collate
from cat.models.norm import NormContext
from cat.optimizers.base import get_checkpoints, sample_switch_points
from cat.optimizers.portfolio import PORTFOLIO
from cat.state.canonical import (
    CanonicalState,
    from_canonical,
    to_canonical,
    warm_start_optimizer,
)
from cat.suite import IOHSuite
from cat.suite.bbob_splits import (
    ALL_DIMS,
    ALL_FUNCTIONS,
    INSTANCE_IDS,
    build_problem_ids,
)
from cat.train_loop import load_translator


# --------------------------------------------------------------------------- #
# optimizer plumbing                                                          #
# --------------------------------------------------------------------------- #


def _problem_cfg(prob) -> dict:
    return {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _make(algo, cfg, max_fe, target_fe, best_y, n_individuals, seed):
    options = {
        "max_function_evaluations": max_fe,
        "target_fe": int(target_fe),
        "best_so_far_y": best_y if best_y < float("inf") else float("inf"),
        "seed_rng": seed,
        "verbose": False,
    }
    if n_individuals is not None:
        options["n_individuals"] = n_individuals
    return PORTFOLIO[algo](cfg, options)


def _translate(pair, native, source_algo, target_algo) -> dict:
    """Run the source native state through the translator, returning the target's
    full warm-start dict (shared population + decoded specific fields)."""
    state = to_canonical(source_algo, native)
    batch = collate([state])
    ctx = NormContext.from_shared(batch.positions, batch.values)
    with torch.no_grad():
        out: CanonicalState = pair.translate(batch, target_algo, ctx)
    return from_canonical(out.index(0))


def _shared(native: dict) -> dict:
    """The lossy default hand-off: carry only the shared population."""
    return {
        "x": native["x"],
        "y": native["y"],
        "best_x": native["best_x"],
        "best_y": native["best_y"],
    }


# --------------------------------------------------------------------------- #
# anytime-performance metrics: AOCC (+ ERT)                                    #
# --------------------------------------------------------------------------- #

# AOCC precision bounds, matching DynamicAlgorithmSelection's
# ``agents/agent_utils.get_runtime_stats``: precision is clipped to [1e-8, 1e8]
# and log10-normalized onto [0, 1].
_AOCC_LB, _AOCC_UB = 1e-8, 1e8
_AOCC_LOG_LB, _AOCC_LOG_UB = -8.0, 8.0

# Fixed precision-target ladder ERT is always reported against (1e-2 down to 1e-7).
_ERT_TARGETS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


class _Recorder:
    """Wraps a problem callable and logs best-so-far improvements as ``(fe, y)``
    over one whole run (all its segments) — the convergence curve AOCC and ERT
    need. IOH only exposes a per-evaluation *snapshot* (``problem.state``), so we
    accumulate the trajectory here instead. ``fe`` counts actual calls, which is
    the run's true FE spend (cold restarts re-evaluate populations, etc.)."""

    def __init__(self, prob):
        self._prob = prob
        self.fe = 0
        self.best = float("inf")
        self.history: list[tuple[int, float]] = []

    def __call__(self, x) -> float:
        y = float(self._prob(x))
        self.fe += 1
        if y < self.best:
            self.best = y
            self.history.append((self.fe, y))
        return y


def _aocc(history, budget, optimum) -> float:
    """Area Over the Convergence Curve in [0, 1] (higher = better anytime
    performance), computed exactly as DynamicAlgorithmSelection's
    ``get_runtime_stats``: precision ``y - optimum`` is clipped to [1e-8, 1e8],
    log10-normalized onto [0, 1], and ``1 - that`` is averaged over the FE budget
    — each improvement's value credited to the interval ending at its FE, plus a
    final plateau out to ``budget``."""
    if not history or budget <= 0:
        return 0.0
    area = 0.0
    last_i = 0
    for i, fitness in history:
        clipped = float(np.clip(fitness - optimum, _AOCC_LB, _AOCC_UB))
        norm = (np.log10(clipped) - _AOCC_LOG_LB) / (_AOCC_LOG_UB - _AOCC_LOG_LB)
        area += (1.0 - norm) * (i - last_i)
        last_i = i
    final_clipped = float(np.clip(history[-1][1] - optimum, _AOCC_LB, _AOCC_UB))
    final_norm = (np.log10(final_clipped) - _AOCC_LOG_LB) / (_AOCC_LOG_UB - _AOCC_LOG_LB)
    area += (1.0 - final_norm) * max(0, budget - history[-1][0])
    return area / budget


def _fe_to_targets(history, optimum) -> list[float]:
    """First FE at which precision (``y - optimum``) reaches each target in
    ``_ERT_TARGETS``, or ``inf`` for targets never reached in the run (the ERT
    'failure' case). Returned in ``_ERT_TARGETS`` order."""
    result = [float("inf")] * len(_ERT_TARGETS)
    for i, fitness in history:
        prec = fitness - optimum
        for j, tgt in enumerate(_ERT_TARGETS):
            if result[j] == float("inf") and prec <= tgt:
                result[j] = float(i)
        if all(r != float("inf") for r in result):
            break  # every target reached; the rest of the trajectory can't help
    return result


def _run_segment(algo, cfg, max_fe, target_fe, n_fe_start, best_y, n, seed, payload):
    """Run one segment; ``payload`` is the warm-start dict, or None to cold-start."""
    opt = _make(algo, cfg, max_fe, target_fe, best_y, n, seed)
    opt.n_function_evaluations = n_fe_start
    if payload is None:
        opt.set_data()  # cold
    else:
        warm_start_optimizer(opt, payload)
    result = opt.optimize()
    return opt, result


# --------------------------------------------------------------------------- #
# runners                                                                      #
# --------------------------------------------------------------------------- #


def _metrics(rec: _Recorder, budget: int, optimum: float) -> dict:
    return {
        "best": rec.best,
        "aocc": _aocc(rec.history, budget, optimum),
        "fett": _fe_to_targets(rec.history, optimum),  # one per _ERT_TARGETS entry
    }


def run_standalone(algo, prob, max_fe, optimum, n, seed) -> dict:
    """One algorithm for the whole budget (no switching)."""
    rec = _Recorder(prob)
    cfg = _problem_cfg(prob)
    cfg["fitness_function"] = rec
    _run_segment(algo, cfg, max_fe, max_fe, 0, float("inf"), n, seed, None)
    return _metrics(rec, max_fe, optimum)


def run_switcher(mode, algos, checkpoints, prob, optimum, n_for, base_seed, pair) -> dict:
    """Run the random algorithm schedule under one hand-off ``mode`` (one of
    ``translated`` / ``lossy`` / ``cold``). The three modes differ **only at real
    algorithm switches**; a "stay" on the same algorithm is warm-continued
    identically for all modes, so the comparison isolates the hand-off. Returns a
    metrics dict (best / aocc / fett) plus ``fallbacks`` — the count of real
    switches the translated mode had to serve lossily/cold because the source
    state was degenerate."""
    rec = _Recorder(prob)
    cfg = _problem_cfg(prob)
    cfg["fitness_function"] = rec
    dim = cfg["ndim_problem"]
    max_fe = int(checkpoints[-1])
    n_fe = 0
    native = None
    cur_algo = None
    fallbacks = 0

    for i, algo in enumerate(algos):
        target_fe = int(checkpoints[i])
        seg_seed = base_seed + i + 1

        if i == 0:
            payload = None  # no source yet: cold-start the first segment
        else:
            full = _has_full_state(cur_algo, native)
            has_shared = native.get("x") is not None and native.get("y") is not None
            if algo == cur_algo:
                # Stay: warm-continue on the algorithm's own state (all modes).
                payload = native if full else (_shared(native) if has_shared else None)
            elif mode == "cold":
                payload = None  # discard state at the switch
            elif mode == "translated" and full:
                payload = _translate(pair, native, cur_algo, algo)
            elif has_shared:
                payload = _shared(native)  # lossy default, or translated fallback
            else:
                payload = None  # degenerate source: nothing to carry
            if mode == "translated" and algo != cur_algo and not full:
                fallbacks += 1

        # A cold-started segment (payload None) knows no incumbent; a warm one
        # carries the running best so the optimizer keeps the current champion.
        best_in = rec.best if payload is not None else float("inf")
        opt, _ = _run_segment(
            algo, cfg, max_fe, target_fe, n_fe, best_in, n_for(algo), seg_seed, payload
        )
        n_fe = opt.n_function_evaluations
        native = _augment(opt, dim)
        cur_algo = algo

    return {**_metrics(rec, max_fe, optimum), "fallbacks": int(fallbacks)}


def _schedule(algo_a, algo_b, n_switches, rng) -> list[str]:
    """A random length-``n_switches+1`` algorithm sequence. Guaranteed to contain
    at least one real switch when ``n_switches >= 1`` (so the translator is
    actually exercised)."""
    algos = [algo_a if rng.random() < 0.5 else algo_b for _ in range(n_switches + 1)]
    if n_switches >= 1 and len(set(algos)) == 1:
        flip = int(rng.integers(1, n_switches + 1))
        algos[flip] = algo_b if algos[flip] == algo_a else algo_a
    return algos


def _checkpoints(mode, n_switches, max_fe, pop, cdb, rng) -> np.ndarray:
    """The ``n_switches + 1`` cumulative segment-end FE targets (last == max_fe)."""
    n = n_switches + 1
    if mode == "cdb":
        return get_checkpoints(n, max_fe, pop, cdb)
    return sample_switch_points(n, max_fe, pop, rng, cdb=cdb)


# --------------------------------------------------------------------------- #
# per-problem evaluation + aggregation                                        #
# --------------------------------------------------------------------------- #


def evaluate_problem(pair, args, pid, suite, n_for, problem_idx):
    prob = suite.get_problem(pid)
    dim = prob.dimension
    max_fe = args.fe_multiplier * dim
    pop = max(n_for(args.algo_a) or 20, n_for(args.algo_b) or 20)
    optimum = getattr(prob, "optimum", None)
    if optimum is None:  # AOCC/ERT are defined relative to the known optimum
        raise RuntimeError(f"{pid}: problem exposes no optimum; cannot score AOCC")
    optimum = float(optimum)
    a, b = args.algo_a, args.algo_b

    base = (args.seed * 1_000_003 + problem_idx * 1_009) % (2**31)
    # Standalone baselines don't depend on the random schedule: run once.
    standalone = {
        f"{a}_only": run_standalone(a, prob, max_fe, optimum, n_for(a), base + 1),
        f"{b}_only": run_standalone(b, prob, max_fe, optimum, n_for(b), base + 2),
    }

    rows = []
    for rep in range(args.repeats):
        rng = np.random.default_rng((base + 7_919 * (rep + 1)) % (2**31))
        algos = _schedule(a, b, args.n_switches, rng)
        checkpoints = _checkpoints(args.switch_mode, args.n_switches, max_fe, pop, args.cdb, rng)
        seg_base = (base + 31 * (rep + 1)) % (2**31)

        row = {
            "problem_id": pid,
            "dimension": dim,
            "optimum": optimum,
            "budget": max_fe,
            **standalone,
        }
        for mode, key in (
            ("cold", "cold_switch"),
            ("lossy", "lossy_switch"),
            ("translated", "translated_switch"),
        ):
            res = run_switcher(
                mode, algos, checkpoints, prob, optimum, n_for, seg_base, pair
            )
            row[key] = res
            if key == "translated_switch":
                row["translated_fallbacks"] = res["fallbacks"]
        rows.append(row)
    return rows


def _avg_ranks(values: list[float]) -> list[float]:
    """1-indexed average ranks (lower value -> lower rank), ties share the mean."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _ert(fett: np.ndarray, budgets: np.ndarray) -> tuple[float, int]:
    """Expected Running Time (COCO definition) at the fixed target the ``fett``
    (FE-to-target, ``inf`` on failure) values were recorded against: total FE
    spent — successes counting their FE-to-target, failures their full budget —
    divided by the number of successes. ``inf`` when no run hit the target."""
    success = np.isfinite(fett)
    n_succ = int(success.sum())
    if n_succ == 0:
        return float("inf"), 0
    spent = fett[success].sum() + budgets[~success].sum()
    return float(spent / n_succ), n_succ


def summarize(rows, args):
    if not rows:
        print("no evaluable problems")
        return
    a, b = args.algo_a, args.algo_b
    methods = [f"{a}_only", f"{b}_only", "cold_switch", "lossy_switch", "translated_switch"]
    n = len(rows)

    aocc = {m: np.array([r[m]["aocc"] for r in rows], dtype=float) for m in methods}
    best = {m: np.array([r[m]["best"] for r in rows], dtype=float) for m in methods}
    # fett[m] is (n_runs, n_targets): FE-to-target per run for each _ERT_TARGETS entry.
    fett = {m: np.array([r[m]["fett"] for r in rows], dtype=float) for m in methods}
    opt = np.array([r["optimum"] for r in rows], dtype=float)
    budgets = np.array([r["budget"] for r in rows], dtype=float)

    # Primary metric: mean rank by AOCC (higher AOCC = better -> rank on -aocc).
    rank_sums = {m: 0.0 for m in methods}
    for idx in range(n):
        ranks = _avg_ranks([-rows[idx][m]["aocc"] for m in methods])
        for m, rk in zip(methods, ranks):
            rank_sums[m] += rk
    mean_rank = {m: rank_sums[m] / n for m in methods}

    print(
        f"\n=== {a} <-> {b}  |  switch-mode={args.switch_mode}  cdb={args.cdb}  "
        f"n_switches={args.n_switches}  fe={args.fe_multiplier}/dim  "
        f"repeats={args.repeats}  |  {n} runs ==="
    )
    print(f"  {'method':20s} {'AOCC_rank':>9s} {'mean_AOCC':>10s}  median precision")
    for m in methods:
        star = " *" if mean_rank[m] == min(mean_rank.values()) else "  "
        prec = np.median(best[m] - opt)
        print(f"{star}{m:20s} {mean_rank[m]:>9.3f} {aocc[m].mean():>10.4f}  {prec:.4g}")

    # ERT across the fixed precision-target ladder, pooled over all runs. NOTE:
    # pooling across heterogeneous functions/dims is non-standard COCO usage (ERT
    # is normally per function+dim); read it as a coarse cross-suite summary.
    # Cell = ERT in FEs ('inf' = no run reached that target); (k) = # successes.
    print(f"\n  ERT (FEs) to precision target, pooled over {n} runs  [(k)=#runs reaching it]:")
    header = "  " + f"{'method':20s}" + "".join(f"{t:>13.0e}" for t in _ERT_TARGETS)
    print(header)
    for m in methods:
        cells = []
        for j in range(len(_ERT_TARGETS)):
            ert, n_succ = _ert(fett[m][:, j], budgets)
            cells.append(("inf" if not np.isfinite(ert) else f"{ert:.0f}") + f"({n_succ})")
        print("  " + f"{m:20s}" + "".join(f"{c:>13s}" for c in cells))

    # Head-to-head on AOCC: is the learned hand-off worth it? (higher = win)
    t = aocc["translated_switch"]
    print("\n  translated_switch AOCC vs:")
    best_standalone = np.maximum(aocc[f"{a}_only"], aocc[f"{b}_only"])
    for label, base_arr in (
        ("cold_switch", aocc["cold_switch"]),
        ("lossy_switch", aocc["lossy_switch"]),
        ("best standalone", best_standalone),
    ):
        ties_mask = np.isclose(t, base_arr)
        wins = int(((t > base_arr) & ~ties_mask).sum())
        ties = int(ties_mask.sum())
        losses = n - wins - ties
        print(f"    {label:16s}: {wins} win / {ties} tie / {losses} loss")

    fb = np.array([r["translated_fallbacks"] for r in rows])
    if fb.sum():
        print(
            f"\n  note: {int((fb > 0).sum())}/{n} runs had a degenerate source state "
            f"served lossily ({int(fb.sum())} hand-offs total)"
        )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))
    p.add_argument("--model", default=None, help="translator checkpoint (default models/<A>_<B>.pt)")
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5],
                   help=f"BBOB dimensions (available: {ALL_DIMS})")
    p.add_argument("--instances", type=int, nargs="+", default=None,
                   help=f"BBOB instance ids (default: all {len(INSTANCE_IDS)} of them)")
    p.add_argument("--functions", type=int, nargs="+", default=None,
                   help="BBOB function ids 1-24 (default: the whole suite)")
    p.add_argument("--fe-multiplier", type=int, default=2000, help="FE budget = this * dim")
    p.add_argument("--n-switches", type=int, default=6, help="number of hand-offs per run")
    p.add_argument("--cdb", type=float, default=1.0,
                   help="concentration of switch points (see module docstring)")
    p.add_argument("--switch-mode", choices=["random", "cdb"], default="random",
                   help="'random' samples switch points log-uniformly per run; "
                        "'cdb' places them deterministically via get_checkpoints")
    p.add_argument("--n-individuals", type=int, default=None, help="shared population size")
    p.add_argument("--n-individuals-a", type=int, default=None, help="override for algo_a")
    p.add_argument("--n-individuals-b", type=int, default=None, help="override for algo_b")
    p.add_argument("--repeats", type=int, default=1,
                   help="random schedules evaluated per problem (averaged into the stats)")
    p.add_argument("--max-problems", type=int, default=None, help="cap the (shuffled) problem set")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model or f"models/{args.algo_a}_{args.algo_b}.pt"
    pair = load_translator(model, device=args.device)
    print(f"loaded translator {pair.algo_a}<->{pair.algo_b} from {model}")

    n_a = args.n_individuals_a if args.n_individuals_a is not None else args.n_individuals
    n_b = args.n_individuals_b if args.n_individuals_b is not None else args.n_individuals

    def n_for(algo):
        return n_a if algo == args.algo_a else n_b

    funcs = set(args.functions) if args.functions else ALL_FUNCTIONS
    instances = args.instances if args.instances else INSTANCE_IDS
    ids = build_problem_ids(funcs, args.dims, instances)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    if args.max_problems is not None:
        ids = ids[: args.max_problems]
    suite = IOHSuite()

    from tqdm import tqdm

    rows = []
    for idx, pid in enumerate(tqdm(ids, desc="benchmark", unit="prob")):
        rows.extend(evaluate_problem(pair, args, pid, suite, n_for, idx))
    summarize(rows, args)


if __name__ == "__main__":
    main()
