"""Shared test fixtures: run a real optimizer one checkpoint on a BBOB problem
and return its augmented native warm-start dict."""

import numpy as np
import pytest

from cat.optimizers.portfolio import PORTFOLIO
from cat.suite import IOHSuite


def run_one(algo: str, problem_id: str = "bbob_f001_i01_d05", target_fe: int = 400):
    """Run *algo* for one checkpoint and return (optimizer, augmented_native)."""
    suite = IOHSuite()
    prob = suite.get_problem(problem_id)
    cfg = {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }
    opts = {
        "max_function_evaluations": 2000,
        "target_fe": target_fe,
        "seed_rng": 7,
        "verbose": False,
    }
    opt = PORTFOLIO[algo](cfg, dict(opts))
    opt.optimize()
    native = dict(opt.get_data())
    native["sigma"] = getattr(opt, "sigma", None)
    native["best_x"] = (
        opt.best_so_far_x if opt.best_so_far_x is not None else np.zeros(prob.dimension)
    )
    native["best_y"] = opt.best_so_far_y
    return opt, native


@pytest.fixture(scope="module")
def pso_native():
    return run_one("PSO")[1]


@pytest.fixture(scope="module")
def cmaes_native():
    return run_one("CMAES")[1]
