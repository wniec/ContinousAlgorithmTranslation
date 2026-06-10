"""MadDE optimizer + archive-as-point-set translation."""

import numpy as np
import torch

from cat.models import NormContext, TranslatorPair
from cat.optimizers.DE import ARATE
from cat.optimizers.portfolio import PORTFOLIO
from cat.state import from_canonical, to_canonical
from cat.state.canonical import warm_start_optimizer
from cat.suite import IOHSuite


def _cfg(problem_id="bbob_f001_i01_d05"):
    prob = IOHSuite().get_problem(problem_id)
    return prob, {
        "fitness_function": prob,
        "ndim_problem": prob.dimension,
        "lower_boundary": prob.lower_bounds,
        "upper_boundary": prob.upper_bounds,
    }


def _augment(opt, dim):
    native = dict(opt.get_data())
    native["sigma"] = None
    native["best_x"] = (
        opt.best_so_far_x if opt.best_so_far_x is not None else np.zeros(dim)
    )
    native["best_y"] = opt.best_so_far_y
    return native


# -- optimizer ----------------------------------------------------------- #


def test_madde_optimizes_sphere():
    prob, cfg = _cfg()  # f001 sphere, optimum 79.48
    opt = PORTFOLIO["MADDE"](
        cfg,
        {
            "max_function_evaluations": 4000,
            "target_fe": 4000,
            "seed_rng": 1,
            "verbose": False,
        },
    )
    res = opt.optimize()
    assert res["best_so_far_y"] <= prob.optimum + 1e-3  # solves the sphere


def test_madde_fixed_pop_archive_capacity():
    _, cfg = _cfg()
    opt = PORTFOLIO["MADDE"](
        cfg,
        {
            "max_function_evaluations": 2000,
            "target_fe": 2000,
            "n_individuals": 12,
            "seed_rng": 2,
            "verbose": False,
        },
    )
    opt.optimize()
    data = opt.get_data()
    assert len(data["x"]) == 12  # LPSR disabled in framework mode
    assert len(data["archive"]) <= round(ARATE * 12)
    assert set(["x", "y", "archive", "m_f", "m_cr", "p_m"]).issubset(data)


def test_madde_warmstart_from_shared_only():
    """A PSO/CMA-ES hand-off supplies only x/y; MadDE must cold-default the rest."""
    _, cfg = _cfg()
    a = PORTFOLIO["MADDE"](
        cfg,
        {
            "max_function_evaluations": 1500,
            "target_fe": 800,
            "n_individuals": 12,
            "seed_rng": 3,
            "verbose": False,
        },
    )
    a.optimize()
    native = _augment(a, 5)
    b = PORTFOLIO["MADDE"](
        cfg,
        {
            "max_function_evaluations": 1500,
            "target_fe": 800,
            "n_individuals": 12,
            "seed_rng": 4,
            "verbose": False,
        },
    )
    warm_start_optimizer(
        b,
        {
            "x": native["x"],
            "y": native["y"],
            "best_x": native["best_x"],
            "best_y": native["best_y"],
        },
    )
    res = b.optimize()
    assert np.isfinite(res["best_so_far_y"])


# -- canonical / translation --------------------------------------------- #


def test_archive_canonical_capacity_and_roundtrip():
    _, cfg = _cfg()
    opt = PORTFOLIO["MADDE"](
        cfg,
        {
            "max_function_evaluations": 1500,
            "target_fe": 800,
            "n_individuals": 12,
            "seed_rng": 5,
            "verbose": False,
        },
    )
    opt.optimize()
    native = _augment(opt, 5)
    st = to_canonical("MADDE", native)
    assert st.specific["archive"].shape == (round(ARATE * 12), 5)  # padded to capacity
    back = from_canonical(st)
    st2 = to_canonical("MADDE", {**native, **back})
    assert np.allclose(
        st.specific["archive"].numpy(), st2.specific["archive"].numpy(), atol=1e-5
    )


def _rand_state(algo, B=2, N=10, D=4):
    g = torch.Generator().manual_seed(abs(hash(algo)) % 2**31)
    pos, vals = torch.randn(B, N, D, generator=g), torch.randn(B, N, generator=g)
    bx, by = torch.randn(B, D, generator=g), torch.randn(B, generator=g)
    if algo == "PSO":
        spec = {
            "velocity": torch.randn(B, N, D, generator=g),
            "pbest_x": torch.randn(B, N, D, generator=g),
        }
    elif algo == "CMAES":
        A = torch.randn(B, D, D, generator=g)
        spec = {
            "mean": torch.randn(B, D, generator=g),
            "sigma_cov": A @ A.transpose(-2, -1) + torch.eye(D),
            "p_c": torch.randn(B, D, generator=g),
            "p_s": torch.randn(B, D, generator=g),
        }
    else:  # MADDE
        spec = {"archive": torch.randn(B, round(ARATE * N), D, generator=g)}
    from cat.state.canonical import CanonicalState

    return CanonicalState(algo, pos, vals, bx, by, spec)


def test_translate_to_madde_produces_archive():
    pair = TranslatorPair("CMAES", "MADDE").eval()
    s = _rand_state("CMAES", N=10, D=4)
    ctx = NormContext.from_shared(s.positions, s.values)
    with torch.no_grad():
        out = pair.translate(s, "MADDE", ctx)
    assert list(out.specific) == ["archive"]
    assert out.specific["archive"].shape == (2, round(ARATE * 10), 4)
    assert torch.isfinite(out.specific["archive"]).all()


def test_madde_archive_decode_is_deterministic():
    """Same state -> same archive (deterministic seeding) so cycle drift is stable."""
    pair = TranslatorPair("PSO", "MADDE").eval()
    s = _rand_state("PSO", N=8, D=3)
    ctx = NormContext.from_shared(s.positions, s.values)
    with torch.no_grad():
        a = pair.translate(s, "MADDE", ctx).specific["archive"]
        b = pair.translate(s, "MADDE", ctx).specific["archive"]
    assert torch.equal(a, b)


def test_madde_cycle_runs_both_directions():
    from cat.losses import field_distance

    for a, b in (("PSO", "MADDE"), ("MADDE", "CMAES")):
        pair = TranslatorPair(a, b).eval()
        s = _rand_state(a)
        ctx = NormContext.from_shared(s.positions, s.values)
        with torch.no_grad():
            _, back = pair.cycle(s, ctx)
            drift = float(field_distance(back, s, ctx))
        assert np.isfinite(drift)
        assert sorted(back.specific) == sorted(s.specific)
