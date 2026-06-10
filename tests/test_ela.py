"""ELA feature computation and its presence in the env observation."""

import numpy as np

from cat.rl.env import TranslationEnv
from cat.suite import IOHSuite, build_problem_ids
from cat.suite.ela import ELA_DIM, compute_ela_features


def test_ela_zeros_for_tiny_or_degenerate_sample():
    # Too few points -> zeros.
    assert np.array_equal(
        compute_ela_features(np.random.randn(10, 3), np.random.randn(10)),
        np.zeros(ELA_DIM, dtype=np.float32),
    )


def test_ela_features_on_real_sample():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, size=(200, 3))
    y = (x**2).sum(axis=1)  # sphere -> well-defined landscape
    feats = compute_ela_features(x, y)
    assert feats.shape == (ELA_DIM,)
    assert np.all(np.isfinite(feats))
    assert np.any(feats != 0.0)  # a real landscape yields non-trivial features


def test_env_observation_includes_ela():
    pids = build_problem_ids({1}, dims=[2], instances=[1])
    env = TranslationEnv(
        "PSO",
        "CMAES",
        pids,
        IOHSuite(),
        fe_multiplier=400,
        n_switches=4,
        n_individuals=12,
        use_ela=True,
        seed=1,
    )
    obs, _ = env.reset()
    assert obs["ela"].shape == (ELA_DIM,)
    assert np.all(np.isfinite(obs["ela"]))
    # Drive a couple of steps with a lossy action; ELA stays well-formed and the
    # accumulated history grows past the minimum so features become non-trivial.
    saw_nonzero = np.any(obs["ela"] != 0.0)
    for _ in range(3):
        n = obs["native"]
        obs, *_ = env.step(
            {"x": n["x"], "y": n["y"], "best_x": n["best_x"], "best_y": n["best_y"]}
        )
        assert obs["ela"].shape == (ELA_DIM,)
        saw_nonzero = saw_nonzero or np.any(obs["ela"] != 0.0)
    assert saw_nonzero


def test_env_ela_disabled_returns_zeros():
    pids = build_problem_ids({1}, dims=[2], instances=[1])
    env = TranslationEnv(
        "PSO",
        "CMAES",
        pids,
        IOHSuite(),
        fe_multiplier=400,
        n_switches=4,
        n_individuals=12,
        use_ela=False,
        seed=1,
    )
    obs, _ = env.reset()
    assert np.array_equal(obs["ela"], np.zeros(ELA_DIM, dtype=np.float32))
