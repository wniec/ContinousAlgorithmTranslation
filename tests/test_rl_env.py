"""TranslationEnv mechanics: episode length, observation structure, finite
rewards, and seed determinism. Driven with a trivial 'lossy' action (carry only
the shared population) so no policy is needed."""

import numpy as np

from cat.rl.env import TranslationEnv, scaled_improvement
from cat.suite import IOHSuite, build_problem_ids


def _env(seed=0, n_switches=4):
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return TranslationEnv(
        "PSO",
        "CMAES",
        pids,
        IOHSuite(),
        fe_multiplier=200,
        n_switches=n_switches,
        n_individuals=12,
        seed=seed,
    )


def _lossy_action(obs):
    n = obs["native"]
    return {"x": n["x"], "y": n["y"], "best_x": n["best_x"], "best_y": n["best_y"]}


def _run_episode(env):
    obs, info = env.reset()
    rewards = []
    steps = 0
    done = False
    while not done:
        assert obs["source_algo"] in ("PSO", "CMAES")
        assert obs["target_algo"] != obs["source_algo"]
        assert obs["context"].shape == (2,)
        obs, r, term, trunc, info = env.step(_lossy_action(obs))
        rewards.append(r)
        steps += 1
        done = term or trunc
    return rewards, steps


def test_episode_runs_n_switches_steps():
    env = _env(n_switches=4)
    rewards, steps = _run_episode(env)
    assert steps == 4
    assert all(np.isfinite(r) and 0.0 <= r <= 1.0 for r in rewards)


def test_targets_alternate():
    env = _env(n_switches=4)
    obs, _ = env.reset()
    seq = [obs["target_algo"]]
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(_lossy_action(obs))
        done = term or trunc
        if not done:
            seq.append(obs["target_algo"])
    # PSO warmup -> first target CMAES, then alternating.
    assert seq[0] == "CMAES"
    assert all(a != b for a, b in zip(seq, seq[1:]))


def test_seed_determinism():
    r1, _ = _run_episode(_env(seed=3))
    r2, _ = _run_episode(_env(seed=3))
    assert r1 == r2


def test_relative_reward_zero_for_lossy_action():
    """In 'relative' mode the reward is improvement *over* the lossy default, so
    driving the env with the lossy action itself must yield ~0 every step."""
    env = TranslationEnv(
        "PSO",
        "CMAES",
        build_problem_ids({1, 2}, dims=[2], instances=[1]),
        IOHSuite(),
        fe_multiplier=300,
        n_switches=4,
        n_individuals=12,
        reward_mode="relative",
        seed=5,
    )
    obs, _ = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(_lossy_action(obs))
        assert abs(r) < 1e-9
        done = term or trunc


def test_scaled_improvement_bounds():
    assert scaled_improvement(10.0, 5.0, 5.0) == 1.0
    assert scaled_improvement(5.0, 5.0, 5.0) == 0.0
    assert scaled_improvement(5.0, 6.0, 5.0) == 0.0  # no improvement -> 0
    assert 0.0 < scaled_improvement(10.0, 9.0, 5.0) < 1.0
