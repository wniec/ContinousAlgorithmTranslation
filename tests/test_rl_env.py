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
    assert all(np.isfinite(r) for r in rewards)


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


def test_noswitch_reward_is_signed_over_source_continuing():
    """noswitch reward = log_scale(translated_impr) - log_scale(source_continuing_impr):
    a *signed* difference of logs against the source-continuing reference, NOT the
    old symmetric -|Δ| penalty. It must be finite, bounded by the log-scale range,
    and able to go strictly negative (a lossy switch usually underperforms the
    source simply continuing)."""
    from cat.rl.env import _log_scale

    env = TranslationEnv(
        "PSO",
        "CMAES",
        build_problem_ids({1, 2}, dims=[2], instances=[1]),
        IOHSuite(),
        fe_multiplier=300,
        n_switches=4,
        n_individuals=12,
        reward_mode="noswitch",
        seed=5,
    )
    bound = _log_scale(1.0)  # each log-scale term is in [0, bound]
    obs, _ = env.reset()
    done = False
    seen_negative = False
    while not done:
        obs, r, term, trunc, _ = env.step(_lossy_action(obs))
        assert np.isfinite(r)
        assert -bound - 1e-9 <= r <= bound + 1e-9
        seen_negative = seen_negative or r < -1e-9
        done = term or trunc
    # A lossy switch generally underperforms the source continuing at some step.
    assert seen_negative


def test_mixed_reward_is_noswitch_plus_relative():
    """'mixed' mode is defined as the sum of the noswitch and relative rewards.
    Driving three identically-seeded envs with the same lossy action must give,
    at every step, mixed == noswitch + relative (the counterfactual optimizers
    are seeded deterministically per segment, so the terms match exactly)."""

    def _make(mode):
        return TranslationEnv(
            "PSO",
            "CMAES",
            build_problem_ids({1, 2}, dims=[2], instances=[1]),
            IOHSuite(),
            fe_multiplier=300,
            n_switches=4,
            n_individuals=12,
            reward_mode=mode,
            seed=7,
        )

    envs = {m: _make(m) for m in ("noswitch", "relative", "mixed")}
    obs = {m: e.reset()[0] for m, e in envs.items()}
    done = False
    saw_step = False
    while not done:
        rs, infos = {}, {}
        for m, e in envs.items():
            obs[m], rs[m], term, trunc, infos[m] = e.step(_lossy_action(obs[m]))
        assert np.isclose(rs["mixed"], rs["noswitch"] + rs["relative"], atol=1e-9)
        # mixed mode exposes the two additive components in info, and they must
        # sum to the reward and match each single-baseline mode's reward.
        parts = infos["mixed"]["reward_parts"]
        assert np.isclose(parts["mixed_noswitch"], rs["noswitch"], atol=1e-9)
        assert np.isclose(parts["mixed_relative"], rs["relative"], atol=1e-9)
        assert np.isclose(
            parts["mixed_noswitch"] + parts["mixed_relative"], rs["mixed"], atol=1e-9
        )
        # single-baseline modes carry no reward_parts.
        assert "reward_parts" not in infos["noswitch"]
        saw_step = True
        done = term or trunc
    assert saw_step


def test_log_scaled_reward_properties():
    from cat.rl.env import log_scaled_improvement, _log_scale

    # no improvement -> 0; improvement -> positive and monotonic in improvement.
    assert log_scaled_improvement(5.0, 5.0, 4.0) == 0.0
    assert log_scaled_improvement(5.0, 6.0, 4.0) == 0.0  # worse -> 0
    r_small = log_scaled_improvement(5.0, 4.9, 4.0)
    r_big = log_scaled_improvement(5.0, 1.0, 4.0)
    assert 0.0 < r_small < r_big
    # full-range improvement saturates at log(1 + 1/eps).
    import math

    from cat.rl.env import _REWARD_EPS

    assert math.isclose(_log_scale(1.0), math.log(1 + 1 / _REWARD_EPS), rel_tol=1e-6)


def test_range_uses_distance_to_optimum():
    env = _env(seed=11)
    env.reset()
    optimum = env._problem.optimum
    # range = best-so-far after warmup minus the global optimum (>= 0).
    assert env._range >= 0.0
    assert abs(env._range - (env._best_y - optimum)) < 1e-6
    assert env._best_y >= optimum - 1e-9


def test_scaled_improvement_bounds():
    assert scaled_improvement(10.0, 5.0, 5.0) == 1.0
    assert scaled_improvement(5.0, 5.0, 5.0) == 0.0
    assert scaled_improvement(5.0, 6.0, 5.0) == 0.0  # no improvement -> 0
    assert 0.0 < scaled_improvement(10.0, 9.0, 5.0) < 1.0


def _env_independent_sizes(seed=0, n_switches=4, n_a=8, n_b=20):
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return TranslationEnv(
        "PSO",
        "CMAES",
        pids,
        IOHSuite(),
        fe_multiplier=200,
        n_switches=n_switches,
        n_individuals_a=n_a,
        n_individuals_b=n_b,
        seed=seed,
    )


def test_target_n_reflects_each_algos_own_population_size():
    env = _env_independent_sizes(n_a=8, n_b=20)
    obs, _ = env.reset()
    assert obs["source_algo"] == "PSO" and obs["target_algo"] == "CMAES"
    assert obs["target_n"] == 20
    obs, r, term, trunc, _ = env.step(_lossy_action(obs))
    assert obs["target_algo"] == "PSO"
    assert obs["target_n"] == 8


def test_episode_runs_with_independent_population_sizes():
    """A lossy hand-off doesn't resize (it just carries x/y verbatim), so a
    smaller-than-target carried population makes the target optimizer's own
    set_data/initialize discard it and cold-init instead — exactly the same
    degenerate-source case the real PPO rollout loop already guards against
    with source_has_full_state() (see collect_rollout in cat/rl/ppo.py).
    Drive the env the same defensive way and confirm every completed step
    still yields a finite reward."""
    env = _env_independent_sizes(n_switches=4, n_a=8, n_b=20)
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        if not env.source_has_full_state():
            break
        obs, r, term, trunc, _ = env.step(_lossy_action(obs))
        rewards.append(r)
        done = term or trunc
    assert rewards  # at least the first (PSO(8) -> CMAES(20)) switch completed
    assert all(np.isfinite(r) for r in rewards)


def test_policy_action_native_matches_target_n():
    """A real translator's assembled warm-start must actually be resized to
    the target's own population size (not just the source's)."""
    from cat.rl.policy import ActorCritic

    env = _env_independent_sizes(n_a=8, n_b=20)
    ac = ActorCritic("PSO", "CMAES", hidden=16, n_layers=1, resample_seed=0)
    obs, _ = env.reset()
    assert obs["native"]["x"].shape[0] == 8  # source (PSO) population
    step = ac.act(obs)
    assert step.native["x"].shape[0] == obs["target_n"] == 20

    obs, r, term, trunc, _ = env.step(step.native)
    assert obs["target_algo"] == "PSO"
    step2 = ac.act(obs)
    assert step2.native["x"].shape[0] == obs["target_n"] == 8
