"""PPO loop runs end-to-end on a tiny problem set, produces finite metrics, and
the sampled covariance action stays PSD."""

import math

import numpy as np

from cat.rl.env import TranslationEnv
from cat.rl.policy import ActorCritic
from cat.rl.ppo import PPOConfig, train_ppo
from cat.suite import IOHSuite, build_problem_ids


def _envs():
    pids = build_problem_ids({1, 2}, dims=[2], instances=[1])
    return [
        TranslationEnv(
            "PSO",
            "CMAES",
            pids,
            IOHSuite(),
            fe_multiplier=200,
            n_switches=3,
            n_individuals=12,
            use_ela=False,
            seed=0,
        )
    ]


def test_ppo_runs_and_metrics_finite():
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    cfg = PPOConfig(
        updates=2,
        rollout_steps=48,
        ppo_epochs=2,
        minibatch_size=16,
        seed=1,
        log_every=0,
    )
    log = train_ppo(ac, _envs(), cfg)
    assert len(log.history) == 2
    for row in log.history:
        for k in ("policy_loss", "value_loss", "policy_std", "cycle", "mean_return"):
            assert math.isfinite(row[k]), f"{k} not finite"
        assert row["n_transitions"] >= cfg.rollout_steps


def test_sampled_covariance_action_is_psd():
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    env = _envs()[0]
    obs, _ = env.reset()
    assert obs["target_algo"] == "CMAES"
    step = ac.act(obs)  # stochastic sample
    cm = step.native["cm"]
    eig = np.linalg.eigvalsh(0.5 * (cm + cm.T))
    assert eig.min() > -1e-6
    assert step.native["sigma"] > 0


def test_parameters_change_after_update():
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    before = [p.detach().clone() for p in ac.parameters()]
    train_ppo(
        ac,
        _envs(),
        PPOConfig(
            updates=1,
            rollout_steps=48,
            ppo_epochs=2,
            minibatch_size=16,
            seed=2,
            log_every=0,
        ),
    )
    after = list(ac.parameters())
    assert any((a - b).abs().sum().item() > 0 for a, b in zip(after, before))
