"""RunningMeanStd correctness + PPO runs with normalization on and off."""

import numpy as np
import torch

from cat.models.norm import NormContext, _CLIP
from cat.rl.env import TranslationEnv
from cat.rl.normalize import RunningMeanStd
from cat.rl.policy import ActorCritic
from cat.rl.ppo import PPOConfig, train_ppo
from cat.state.canonical import CanonicalState
from cat.suite import IOHSuite, build_problem_ids


def test_converged_population_keeps_critic_bounded():
    """A converged population (near-identical values -> val_std at its 1e-6 floor)
    must not blow up the encoder's best_y feature or the critic value. Regression
    for the value-loss spikes seen with strongly-converging locals like BOBYQA."""
    D, N = 4, 10
    A = torch.randn(1, D, D)
    s = CanonicalState(
        "BOBYQA",
        torch.randn(1, N, D) * 1e-3,
        torch.full((1, N), 100.0) + torch.randn(1, N) * 1e-7,  # ~identical values
        torch.zeros(1, D),
        torch.tensor([99.9]),
        {
            "grad": torch.randn(1, D),
            "hessian": A @ A.transpose(-2, -1) + torch.eye(D),
            "radius": torch.tensor([0.5]),
        },
    )
    ctx = NormContext.from_shared(s.positions, s.values)
    # The unbounded feature was best_y_std; it is now clamped to the shared band.
    best_y_std = (s.best_y - ctx.val_mean) / ctx.val_std
    assert best_y_std.abs().item() > 1e4  # the raw ratio really is pathological
    assert (best_y_std.clamp(-_CLIP, _CLIP)).abs().item() == _CLIP  # ...but clamped

    torch.manual_seed(0)
    ac = ActorCritic("PSO", "BOBYQA", hidden=32, n_layers=1).eval()
    with torch.no_grad():
        z = ac.critic_encoders["BOBYQA"](s, ctx)
        value = ac.critic(torch.cat([z.global_vec, torch.zeros(1, 2)], dim=-1))
    assert torch.isfinite(z.global_vec).all()
    assert value.abs().item() < 100.0  # no runaway value prediction (was ~477)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(500, 4))
    rms = RunningMeanStd((4,))
    for chunk in np.array_split(data, 7):  # streamed in batches
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-6)
    assert np.allclose(rms.var, data.var(0), atol=1e-4)


def test_state_dict_roundtrip():
    rms = RunningMeanStd((2,))
    rms.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
    other = RunningMeanStd((2,))
    other.load_state_dict(rms.state_dict())
    assert np.allclose(other.mean, rms.mean) and np.allclose(other.var, rms.var)


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
            seed=0,
        )
    ]


def _run(norm_reward, norm_obs):
    ac = ActorCritic("PSO", "CMAES", hidden=32, n_layers=1)
    cfg = PPOConfig(
        updates=2,
        rollout_steps=48,
        ppo_epochs=2,
        minibatch_size=16,
        norm_reward=norm_reward,
        norm_obs=norm_obs,
        seed=1,
        log_every=0,
    )
    return train_ppo(ac, _envs(), cfg)


def test_ppo_with_normalization():
    log = _run(norm_reward=True, norm_obs=True)
    assert len(log.history) == 2
    assert log.ret_rms is not None and log.obs_rms is not None
    assert log.ret_rms.count > 1e-4  # statistics actually accumulated
    for row in log.history:
        for k in ("policy_loss", "value_loss", "cycle", "mean_return"):
            assert np.isfinite(row[k])


def test_ppo_without_normalization():
    log = _run(norm_reward=False, norm_obs=False)
    assert log.ret_rms is None and log.obs_rms is None
    assert all(np.isfinite(r["value_loss"]) for r in log.history)
