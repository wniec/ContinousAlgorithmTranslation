"""Train the state translator with TD3 (off-policy reinforcement learning).

    python train_td3.py PSO CMAES [options]

Same environment, same continuous action (the target optimizer's full decoded
warm-start) and the same set-equivariant translator as train_ppo.py — but
trained off-policy: a deterministic actor (the translator's mean action, no
learned std) explores via Gaussian noise added in action space, transitions go
into a replay buffer, and twin critics (Q1, Q2) are trained via Double-Q-style
target computation with delayed, Polyak-averaged target networks (see
cat/rl/td3.py for the full design rationale, in particular how the critic
handles the action's variable dimensionality). Shared CLI/env-building logic
lives in rl_common.py.

Saves a checkpoint compatible with evaluate.py (`models/<A>_<B>_td3.pt`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cat.rl.td3 import TD3Actor, TD3Config, TwinQCritic, train_td3
from cat.train_loop import resolve_device, translator_meta
from rl_common import (
    add_algo_args,
    add_env_args,
    add_logging_args,
    add_runtime_args,
    build_envs,
    build_logger,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_algo_args(p)
    add_env_args(p)
    # network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--cov-rank", type=int, default=4)
    p.add_argument(
        "--resample-seed",
        type=int,
        default=None,
        help="seed for the population-resizing jitter (when n-individuals-a/b differ)",
    )
    # TD3
    p.add_argument("--updates", type=int, default=500)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=1024,
        help="env steps collected (actor + exploration noise) per update",
    )
    p.add_argument(
        "--gradient-steps",
        type=int,
        default=64,
        help="critic updates performed per update (actor/targets updated every "
        "--policy-freq of these)",
    )
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument(
        "--buffer-capacity",
        type=int,
        default=20_000,
        help="replay buffer size (FIFO once exceeded)",
    )
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--actor-lr", type=float, default=8e-5)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument(
        "--expl-noise",
        type=float,
        default=0.1,
        help="std of Gaussian exploration noise added to the actor's action "
        "(in the translator's normalized action-coordinate space) during rollout",
    )
    p.add_argument(
        "--policy-noise",
        type=float,
        default=0.2,
        help="std of the target policy smoothing noise added to the target "
        "actor's action when computing the TD-target",
    )
    p.add_argument(
        "--noise-clip",
        type=float,
        default=0.5,
        help="clip range for the target policy smoothing noise",
    )
    p.add_argument(
        "--policy-freq",
        type=int,
        default=2,
        help="delayed actor/target-network update frequency, in critic gradient steps",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Polyak averaging coefficient for the target actor/critic networks",
    )
    p.add_argument("--lambda-cycle", type=float, default=0.0)
    p.add_argument(
        "--cycle-mode",
        choices=["field", "latent"],
        default="field",
        help="score the A->B->A cycle penalty on decoded fields or re-encoded latents",
    )
    p.add_argument(
        "--no-norm-reward",
        dest="norm_reward",
        action="store_false",
        help="disable reward normalization (running std of the discounted return)",
    )
    p.add_argument(
        "--no-norm-obs",
        dest="norm_obs",
        action="store_false",
        help="disable observation (context) normalization",
    )
    add_runtime_args(p)
    add_logging_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)

    envs = build_envs(args)
    print(
        f"TD3 training {args.algo_a}<->{args.algo_b} | dims={args.dims} "
        f"| {len(envs)} env(s) | n_switches={args.n_switches} | device={args.device}"
    )

    actor = TD3Actor(
        args.algo_a,
        args.algo_b,
        hidden=args.hidden,
        n_layers=args.n_layers,
        cov_rank=args.cov_rank,
        resample_seed=args.resample_seed,
    )
    critic = TwinQCritic(
        args.algo_a, args.algo_b, hidden=args.hidden, n_layers=args.n_layers
    )
    cfg = TD3Config(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        gradient_steps=args.gradient_steps,
        minibatch_size=args.minibatch_size,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        expl_noise=args.expl_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        tau=args.tau,
        lambda_cycle=args.lambda_cycle,
        cycle_mode=args.cycle_mode,
        norm_reward=args.norm_reward,
        norm_obs=args.norm_obs,
        device=args.device,
        seed=args.seed,
    )
    logger = build_logger(args, "td3")
    td3_log = train_td3(actor, critic, envs, cfg, log_fn=logger.log)
    logger.finish()

    out = args.out or os.path.join("models", f"{args.algo_a}_{args.algo_b}_td3.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Save the actor's TranslatorPair under "state_dict" so evaluate.py /
    # load_translator can consume it directly (same architecture PPO saves);
    # keep the critic and running normalizer statistics too (for resuming).
    ckpt = {
        "state_dict": actor.translator.state_dict(),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        **translator_meta(actor.translator),
    }
    if td3_log.ret_rms is not None:
        ckpt["ret_rms"] = td3_log.ret_rms.state_dict()
    if td3_log.obs_rms is not None:
        ckpt["obs_rms"] = td3_log.obs_rms.state_dict()
    torch.save(ckpt, out)
    print(f"saved TD3 translator to {out}")


if __name__ == "__main__":
    main()
