"""Train the state translator with PPO (reinforcement learning).

    python train_rl.py PSO CMAES [options]

Each episode is a BBOB run that switches between the two named algorithms at
log-uniformly sampled points; at every switch the policy (the set-equivariant
translator) maps the source optimizer's internal state into the target's. The
reward is the true downstream best-so-far improvement over the next segment, and
the A->B->A cycle drift enters as a differentiable penalty — so the network
learns hand-offs that are both *useful* and *reversible*, optimizing the real
(non-differentiable) objective rather than the supervised proxy in train.py.

Saves a checkpoint compatible with evaluate.py (`models/<A>_<B>_rl.pt`).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cat.rl.env import TranslationEnv
from cat.rl.policy import ActorCritic
from cat.rl.ppo import PPOConfig, train_ppo
from cat.optimizers.portfolio import PORTFOLIO
from cat.tracking import WandbLogger
from cat.suite import IOHSuite
from cat.suite.bbob_splits import ALL_FUNCTIONS, EASY_TRAIN_FUNCTIONS, build_problem_ids
from cat.train_loop import resolve_device, translator_meta


def build_envs(args) -> list[TranslationEnv]:
    funcs = EASY_TRAIN_FUNCTIONS if args.split == "easy" else ALL_FUNCTIONS
    suite = IOHSuite()
    envs = []
    for dim in args.dims:
        ids = build_problem_ids(funcs, [dim], args.instances)
        envs.append(
            TranslationEnv(
                args.algo_a,
                args.algo_b,
                ids,
                suite,
                fe_multiplier=args.fe_multiplier,
                n_switches=args.n_switches,
                n_individuals=args.n_individuals,
                reward_mode=args.reward_mode,
                use_ela=args.use_ela,
                seed=args.seed * 100 + dim,
            )
        )
    return envs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))
    # problem set / episode
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--split", choices=["easy", "all"], default="easy")
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-switches", type=int, default=6)
    p.add_argument("--n-individuals", type=int, default=12)
    p.add_argument(
        "--no-ela",
        dest="use_ela",
        action="store_false",
        help="disable ELA landscape features in the critic observation (faster)",
    )
    p.add_argument(
        "--reward-mode",
        choices=["noswitch", "absolute", "relative"],
        default="noswitch",
        help="'noswitch' (default) rewards matching the no-switch counterfactual "
        "(the source optimizer continuing); 'absolute' rewards raw improvement; "
        "'relative' rewards improvement over the lossy default. noswitch/relative "
        "run an extra counterfactual optimizer per step (~2x cost).",
    )
    # network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--cov-rank", type=int, default=4)
    p.add_argument("--log-std-init", type=float, default=1.0)
    # PPO
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.05)
    p.add_argument("--lambda-cycle", type=float, default=5.0)
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
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' uses CUDA when available else CPU; or pass cuda / cuda:0 / cpu / mps",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    # logging
    p.add_argument(
        "--wandb", action="store_true", help="log metrics to Weights & Biases"
    )
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = resolve_device(args.device)

    envs = build_envs(args)
    print(
        f"RL training {args.algo_a}<->{args.algo_b} | dims={args.dims} "
        f"| {len(envs)} env(s) | n_switches={args.n_switches} | device={args.device}"
    )

    ac = ActorCritic(
        args.algo_a,
        args.algo_b,
        hidden=args.hidden,
        n_layers=args.n_layers,
        cov_rank=args.cov_rank,
        log_std_init=args.log_std_init,
    )
    cfg = PPOConfig(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        lambda_cycle=args.lambda_cycle,
        lr=args.lr,
        norm_reward=args.norm_reward,
        norm_obs=args.norm_obs,
        device=args.device,
        seed=args.seed,
    )
    logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"ppo_{args.algo_a}_{args.algo_b}",
        group=f"{args.algo_a}_{args.algo_b}",
        config={**vars(args), "trainer": "ppo"},
        x_axis="update",
    )
    ppo_log = train_ppo(ac, envs, cfg, log_fn=logger.log)
    logger.finish()

    out = args.out or os.path.join("models", f"{args.algo_a}_{args.algo_b}_rl.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Save the actor's TranslatorPair under "state_dict" so evaluate.py /
    # load_translator can consume it directly; keep the full actor-critic and the
    # running normalizer statistics too (for resuming / reproducibility).
    ckpt = {
        "state_dict": ac.translator.state_dict(),
        "actor_critic_state_dict": ac.state_dict(),
        **translator_meta(ac.translator),
    }
    if ppo_log.ret_rms is not None:
        ckpt["ret_rms"] = ppo_log.ret_rms.state_dict()
    if ppo_log.obs_rms is not None:
        ckpt["obs_rms"] = ppo_log.obs_rms.state_dict()
    torch.save(ckpt, out)
    print(f"saved RL translator to {out}")


if __name__ == "__main__":
    main()
