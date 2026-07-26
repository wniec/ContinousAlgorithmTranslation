"""Shared CLI argument groups and env/logger construction for train_ppo.py and
train_td3.py — the two trainers train the same ``TranslationEnv`` (see
``cat/rl/env.py``), so problem-set/episode configuration, W&B logging, and
device/seed/output plumbing are identical between them; only the network and
algorithm hyperparameters differ (kept in each script)."""

from __future__ import annotations

import argparse

from cat.optimizers.portfolio import PORTFOLIO
from cat.rl.env import TranslationEnv
from cat.suite import IOHSuite
from cat.suite.bbob_splits import ALL_FUNCTIONS, EASY_TRAIN_FUNCTIONS, build_problem_ids
from cat.tracking import WandbLogger


def add_algo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))


def add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--split", choices=["easy", "all"], default="easy")
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument("--n-switches", type=int, default=6)
    p.add_argument(
        "--n-individuals",
        type=int,
        default=None,
        help="shared population size for both algos; omit (with -a/-b also "
        "unset) to let each algorithm use its own built-in default",
    )
    p.add_argument(
        "--n-individuals-a",
        type=int,
        default=None,
        help="override --n-individuals for algo_a only",
    )
    p.add_argument(
        "--n-individuals-b",
        type=int,
        default=None,
        help="override --n-individuals for algo_b only",
    )
    p.add_argument(
        "--switch-cdb",
        type=float,
        default=1.0,
        help="base of the switch-point log-FE warp: 1.0 (default) = plain "
        "log-uniform; >1 concentrates switches earlier; 0<cdb<1 later.",
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


def add_logging_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wandb", action="store_true", help="log metrics to Weights & Biases"
    )
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' uses CUDA when available else CPU; or pass cuda / cuda:0 / cpu / mps",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)


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
                n_individuals_a=args.n_individuals_a,
                n_individuals_b=args.n_individuals_b,
                reward_mode=args.reward_mode,
                switch_cdb=args.switch_cdb,
                seed=args.seed * 100 + dim,
            )
        )
    return envs


def build_logger(args, trainer: str) -> WandbLogger:
    return WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"{trainer}_{args.algo_a}_{args.algo_b}",
        group=f"{args.algo_a}_{args.algo_b}",
        config={**vars(args), "trainer": trainer},
        x_axis="update",
    )
