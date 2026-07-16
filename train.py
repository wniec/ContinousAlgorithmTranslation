"""Train a bidirectional state translator between two metaheuristics.

    python train.py PSO CMAES [options]

Runs many COCO-BBOB problems while switching between the two named algorithms,
snapshots their internal states, and trains a network so that — with the
population held fixed — translating A->B->A perturbs A's own parameters
minimally (cycle-consistency) while the translated state remains a useful
warm-start for the target optimizer (utility).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cat.data.collector import collect_dataset
from cat.data.dataset import StateDataset, load_snapshots, save_snapshots
from cat.models.translator import TranslatorPair
from cat.optimizers.portfolio import PORTFOLIO
from cat.tracking import WandbLogger
from cat.suite.bbob_splits import (
    ALL_FUNCTIONS,
    EASY_TRAIN_FUNCTIONS,
    build_problem_ids,
)
from cat.train_loop import TrainConfig, resolve_device, save_translator, train


def build_train_ids(
    split: str, dims: list[int], instances: list[int] | None
) -> list[str]:
    if split == "easy":
        funcs = EASY_TRAIN_FUNCTIONS
    elif split == "all":
        funcs = ALL_FUNCTIONS
    else:
        raise ValueError(f"unknown split {split!r}")
    return build_problem_ids(funcs, dims, instances)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algo_a", choices=sorted(PORTFOLIO))
    p.add_argument("algo_b", choices=sorted(PORTFOLIO))
    # problem set / collection
    p.add_argument("-d", "--dims", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--split", choices=["easy", "all"], default="easy")
    p.add_argument("--fe-multiplier", type=int, default=2000)
    p.add_argument(
        "--n-switches",
        type=int,
        default=8,
        help="segments per episode; boundaries sampled log-uniformly over the FE budget",
    )
    p.add_argument("--schedule", choices=["alternate", "random"], default="alternate")
    p.add_argument("--n-individuals", type=int, default=None)
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
    # training
    p.add_argument("-E", "--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--w-cycle", type=float, default=1.0)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-utility", type=float, default=0.1)
    p.add_argument(
        "--cycle-mode",
        choices=["field", "latent"],
        default="field",
        help="score the A->B->A round trip on decoded fields or re-encoded latents",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' uses CUDA when available else CPU; or pass cuda / cuda:0 / cpu / mps",
    )
    p.add_argument("--seed", type=int, default=42)
    # io
    p.add_argument(
        "--cache", default=None, help="path to cache/load collected snapshots"
    )
    p.add_argument("--out", default=None, help="output checkpoint path")
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
    print(f"[device] using {args.device}")

    if args.cache and os.path.exists(args.cache):
        print(f"loading cached snapshots from {args.cache}")
        snapshots = load_snapshots(args.cache)
    else:
        ids = build_train_ids(args.split, args.dims, args.instances)
        print(
            f"collecting states: {args.algo_a}<->{args.algo_b} over {len(ids)} problems "
            f"(dims={args.dims}, split={args.split}, schedule={args.schedule})"
        )
        snapshots = collect_dataset(
            args.algo_a,
            args.algo_b,
            ids,
            fe_multiplier=args.fe_multiplier,
            n_switches=args.n_switches,
            schedule=args.schedule,
            n_individuals=args.n_individuals,
            n_individuals_a=args.n_individuals_a,
            n_individuals_b=args.n_individuals_b,
            seed=args.seed,
        )
        if args.cache:
            save_snapshots(snapshots, args.cache)
            print(f"cached {len(snapshots)} snapshots to {args.cache}")

    dataset = StateDataset.from_snapshots(snapshots)
    print(f"dataset: {len(dataset)} states -> {dataset.summary()}")
    for algo in (args.algo_a, args.algo_b):
        if dataset.count(algo) == 0:
            raise SystemExit(
                f"no states collected for {algo}; check schedule / problem set"
            )

    logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"sup_{args.algo_a}_{args.algo_b}",
        group=f"{args.algo_a}_{args.algo_b}",
        config={**vars(args), "trainer": "supervised"},
        x_axis="epoch",
    )

    pair = TranslatorPair(
        args.algo_a, args.algo_b, hidden=args.hidden, n_layers=args.n_layers
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_cycle=args.w_cycle,
        w_recon=args.w_recon,
        w_utility=args.w_utility,
        cycle_mode=args.cycle_mode,
        device=args.device,
        seed=args.seed,
    )
    train(pair, dataset, cfg, log_fn=logger.log)
    logger.finish()

    out = args.out or os.path.join("models", f"{args.algo_a}_{args.algo_b}.pt")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_translator(pair, out, cfg)
    print(f"saved translator to {out}")


if __name__ == "__main__":
    main()
