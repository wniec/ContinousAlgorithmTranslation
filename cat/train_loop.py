"""Training orchestration for the bidirectional state translator."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cat.data.dataset import StateDataset
from cat.losses import batch_losses
from cat.models.norm import NormContext
from cat.models.translator import TranslatorPair


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    w_cycle: float = 1.0
    w_recon: float = 1.0
    w_utility: float = 0.1
    device: str = "cpu"
    seed: int = 42
    log_every: int = 1


@dataclass
class TrainLog:
    history: list[dict] = field(default_factory=list)


def _ctx(state):
    return NormContext.from_shared(state.positions, state.values)


def train(
    pair: TranslatorPair, dataset: StateDataset, cfg: TrainConfig, log_fn=None
) -> TrainLog:
    """Train the translator. ``log_fn(row)`` is called once per epoch with the
    metric dict (used for Weights & Biases logging; optional)."""
    device = torch.device(cfg.device)
    pair.to(device)
    opt = torch.optim.Adam(pair.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    log = TrainLog()

    for epoch in range(cfg.epochs):
        pair.train()
        sums = {"cycle": 0.0, "recon": 0.0, "utility": 0.0, "total": 0.0}
        n_batches = 0
        for state in dataset.iter_batches(cfg.batch_size, generator=gen):
            state = state.to(device)
            ctx = _ctx(state)
            losses = batch_losses(pair, state, ctx)
            total = (
                cfg.w_cycle * losses["cycle"]
                + cfg.w_recon * losses["recon"]
                + cfg.w_utility * losses["utility"]
            )
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(pair.parameters(), 5.0)
            opt.step()

            for k in ("cycle", "recon", "utility"):
                sums[k] += float(losses[k].detach())
            sums["total"] += float(total.detach())
            n_batches += 1

        n_batches = max(n_batches, 1)
        row = {k: v / n_batches for k, v in sums.items()}
        row["epoch"] = epoch
        log.history.append(row)
        if log_fn is not None:
            log_fn(row)
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"epoch {epoch:3d} | total {row['total']:.4f} "
                f"| cycle {row['cycle']:.4f} | recon {row['recon']:.4f} "
                f"| utility {row['utility']:.4f}"
            )
    return log


def translator_meta(pair: TranslatorPair) -> dict:
    """Architecture hyperparameters needed to rebuild a TranslatorPair."""
    enc = pair.encoders[pair.algo_a]
    dec = pair.decoders[pair.algo_a]
    return {
        "algo_a": pair.algo_a,
        "algo_b": pair.algo_b,
        "hidden": enc.hidden,
        "n_layers": len(enc.attn),
        "cov_rank": dec.cov_rank,
    }


def save_translator(pair: TranslatorPair, path: str, cfg: TrainConfig) -> None:
    torch.save({"state_dict": pair.state_dict(), **translator_meta(pair)}, path)


def load_translator(path: str, device="cpu") -> TranslatorPair:
    """Rebuild a TranslatorPair from a supervised or RL checkpoint.

    RL checkpoints (train_rl.py) store the actor's TranslatorPair under the same
    ``state_dict`` key, so this loads both transparently.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pair = TranslatorPair(
        ckpt["algo_a"],
        ckpt["algo_b"],
        hidden=ckpt.get("hidden", 64),
        n_layers=ckpt.get("n_layers", 2),
        cov_rank=ckpt.get("cov_rank", 4),
    )
    pair.load_state_dict(ckpt["state_dict"])
    pair.to(device)
    pair.eval()
    return pair
