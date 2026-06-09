"""Sanity: the bidirectional model can drive cycle + reconstruction loss down
on a small fixed set of states (it is expressive enough to represent identity)."""

import torch

from cat.data.dataset import StateDataset
from cat.models.translator import TranslatorPair
from cat.train_loop import TrainConfig, train
from cat.state.canonical import CanonicalState


def _states(algo, count, N, D, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        pos = torch.randn(N, D, generator=g)
        vals = torch.randn(N, generator=g)
        best_x = torch.randn(D, generator=g)
        best_y = torch.randn((), generator=g)
        if algo == "PSO":
            specific = {
                "velocity": 0.3 * torch.randn(N, D, generator=g),
                "pbest_x": torch.randn(N, D, generator=g),
            }
        else:
            A = torch.randn(D, D, generator=g)
            specific = {
                "mean": torch.randn(D, generator=g),
                "sigma_cov": A @ A.t() + torch.eye(D),
                "p_c": torch.randn(D, generator=g),
                "p_s": torch.randn(D, generator=g),
            }
        out.append(CanonicalState(algo, pos, vals, best_x, best_y, specific))
    return out


def test_overfit_drives_cycle_and_recon_down():
    torch.manual_seed(0)  # deterministic weight init (independent of test order)
    states = _states("PSO", 12, N=10, D=3, seed=1) + _states(
        "CMAES", 12, N=8, D=3, seed=2
    )
    ds = StateDataset(states)
    pair = TranslatorPair("PSO", "CMAES", hidden=48)
    cfg = TrainConfig(
        epochs=60, batch_size=8, lr=2e-3, w_utility=0.0, device="cpu", log_every=0
    )
    log = train(pair, ds, cfg)
    first = log.history[0]
    last = log.history[-1]
    # Both anchoring losses should drop substantially.
    assert last["cycle"] < 0.5 * first["cycle"]
    assert last["recon"] < 0.5 * first["recon"]
    assert last["recon"] < 0.5  # reconstruction is an autoencoder identity
