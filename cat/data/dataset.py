"""Dataset of optimizer states, batched homogeneously by (algo, D, N).

States of different problem dimension ``D`` or population size ``N`` cannot be
stacked into one tensor, so batches are drawn from a single (algo, D, N) group.
The training loop receives one batched ``CanonicalState`` at a time, tagged by
its algorithm, and runs the appropriate translation cycle on it.
"""

from __future__ import annotations

import pickle
from collections import defaultdict

import torch

from cat.data.collector import Snapshot
from cat.state.canonical import CanonicalState, to_canonical


def collate(states: list[CanonicalState]) -> CanonicalState:
    """Stack single states (same algo, D, N) into one batched CanonicalState."""
    algo = states[0].algo
    return CanonicalState(
        algo=algo,
        positions=torch.stack([s.positions for s in states]),
        values=torch.stack([s.values for s in states]),
        best_x=torch.stack([s.best_x for s in states]),
        best_y=torch.stack([s.best_y for s in states]),
        specific={
            k: torch.stack([s.specific[k] for s in states]) for k in states[0].specific
        },
    )


class StateDataset:
    """Holds canonical states grouped by (algo, D, N) for homogeneous batching."""

    def __init__(self, states: list[CanonicalState]):
        self.states = states
        self.groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for i, s in enumerate(states):
            self.groups[(s.algo, s.d, s.n)].append(i)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_snapshots(cls, snapshots: list[Snapshot], device="cpu") -> "StateDataset":
        states = [to_canonical(s.algo, s.native, device=device) for s in snapshots]
        return cls(states)

    # -- introspection ---------------------------------------------------- #

    def summary(self) -> dict:
        out: dict = {}
        for (algo, d, n), idxs in sorted(self.groups.items()):
            out.setdefault(algo, {})[f"d{d}_n{n}"] = len(idxs)
        return out

    def count(self, algo: str) -> int:
        return sum(len(v) for (a, _, _), v in self.groups.items() if a == algo)

    def __len__(self) -> int:
        return len(self.states)

    # -- iteration -------------------------------------------------------- #

    def iter_batches(self, batch_size: int, generator: torch.Generator | None = None):
        """Yield batched CanonicalStates in random group/order each call (one epoch)."""
        group_keys = list(self.groups.keys())
        perm = torch.randperm(len(group_keys), generator=generator).tolist()
        for gi in perm:
            idxs = self.groups[group_keys[gi]]
            order = torch.randperm(len(idxs), generator=generator).tolist()
            shuffled = [idxs[o] for o in order]
            for start in range(0, len(shuffled), batch_size):
                chunk = shuffled[start : start + batch_size]
                yield collate([self.states[i] for i in chunk])


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #


def save_snapshots(snapshots: list[Snapshot], path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(snapshots, fh)


def load_snapshots(path: str) -> list[Snapshot]:
    with open(path, "rb") as fh:
        return pickle.load(fh)
