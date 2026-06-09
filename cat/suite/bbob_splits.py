"""BBOB problem-set constants and train/test split helpers.

Vendored (and trimmed) from DynamicAlgorithmSelection2 (das/env/bbob_splits.py).
"""

from itertools import product

import numpy as np

ALL_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FUNCTIONS = set(range(1, 25))
INSTANCE_IDS = [1, 2, 3, 4, 5, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
EASY_TRAIN_FUNCTIONS = {4, *range(6, 15), 18, 19, 20, 22, 23, 24}


def build_problem_ids(
    functions: set[int],
    dims: list[int],
    instances: list[int] | None = None,
) -> list[str]:
    insts = instances if instances is not None else INSTANCE_IDS
    return [
        f"bbob_f{f:03d}_i{i:02d}_d{d:02d}"
        for i, f, d in product(insts, sorted(functions), dims)
    ]


def get_train_test_split(mode: str, dims: list[int]) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) for the given split mode and dimensions.

    Modes:
      easy   - train on easy BBOB functions, test on hard ones
      hard   - inverse of easy
      random - random 2/3 / 1/3 split on all problem IDs
    """
    if mode == "easy":
        return (
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
        )
    if mode == "hard":
        return (
            build_problem_ids(ALL_FUNCTIONS - EASY_TRAIN_FUNCTIONS, dims),
            build_problem_ids(EASY_TRAIN_FUNCTIONS, dims),
        )
    # random 2/3 - 1/3 split
    all_ids = build_problem_ids(ALL_FUNCTIONS, dims)
    rng = np.random.default_rng()
    rng.shuffle(all_ids)
    split = 2 * len(all_ids) // 3
    return all_ids[:split], all_ids[split:]
