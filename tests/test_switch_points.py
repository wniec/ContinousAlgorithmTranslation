"""Log-uniform switch-point sampling: fixed count, randomized early-biased
positions, valid spacing."""

import numpy as np

from cat.optimizers.base import sample_switch_points


def test_count_monotonic_and_finishes_on_budget():
    rng = np.random.default_rng(0)
    max_fe, pop = 5000, 12
    pts = sample_switch_points(8, max_fe, pop, rng)
    assert len(pts) == 8
    assert pts[-1] == max_fe
    assert np.all(np.diff(pts) >= pop)  # every segment can evaluate a population


def test_independent_per_call():
    a = sample_switch_points(8, 5000, 12, np.random.default_rng(1))
    b = sample_switch_points(8, 5000, 12, np.random.default_rng(2))
    assert not np.array_equal(a, b)  # different streams -> different positions


def test_single_switch_is_budget():
    pts = sample_switch_points(1, 5000, 12, np.random.default_rng(0))
    assert pts.tolist() == [5000]


def test_log_uniform_is_early_biased():
    """More interior switch points fall in the first half of the budget than the
    second, reflecting the log-FE distribution."""
    rng = np.random.default_rng(3)
    max_fe = 10_000
    interior = []
    for _ in range(400):
        pts = sample_switch_points(3, max_fe, 1, rng)
        interior.extend(pts[:-1].tolist())  # drop the final max_fe target
    interior = np.array(interior)
    first_half = (interior < max_fe / 2).mean()
    assert first_half > 0.6  # log-uniform concentrates mass early
