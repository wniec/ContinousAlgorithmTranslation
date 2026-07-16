"""State adapter round-trip: native -> canonical -> native preserves fields."""

import numpy as np
import pytest
import torch

from cat.state import to_canonical, from_canonical, get_spec
from cat.state.canonical import resample_population
from cat.state.schema import Struct


@pytest.mark.parametrize("algo", ["PSO", "CMAES"])
def test_to_canonical_shapes(algo, pso_native, cmaes_native):
    native = pso_native if algo == "PSO" else cmaes_native
    state = to_canonical(algo, native, device="cpu")
    D = state.d
    assert state.positions.shape == (state.n, D)
    assert state.best_x.shape == (D,)
    for f in get_spec(algo).specific:
        t = state.specific[f.name]
        if f.struct is Struct.PER_PARTICLE_PER_DIM:
            assert t.shape == (state.n, D)
        elif f.struct is Struct.PER_PARTICLE:
            assert t.shape == (state.n,)
        elif f.struct is Struct.PER_DIM:
            assert t.shape == (D,)
        elif f.struct is Struct.MATRIX:
            assert t.shape == (D, D)
        elif f.struct is Struct.SCALAR:
            assert t.shape == ()


@pytest.mark.parametrize("algo", ["PSO", "CMAES"])
def test_roundtrip_identity(algo, pso_native, cmaes_native):
    native = pso_native if algo == "PSO" else cmaes_native
    state = to_canonical(algo, native, device="cpu")
    back = from_canonical(state)
    state2 = to_canonical(algo, {**native, **back}, device="cpu")

    # Shared fields preserved.
    assert np.allclose(state.positions.numpy(), state2.positions.numpy(), atol=1e-5)
    assert np.allclose(state.best_x.numpy(), state2.best_x.numpy(), atol=1e-5)
    # Specific fields preserved.
    for name in get_spec(algo).field_names:
        a = state.specific[name].numpy()
        b = state2.specific[name].numpy()
        assert np.allclose(a, b, atol=1e-5), f"{algo}.{name} not preserved"


def test_cmaes_covariance_is_psd(cmaes_native):
    state = to_canonical("CMAES", cmaes_native, device="cpu")
    native = from_canonical(state)
    cm = native["cm"]
    eigvals = np.linalg.eigvalsh(0.5 * (cm + cm.T))
    assert eigvals.min() > -1e-8
    # Derived sampling basis is present so the covariance takes effect immediately.
    assert "e_ve" in native and "e_va" in native and "d" in native


def test_resample_population_noop_when_same_size(pso_native):
    state = to_canonical("PSO", pso_native, device="cpu")
    same = resample_population(state, state.n, np.random.default_rng(0))
    assert same is state


def test_resample_population_downsamples_to_best_by_value(pso_native):
    state = to_canonical("PSO", pso_native, device="cpu")
    n_target = state.n - 2
    assert n_target > 0
    out = resample_population(state, n_target, np.random.default_rng(0))
    assert out.n == n_target
    expected_idx = torch.argsort(state.values)[:n_target]
    assert torch.allclose(out.positions, state.positions[expected_idx])
    assert torch.allclose(out.values, state.values[expected_idx])
    assert torch.allclose(out.specific["velocity"], state.specific["velocity"][expected_idx])
    # Per-distribution fields are untouched.
    assert torch.allclose(out.best_x, state.best_x)
    assert torch.allclose(out.best_y, state.best_y)


def test_resample_population_upsamples_with_jitter(pso_native):
    state = to_canonical("PSO", pso_native, device="cpu")
    n = state.n
    n_target = n + 5
    out = resample_population(state, n_target, np.random.default_rng(0))
    assert out.n == n_target
    # Original individuals are kept verbatim at the front.
    assert torch.allclose(out.positions[:n], state.positions)
    assert torch.allclose(out.values[:n], state.values)
    # Synthesized rows are close to (but not identical to) some source row.
    extra = out.positions[n:]
    assert not torch.allclose(extra, state.positions[: n_target - n])
    dists = torch.cdist(extra, state.positions)
    assert (dists.min(dim=1).values < 1.0).all()
    assert out.specific["velocity"].shape == (n_target, state.d)


def test_resample_population_cmaes_specific_fields_untouched(cmaes_native):
    """CMA-ES's specific fields (mean/sigma_cov/p_c/p_s) are per-distribution,
    not per-individual, so resampling the population must leave them exactly
    as they are regardless of direction."""
    state = to_canonical("CMAES", cmaes_native, device="cpu")
    down = resample_population(state, state.n - 1, np.random.default_rng(0))
    up = resample_population(state, state.n + 3, np.random.default_rng(0))
    for out in (down, up):
        for name in ("mean", "sigma_cov", "p_c", "p_s"):
            assert torch.allclose(out.specific[name], state.specific[name])
