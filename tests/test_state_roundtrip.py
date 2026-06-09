"""State adapter round-trip: native -> canonical -> native preserves fields."""

import numpy as np
import pytest

from cat.state import to_canonical, from_canonical, get_spec
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
