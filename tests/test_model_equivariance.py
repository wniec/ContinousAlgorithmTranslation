"""Symmetry guarantees of the translator:

* particle-permutation invariance of the latent,
* coordinate-permutation equivariance end-to-end,
* PSD covariance output.
"""

import torch

from cat.models import NormContext, TranslatorPair
from cat.state.canonical import CanonicalState

torch.manual_seed(0)


def _random_state(algo: str, B=2, N=6, D=4) -> CanonicalState:
    g = torch.Generator().manual_seed(hash(algo) % 2**31)
    pos = torch.randn(B, N, D, generator=g)
    vals = torch.randn(B, N, generator=g)
    best_x = torch.randn(B, D, generator=g)
    best_y = torch.randn(B, generator=g)
    if algo == "PSO":
        specific = {
            "velocity": torch.randn(B, N, D, generator=g),
            "pbest_x": torch.randn(B, N, D, generator=g),
        }
    else:  # CMAES
        A = torch.randn(B, D, D, generator=g)
        specific = {
            "mean": torch.randn(B, D, generator=g),
            "sigma_cov": A @ A.transpose(-2, -1) + torch.eye(D),
            "p_c": torch.randn(B, D, generator=g),
            "p_s": torch.randn(B, D, generator=g),
        }
    return CanonicalState(algo, pos, vals, best_x, best_y, specific)


def _ctx(state):
    return NormContext.from_shared(state.positions, state.values)


def test_particle_permutation_invariance_of_global_vec():
    pair = TranslatorPair("PSO", "CMAES").eval()
    s = _random_state("PSO")
    perm = torch.randperm(s.n)
    s_perm = CanonicalState(
        "PSO",
        s.positions[:, perm],
        s.values[:, perm],
        s.best_x,
        s.best_y,
        {k: v[:, perm] for k, v in s.specific.items()},
    )
    with torch.no_grad():
        z1 = pair.encode(s, _ctx(s))
        z2 = pair.encode(s_perm, _ctx(s_perm))
    # Pooling over particles -> global vector and dim-tokens are permutation-invariant.
    assert torch.allclose(z1.global_vec, z2.global_vec, atol=1e-5)
    assert torch.allclose(z1.dim_tokens, z2.dim_tokens, atol=1e-5)


def test_coordinate_permutation_equivariance():
    pair = TranslatorPair("PSO", "CMAES").eval()
    s = _random_state("PSO")
    perm = torch.randperm(s.d)

    def permute_coords(state):
        spec = {}
        for k, v in state.specific.items():
            spec[k] = v[..., perm]
        return CanonicalState(
            state.algo,
            state.positions[..., perm],
            state.values,
            state.best_x[..., perm],
            state.best_y,
            spec,
        )

    s_perm = permute_coords(s)
    with torch.no_grad():
        out = pair.translate(s, "CMAES", _ctx(s))
        out_perm = pair.translate(s_perm, "CMAES", _ctx(s_perm))

    # mean (per-dim) should permute with the coordinate axes.
    assert torch.allclose(
        out.specific["mean"][..., perm], out_perm.specific["mean"], atol=1e-4
    )
    # covariance should permute rows and columns.
    cov_p = out.specific["sigma_cov"][:, perm][:, :, perm]
    assert torch.allclose(cov_p, out_perm.specific["sigma_cov"], atol=1e-4)


def test_covariance_output_is_psd():
    pair = TranslatorPair("PSO", "CMAES").eval()
    s = _random_state("PSO")
    with torch.no_grad():
        out = pair.translate(s, "CMAES", _ctx(s))
    cov = out.specific["sigma_cov"]
    eig = torch.linalg.eigvalsh(0.5 * (cov + cov.transpose(-2, -1)))
    assert eig.min().item() > -1e-5
