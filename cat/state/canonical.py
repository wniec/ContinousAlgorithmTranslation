"""Conversion between an optimizer's native warm-start dict and a typed,
tensor-based ``CanonicalState``.

``CanonicalState`` is the common currency the neural translator operates on. It
separates the shared population (positions / values / best) from the
algorithm-specific fields declared in ``cat/state/schema.py``. Tensors are kept
in *raw* units (no normalization) so the round-trip is lossless; any
standardization happens inside the model encoder.

A single state has no batch dimension: ``positions`` is ``(N, D)``,
``best_x`` is ``(D,)``, scalars are 0-dim. Batched states (added by the dataset
collation) carry a leading batch dimension and are handled transparently by the
shape-agnostic ``n`` / ``d`` properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from cat.optimizers.DE import ARATE
from cat.state.schema import Struct, get_spec


@dataclass
class CanonicalState:
    algo: str
    positions: Tensor  # (..., N, D)
    values: Tensor  # (..., N)
    best_x: Tensor  # (..., D)
    best_y: Tensor  # (...,)
    specific: dict[str, Tensor]

    @property
    def n(self) -> int:
        return self.positions.shape[-2]

    @property
    def d(self) -> int:
        return self.positions.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to(self, device) -> "CanonicalState":
        return CanonicalState(
            algo=self.algo,
            positions=self.positions.to(device),
            values=self.values.to(device),
            best_x=self.best_x.to(device),
            best_y=self.best_y.to(device),
            specific={k: v.to(device) for k, v in self.specific.items()},
        )

    def index(self, i: int) -> "CanonicalState":
        """Select one sample from a batched state, returning an un-batched state."""
        return CanonicalState(
            algo=self.algo,
            positions=self.positions[i],
            values=self.values[i],
            best_x=self.best_x[i],
            best_y=self.best_y[i],
            specific={k: v[i] for k, v in self.specific.items()},
        )

    def detach_clone(self) -> "CanonicalState":
        return CanonicalState(
            algo=self.algo,
            positions=self.positions.detach().clone(),
            values=self.values.detach().clone(),
            best_x=self.best_x.detach().clone(),
            best_y=self.best_y.detach().clone(),
            specific={k: v.detach().clone() for k, v in self.specific.items()},
        )


def _as_tensor(x, device, dtype=torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# native dict  ->  CanonicalState                                              #
# --------------------------------------------------------------------------- #


def to_canonical(algo: str, native: dict, device="cpu") -> CanonicalState:
    """Build a ``CanonicalState`` from an augmented native warm-start dict.

    ``native`` must contain the shared keys (``x``, ``y``, ``best_x``,
    ``best_y``) plus the optimizer's specific native keys (e.g. ``v`` / ``cm`` /
    ``sigma``). The collector produces exactly such a dict (see
    ``cat/data/collector.py``).
    """
    spec = get_spec(algo)

    def _get(native_key: str):
        if native_key not in native or native[native_key] is None:
            raise ValueError(
                f"{algo}: missing native field {native_key!r} required by canonical state"
            )
        return native[native_key]

    positions = _as_tensor(_get("x"), device)
    values = _as_tensor(_get("y"), device).reshape(-1)
    best_x = _as_tensor(_get("best_x"), device).reshape(-1)
    best_y = _as_tensor(_get("best_y"), device).reshape(())

    D = positions.shape[-1]
    specific: dict[str, Tensor] = {}
    for f in spec.specific:
        raw = _get(f.native_key)
        t = _as_tensor(raw, device)
        if f.struct is Struct.SCALAR:
            t = t.reshape(())
        elif f.struct in (Struct.PER_PARTICLE, Struct.PER_DIM):
            t = t.reshape(-1)
        elif f.struct is Struct.POINT_SET:
            t = t.reshape(-1, D)
        specific[f.name] = t

    # CMA-ES: fold step-size into the covariance -> full search covariance Sigma.
    if algo == "CMAES":
        sigma = native.get("sigma")
        if sigma is None:
            raise ValueError("CMAES: missing native field 'sigma'")
        specific["sigma_cov"] = specific["sigma_cov"] * float(sigma) ** 2

    # MadDE: resize the archive to its fixed capacity round(ARATE * N) so the
    # point-set has a constant size within each (algo, D, N) batch group.
    if algo == "MADDE":
        m_cap = max(1, int(round(ARATE * positions.shape[0])))
        specific["archive"] = _resize_point_set(specific["archive"], m_cap, positions)

    return CanonicalState(algo, positions, values, best_x, best_y, specific)


def _resize_point_set(points: Tensor, m_cap: int, positions: Tensor) -> Tensor:
    """Resize a point-set to exactly ``m_cap`` rows, deterministically: truncate
    if too many; if too few, cycle the existing points (or the population, when
    the set is empty). Deterministic so the canonical state — and hence the
    cycle-consistency loss — is well-defined."""
    m = points.shape[0]
    if m == m_cap:
        return points
    if m > m_cap:
        return points[:m_cap]
    src = points if m > 0 else positions
    idx = torch.arange(m_cap, device=points.device) % src.shape[0]
    return src[idx]


def resample_population(
    state: CanonicalState, n_target: int, rng: np.random.Generator
) -> CanonicalState:
    """Resize an *un-batched* state's population to exactly ``n_target`` rows.

    Downsampling keeps the best ``n_target`` individuals by ``values`` —
    mirrors the truncation every optimizer's own ``set_data``/``initialize``
    already performs internally. Upsampling keeps every existing individual
    and synthesizes the rest by resampling rows with replacement and jittering
    their positions (scaled to the population's own spread); the synthesized
    rows' ``values`` and any per-individual specific field are copied from
    their parent row unchanged — no objective-function evaluation happens
    here, so this is a warm-start seed, not a ground-truth fitness.

    Only fields that scale with population size (shared ``positions``/
    ``values`` plus any ``Struct.PER_PARTICLE_PER_DIM``/``PER_PARTICLE``
    specific field, e.g. PSO's ``velocity``/``pbest_x``) are resampled;
    per-distribution fields (``best_x``, ``best_y``, CMA-ES's mean/covariance,
    MadDE's archive) pass through unchanged.
    """
    if state.positions.dim() != 2:
        raise ValueError("resample_population expects a single un-batched state")
    n = state.n
    if n_target == n:
        return state

    per_row = {
        f.name
        for f in get_spec(state.algo).specific
        if f.struct in (Struct.PER_PARTICLE_PER_DIM, Struct.PER_PARTICLE)
    }
    device = state.positions.device

    if n_target < n:
        idx = torch.argsort(state.values)[:n_target]
    else:
        extra = n_target - n
        extra_idx = torch.as_tensor(
            rng.integers(0, n, size=extra), dtype=torch.long, device=device
        )
        idx = torch.cat([torch.arange(n, device=device), extra_idx])

    positions = state.positions[idx].clone()
    values = state.values[idx]
    specific = {
        name: (val[idx] if name in per_row else val)
        for name, val in state.specific.items()
    }

    if n_target > n:
        std = state.positions.std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = torch.as_tensor(
            rng.normal(size=(n_target - n, state.d)),
            dtype=positions.dtype,
            device=device,
        )
        positions[n:] = positions[n:] + 0.1 * std * noise

    return CanonicalState(state.algo, positions, values, state.best_x, state.best_y, specific)


# --------------------------------------------------------------------------- #
# CanonicalState  ->  native dict (for set_data)                               #
# --------------------------------------------------------------------------- #


def from_canonical(state: CanonicalState) -> dict:
    """Serialize a (single, un-batched) ``CanonicalState`` into a numpy dict
    accepted by the target optimizer's ``set_data(**dict)``.

    For CMA-ES the eigen-decomposition (``e_ve`` / ``e_va`` / ``d``) is derived
    from the covariance so the warm-started search basis takes effect on the
    very first post-switch generation rather than defaulting to the identity.
    """
    if state.positions.dim() != 2:
        raise ValueError("from_canonical expects a single un-batched state")

    spec = get_spec(state.algo)

    def _np(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    native: dict = {
        "x": _np(state.positions),
        "y": _np(state.values),
        "best_x": _np(state.best_x),
        "best_y": float(state.best_y),
    }

    for f in spec.specific:
        val = state.specific[f.name]
        if f.struct is Struct.SCALAR:
            native[f.native_key] = float(val)
        else:
            native[f.native_key] = _np(val)

    # CMA-ES: split the full search covariance Sigma back into the standard
    # (sigma, C) pair with det(C)=1, and derive the sampling basis so the
    # warm-started distribution takes effect on the first post-switch generation.
    if state.algo == "CMAES":
        sigma_cov = native.pop("cm")  # this field carried Sigma = sigma^2 * C
        sigma_cov = 0.5 * (sigma_cov + sigma_cov.T)
        eigvals, eigvecs = np.linalg.eigh(sigma_cov)
        eigvals = np.maximum(eigvals, 1e-20)
        log_det = float(np.mean(np.log(eigvals)))  # = (1/D) * logdet(Sigma)
        sigma = float(np.exp(0.5 * log_det))  # det(C)=1 convention
        sigma = float(np.clip(sigma, 1e-8, 1e8))
        c_eigvals = eigvals / sigma**2
        e_va = np.sqrt(c_eigvals)
        native["cm"] = (eigvecs * c_eigvals) @ eigvecs.T  # C = Sigma / sigma^2
        native["sigma"] = sigma
        native["e_ve"] = eigvecs
        native["e_va"] = e_va
        native["d"] = e_va

    return native


def warm_start_optimizer(opt, native: dict) -> None:
    """Warm-start a freshly constructed optimizer from a native state dict.

    ``set_data`` consumes most fields, but CMA-ES's ``set_data`` ignores
    ``sigma`` (it is normally an init option), so we assign it explicitly when
    present — otherwise the translated step-size would be silently dropped.
    """
    opt.set_data(**native)
    if native.get("sigma") is not None and hasattr(opt, "sigma"):
        opt.sigma = float(native["sigma"])
