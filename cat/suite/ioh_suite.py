"""IOH-based BBOB problem suite.

Vendored from DynamicAlgorithmSelection2 (das/env/ioh_suite.py).

    suite = IOHSuite()
    problem = suite.get_problem("bbob_f001_i01_d05")
    y = problem(x)
    dim = problem.dimension
    lb = problem.lower_bounds   # np.ndarray
    ub = problem.upper_bounds   # np.ndarray

Problem IDs follow the format ``bbob_f{fid:03d}_i{iid:02d}_d{dim:02d}``.
"""

from __future__ import annotations

import re

import numpy as np

_ID_PATTERN = re.compile(r"^bbob_f(\d+)_i(\d+)_d(\d+)$")


class IOHProblemWrapper:
    """Wraps an IOH BBOB problem with the interface expected by the collector."""

    __slots__ = ("_p",)

    def __init__(self, p) -> None:
        self._p = p

    @property
    def dimension(self) -> int:
        return int(self._p.meta_data.n_variables)

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.asarray(self._p.bounds.lb, dtype=np.float64)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.asarray(self._p.bounds.ub, dtype=np.float64)

    @property
    def optimum(self) -> float:
        """Known global minimum (objective value) of the problem."""
        return float(self._p.optimum.y)

    def __call__(self, x) -> float:
        return float(self._p(x))


class IOHSuite:
    """Drop-in replacement for ``cocoex.Suite("bbob", "", "")``.

    Stateless -- a new IOH problem object is created for every ``get_problem``
    call, so the suite is safely shareable across threads and picklable.
    """

    def get_problem(self, problem_id: str) -> IOHProblemWrapper:
        """Return a wrapped BBOB problem for the given *problem_id*.

        Parameters
        ----------
        problem_id:
            String of the form ``bbob_f{fid:03d}_i{iid:02d}_d{dim:02d}``.
        """
        m = _ID_PATTERN.match(problem_id)
        if m is None:
            raise ValueError(
                f"Cannot parse problem_id {problem_id!r}. "
                "Expected format: bbob_f<fid>_i<iid>_d<dim>"
            )
        fid, iid, dim = int(m.group(1)), int(m.group(2)), int(m.group(3))

        import ioh

        p = ioh.get_problem(fid, iid, dim, problem_class=ioh.ProblemClass.BBOB)
        return IOHProblemWrapper(p)
