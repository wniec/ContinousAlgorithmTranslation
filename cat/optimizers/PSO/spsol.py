"""SPSOL: SPSO with local ring topology."""

import numpy as np

from .base import PSO


class SPSOL(PSO):
    """SPSO with local ring topology: each particle's guide is the best among
    its left neighbour, itself, and its right neighbour."""

    def _social_guide(self, i, p_x, p_y, n_x):
        left = (i - 1) % self.n_individuals
        right = (i + 1) % self.n_individuals
        ring = [left, i, right]
        return p_x[ring[int(np.argmin(p_y[ring]))]]
