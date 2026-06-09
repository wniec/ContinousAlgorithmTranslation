"""Running mean/std estimators for PPO reward and observation normalization.

``RunningMeanStd`` uses Welford/parallel moments so statistics accumulate stably
across the whole training run (not just one rollout). Two uses:

* **reward normalization** — rewards are divided by the running std of the
  *discounted return* (the standard PPO/`VecNormalize` trick), which keeps the
  value targets and advantages at ~unit scale even though the log-scaled,
  optimum-relative reward can vary a lot across problems.
* **observation normalization** — the scalar context vector fed to the critic is
  standardized by its running mean/std. (The structured optimizer state is
  already per-sample normalized inside the network via ``NormContext``.)
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        """Update from a batch ``x`` of shape ``(n, *shape)``."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def normalize(
        self, x: np.ndarray, center: bool = True, eps: float = 1e-8
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) if center else x
        return (out / (self.std + eps)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])
