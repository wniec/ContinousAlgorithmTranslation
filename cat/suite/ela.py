"""Exploratory Landscape Analysis (ELA) features.

Vendored from DynamicAlgorithmSelection2 (das/env/observation.py): 22 ELA
features computed with pflacco from a sample of evaluated points ``(x, y)``.
They describe the local landscape (linearity, multimodality, dispersion,
information content, distribution shape) and are used as extra context for the
RL critic's value estimate.

Returns zeros when the sample is too small/degenerate to characterize.
"""

from __future__ import annotations

import warnings

import numpy as np

ELA_DIM = 22
MAX_HISTORY_SAMPLE = 2500
MIN_SAMPLE = 50  # pflacco needs a reasonable sample to be meaningful

ELA_FEATURE_KEYS = [
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_simple.cond",
    "ela_meta.quad_w_interact.adj_r2",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "disp.ratio_mean_02",
    "disp.ratio_median_25",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "ic.h_max",
    "ic.eps_s",
    "ic.eps_max",
    "ic.m0",
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
]


def compute_ela_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the 22 ELA features from a population sample.

    Falls back to zeros when the (deduplicated) sample is too small or the
    objective values are degenerate. pflacco is imported lazily so the module is
    cheap to import when ELA is disabled.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) < MIN_SAMPLE:
        return np.zeros(ELA_DIM, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pandas as pd
        from pflacco.classical_ela_features import (
            calculate_dispersion,
            calculate_ela_distribution,
            calculate_ela_meta,
            calculate_information_content,
            calculate_nbc,
        )

        _, unique_idx = np.unique(x, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        x = x[unique_idx][-MAX_HISTORY_SAMPLE:]
        y = y[unique_idx][-MAX_HISTORY_SAMPLE:]

        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)

        x_df = pd.DataFrame(x_norm, columns=[f"x_{i}" for i in range(x_norm.shape[1])])
        y_series = pd.Series(y_norm)

        is_unique = ~x_df.duplicated()
        if not is_unique.all():
            x_df = x_df[is_unique].reset_index(drop=True)
            y_series = y_series[is_unique].reset_index(drop=True)

        if len(x_df) < MIN_SAMPLE or np.var(y_series) < 1e-8:
            return np.zeros(ELA_DIM, dtype=np.float32)

        meta = calculate_ela_meta(x_df, y_series)
        nbc = calculate_nbc(x_df, y_series)
        disp = calculate_dispersion(x_df, y_series)
        ic = calculate_information_content(x_df, y_series)
        if (y**2).sum() > 0 and np.var(y_series) > 1e-8:
            ela_distr = calculate_ela_distribution(x_df, y_series)
        else:
            ela_distr = {
                "ela_distr.skewness": 0.0,
                "ela_distr.kurtosis": 0.0,
                "ela_distr.number_of_peaks": 0.0,
            }

        feats = {**meta, **nbc, **disp, **ic, **ela_distr}
        out = np.array([feats.get(k, 0.0) for k in ELA_FEATURE_KEYS], dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
