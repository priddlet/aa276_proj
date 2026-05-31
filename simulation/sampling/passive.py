"""Sampling-based passive safety (Option 2) and maneuver filtering."""

from __future__ import annotations

import numpy as np

from simulation.cw_dynamics import CWDynamics, propagate_coast_samples
from simulation.keepout import EllipsoidKeepOut


def natural_coast_hits_inner_koz(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    inner: EllipsoidKeepOut,
    horizon_s: float,
    *,
    n_samples: int = 256,
) -> bool:
    """True if free drift from ``x_lvlh_m`` enters ``inner`` within ``horizon_s``."""
    if horizon_s <= 0:
        return inner.is_inside(np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)[:3])
    times = np.linspace(0.0, float(horizon_s), int(n_samples), dtype=np.float64)
    xs = propagate_coast_samples(plant, x_lvlh_m, times)
    for k in range(xs.shape[0]):
        if inner.is_inside(xs[k, :3]):
            return True
    return False


def is_passively_safe_natural_coast(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    inner: EllipsoidKeepOut,
    horizon_s: float,
    *,
    n_samples: int = 256,
) -> bool:
    return not natural_coast_hits_inner_koz(
        plant, x_lvlh_m, inner, horizon_s, n_samples=n_samples
    )
