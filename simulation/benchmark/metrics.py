"""Rollout metrics for paper-style evaluation (interception, success, Δv overhead)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.cw_dynamics import CWDynamics, simulate_impulsive_segments_dense
from simulation.keepout import EllipsoidKeepOut


@dataclass(frozen=True)
class RolloutMetrics:
    """Per-plan rollout statistics."""

    intercepted: bool
    mission_success: bool
    mean_dv_overhead_m_s: float
    n_burns: int
    n_burns_corrected: int
    final_range_m: float
    min_koz_shape_value: float


def trajectory_koz_stats(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    *,
    sample_dt_s: float = 5.0,
) -> tuple[bool, float]:
    """Return '(intercepted, min_shape_value)' along a dense CW rollout."""
    substeps = max(4, int(round(20.0 / max(sample_dt_s, 0.1))))
    times, states = simulate_impulsive_segments_dense(plant, x0, segments, substeps=substeps)
    min_shape = float("inf")
    intercepted = False
    for x in states:
        r = x[:3]
        s = float(inner.shape_value(r))
        min_shape = min(min_shape, s)
        if inner.is_inside(r):
            intercepted = True
    if not np.isfinite(min_shape):
        min_shape = 0.0
    return intercepted, min_shape


def mission_success_at_final(
    x_final: np.ndarray,
    *,
    target_pos_m: np.ndarray,
    capture_radius_m: float,
    intercepted: bool,
) -> bool:
    """Chief at 'target_pos_m'; success = captured without KOZ entry."""
    if intercepted:
        return False
    r = np.asarray(x_final, dtype=np.float64).reshape(6)[:3]
    tgt = np.asarray(target_pos_m, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(r - tgt)) <= float(capture_radius_m)


def compute_rollout_metrics(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    *,
    nominal_segments: list[tuple[float, np.ndarray | None]] | None = None,
    target_pos_m: np.ndarray | None = None,
    capture_radius_m: float = 100.0,
    sample_dt_s: float = 5.0,
) -> RolloutMetrics:
    """Interception = inner KOZ entry; success = final capture without interception."""
    intercepted, min_shape = trajectory_koz_stats(plant, x0, segments, inner, sample_dt_s=sample_dt_s)

    _, states = simulate_impulsive_segments_dense(
        plant, x0, segments, substeps=max(4, int(round(20.0 / max(sample_dt_s, 0.1))))
    )
    x_final = states[-1]
    tgt = np.zeros(3, dtype=np.float64) if target_pos_m is None else np.asarray(target_pos_m, dtype=np.float64).reshape(3)
    success = mission_success_at_final(
        x_final,
        target_pos_m=tgt,
        capture_radius_m=capture_radius_m,
        intercepted=intercepted,
    )

    overheads: list[float] = []
    n_corrected = 0
    if nominal_segments is not None:
        x = np.asarray(x0, dtype=np.float64).reshape(6).copy()
        t = 0.0
        nom = list(nominal_segments)
        for i, (dt, dv) in enumerate(segments):
            dt = float(dt)
            dv_nom = nom[i][1] if i < len(nom) else None
            if dv is not None and dv_nom is not None:
                dv_a = np.asarray(dv, dtype=np.float64).reshape(3)
                dv_n = np.asarray(dv_nom, dtype=np.float64).reshape(3)
                res = float(np.linalg.norm(dv_a - dv_n))
                overheads.append(res)
                if res > 1e-9:
                    n_corrected += 1
            x = plant.step_with_dv(x, dt, dv)
            t += dt

    mean_overhead = float(np.mean(overheads)) if overheads else 0.0
    final_range = float(np.linalg.norm(x_final[:3] - tgt))

    return RolloutMetrics(
        intercepted=intercepted,
        mission_success=success,
        mean_dv_overhead_m_s=mean_overhead,
        n_burns=sum(1 for _, dv in segments if dv is not None),
        n_burns_corrected=n_corrected,
        final_range_m=final_range,
        min_koz_shape_value=min_shape,
    )
