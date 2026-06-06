"""Clohessy–Wiltshire (Hill) linearized relative motion for circular chief orbit.

State in LVLH frame fixed to the target (chief), x = [x, y, z, vx, vy, vz]^T:
  x: radial (positive away from Earth, along -R from chief in typical convention;
     here we use the usual Hills form with x radial outward from Earth)
  y: along-track (positive in the direction of chief velocity)
  z: cross-track (positive along chief angular momentum h = r × v)

Impulsive delta-v is applied to velocity components only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from simulation.cw_closed_form import cw_state_closed_form
# Earth gravitational parameter (km^3/s^2)
MU_EARTH_KM3_S2 = 398_600.4418
# Earth equatorial radius (km)
R_EARTH_KM = 6_378.137


def cw_mean_motion_circular(a_km: float) -> float:
    """Mean motion n = sqrt(mu / a^3) for a circular orbit (rad/s)."""
    n_rad_s = np.sqrt(MU_EARTH_KM3_S2 / (a_km**3))
    return float(n_rad_s)


def cw_mean_motion_leo(altitude_km: float = 400.0) -> float:
    """Mean motion for a circular LEO at given altitude above Earth (km)."""
    a_km = R_EARTH_KM + float(altitude_km)
    return cw_mean_motion_circular(a_km)


@dataclass(frozen=True)
class LEOCircularOrbit:
    """Circular LEO chief: mean motion and semi-major axis from altitude (WGS84-like Earth)."""

    n_rad_s: float
    a_km: float
    altitude_km: float


def leo_circular_orbit(altitude_km: float = 400.0) -> LEOCircularOrbit:
    """'a = R_Earth + h', 'n = sqrt(mu/a^3)' (no optional poliastro dependency)."""
    h = float(altitude_km)
    a_km = float(R_EARTH_KM + h)
    return LEOCircularOrbit(n_rad_s=cw_mean_motion_circular(a_km), a_km=a_km, altitude_km=h)


def _cw_A(n: float) -> np.ndarray:
    n = float(n)
    return np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3.0 * n * n, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
            [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
            [0.0, 0.0, -n * n, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


class CWDynamics:
    """Discrete-time CW plant with impulsive velocity changes."""

    def __init__(self, n: float) -> None:
        self.n = float(n)
        self._A = _cw_A(self.n)

    def continuous_A(self) -> np.ndarray:
        return self._A.copy()

    def state_transition(self, dt: float) -> np.ndarray:
        """Phi(dt) = exp(A dt), 6x6."""
        dt = float(dt)
        return expm(self._A * dt)

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        """x' = Phi(dt) x using closed-form CW (same as exp(A dt); see 'cw_closed_form')."""
        x = np.asarray(x, dtype=np.float64).reshape(6)
        return cw_state_closed_form(x, self.n, float(dt))

    @staticmethod
    def apply_impulsive_dv(x: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Add impulsive Delta-v to velocities: x^+ = x^- + [0,0,0,dvx,dvy,dvz]."""
        x = np.asarray(x, dtype=np.float64).reshape(6)
        dv = np.asarray(dv, dtype=np.float64).reshape(3)
        out = x.copy()
        out[3:6] += dv
        return out

    def step_with_dv(self, x: np.ndarray, dt: float, dv: np.ndarray | None = None) -> np.ndarray:
        """Propagate over dt; optional impulsive dv applied at segment start."""
        x0 = np.asarray(x, dtype=np.float64).reshape(6)
        if dv is not None:
            x0 = self.apply_impulsive_dv(x0, dv)
        return self.propagate(x0, dt)


def simulate_impulsive_segments(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out a maneuver plan as piecewise-CW segments with impulsive delta-v at each
    segment boundary (applied at the start of that segment).

    segments: list of (dt, dv) where dv may be None for coast arcs.

    Returns (times, states) with times[0] = 0, states[k] is state at times[k],
    length len(segments) + 1.
    """
    x = np.asarray(x0, dtype=np.float64).reshape(6)
    t_nodes: list[float] = [0.0]
    x_nodes: list[np.ndarray] = [x.copy()]
    t = 0.0
    for dt, dv in segments:
        dt = float(dt)
        x = plant.step_with_dv(x, dt, dv)
        t += dt
        t_nodes.append(t)
        x_nodes.append(x.copy())
    return np.array(t_nodes, dtype=np.float64), np.stack(x_nodes, axis=0)


def simulate_impulsive_segments_dense(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    substeps: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Same maneuver plan as 'simulate_impulsive_segments' with uniform CW substeps.

    Impulsive Δv is applied at the start of each segment, then the segment duration
    is split into 'substeps' propagations for smooth visualization.

    Returns (times, states) with length '1 + len(segments) * substeps'.
    """
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    x = np.asarray(x0, dtype=np.float64).reshape(6)
    t = 0.0
    t_list: list[float] = [t]
    x_list: list[np.ndarray] = [x.copy()]
    for dt, dv in segments:
        dt = float(dt)
        h = dt / substeps
        if dv is not None:
            x = plant.apply_impulsive_dv(x, dv)
        for _ in range(substeps):
            x = plant.propagate(x, h)
            t += h
            t_list.append(t)
            x_list.append(x.copy())
    return np.array(t_list, dtype=np.float64), np.stack(x_list, axis=0)


def maneuver_total_duration_s(segments: list[tuple[float, np.ndarray | None]]) -> float:
    """Total elapsed time (sum of segment 'dt') for an impulsive maneuver list."""
    return float(sum(float(dt) for dt, _ in segments))


def state_at_maneuver_elapsed_time(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    t_elapsed_s: float,
) -> np.ndarray:
    """CW LVLH state at 't_elapsed_s' along the same impulsive timing as 'simulate_impulsive_segments'.

    't_elapsed_s <= 0' returns 'x0_lvlh_m'. Otherwise time advances segment-by-segment: optional
    impulse at each segment start, then CW coast for the remainder of that segment's ``dt``.
    """
    if t_elapsed_s <= 0.0:
        return np.asarray(x0_lvlh_m, dtype=np.float64).reshape(6).copy()
    x = np.asarray(x0_lvlh_m, dtype=np.float64).reshape(6).copy()
    t_left = float(t_elapsed_s)
    for dt, dv in segments:
        dt = float(dt)
        x_body = plant.apply_impulsive_dv(x, dv) if dv is not None else x
        if t_left <= dt + 1e-12:
            return plant.propagate(x_body, t_left)
        x = plant.propagate(x_body, dt)
        t_left -= dt
    return x


def propagate_coast_samples(
    plant: CWDynamics,
    x0: np.ndarray,
    sample_times_s: np.ndarray,
) -> np.ndarray:
    """CW coast from 'x0' at each 'sample_times_s[k]' (seconds), vectorized loop.

    Returns 'states' shaped (N, 6) in LVLH meters / (m/s).
    """
    x0 = np.asarray(x0, dtype=np.float64).reshape(6)
    ts = np.asarray(sample_times_s, dtype=np.float64).reshape(-1)
    out = np.zeros((ts.shape[0], 6), dtype=np.float64)
    for k, t in enumerate(ts):
        out[k] = plant.propagate(x0, float(t))
    return out
