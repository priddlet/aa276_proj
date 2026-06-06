"""CW plan helpers for ``generate_plans_leo.py`` (impulsive LVLH, ellipsoidal KOZ labels)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut


def propagate(state6: np.ndarray, dt: float, n: float) -> np.ndarray:
    return CWDynamics(float(n)).propagate(np.asarray(state6, dtype=np.float64).reshape(6), float(dt))


def intercept_velocity(
    r0: np.ndarray,
    T: float,
    rf: np.ndarray,
    n: float,
    v0: np.ndarray | None = None,
) -> np.ndarray:
    """Impulsive Δv (m/s) so CW coast from ``(r0, v0)`` reaches ``rf`` at time ``T``."""
    plant = CWDynamics(float(n))
    r0 = np.asarray(r0, dtype=np.float64).reshape(3)
    rf = np.asarray(rf, dtype=np.float64).reshape(3)
    v0 = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).reshape(3)
    phi = plant.state_transition(float(T))
    phi_rr = phi[:3, :3]
    phi_rv = phi[:3, 3:6]
    v_total = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    return v_total - v0


def _ellipsoid_g(pos: np.ndarray, semi: np.ndarray) -> float:
    r = np.asarray(pos, dtype=np.float64).reshape(3)
    ax, ay, az = (float(x) for x in np.asarray(semi, dtype=np.float64).reshape(3))
    s = (r[0] / ax) ** 2 + (r[1] / ay) ** 2 + (r[2] / az) ** 2
    return float(np.sqrt(s + 1e-18) - 1.0)


def free_drift_metrics(
    state6: np.ndarray,
    *,
    tau: float,
    n: float,
    semi: np.ndarray,
) -> dict[str, float | bool | None]:
    """Passive CW drift: does the deputy enter the KOZ ellipsoid within ``tau`` s?"""
    plant = CWDynamics(float(n))
    inner = EllipsoidKeepOut(np.asarray(semi, dtype=np.float64).reshape(3))
    x0 = np.asarray(state6, dtype=np.float64).reshape(6)
    dt = 5.0
    tau = float(tau)
    min_g = float("inf")
    min_sep = float("inf")
    t_hit: float | None = None
    t = 0.0
    while t <= tau + 1e-9:
        x = plant.propagate(x0, t)
        pos = x[:3]
        g = _ellipsoid_g(pos, semi)
        min_g = min(min_g, g)
        if inner.is_inside(pos):
            sep = 0.0
        else:
            # approximate clearance as g * min semi-axis scale
            sep = g * float(np.min(semi))
        min_sep = min(min_sep, sep)
        if g <= 0.0 and t_hit is None:
            t_hit = t
        t += dt
    return {
        "reaches_koz": t_hit is not None,
        "t_hit_s": t_hit,
        "min_koz_val": min_g,
        "min_sep_m": min_sep,
    }


def absolute_to_segments(maneuvers: list[dict]) -> list[dict]:
    """``[{t_s, dv_m_s}]`` absolute burns → ``[{coast_s, dv_m_s|null}]`` sim segments."""
    burns = sorted(maneuvers, key=lambda m: float(m["t_s"]))
    segs: list[dict] = []
    t_prev = 0.0
    for m in burns:
        t_b = float(m["t_s"])
        gap = t_b - t_prev
        if gap > 0.0:
            segs.append({"coast_s": round(gap, 4), "dv_m_s": None})
        dv = [float(x) for x in m["dv_m_s"]]
        segs.append({"coast_s": 0.0, "dv_m_s": dv})
        t_prev = t_b
    return segs


def simulate_segments(
    segments: list[dict],
    *,
    x0: np.ndarray,
    n: float,
) -> list[np.ndarray]:
    """Post-burn states after each impulse (for label cross-check)."""
    plant = CWDynamics(float(n))
    x = np.asarray(x0, dtype=np.float64).reshape(6).copy()
    posts: list[np.ndarray] = []
    for seg in segments:
        dt = float(seg["coast_s"])
        dv_raw = seg.get("dv_m_s")
        if dv_raw is not None:
            dv = np.asarray(dv_raw, dtype=np.float64).reshape(3)
            x = plant.apply_impulsive_dv(x, dv)
            posts.append(x.copy())
        if dt > 0.0:
            x = plant.propagate(x, dt)
    return posts
