"""Earth-centered inertial (ECI) kinematics for circular chief + CW deputy.

Assumes a **prograde circular equatorial** chief orbit in the ECI x-y plane, with
angular rate 'n' and semi-major axis 'a_km'. LVLH columns match 'cw_dynamics'
(x radial outward from Earth, y along-track, z cross-track = +ECI z).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from simulation.cw_dynamics import R_EARTH_KM, propagate_coast_samples, state_at_maneuver_elapsed_time

if TYPE_CHECKING:
    from simulation.cw_dynamics import CWDynamics


def circular_orbit_radius_km(altitude_km: float) -> float:
    return float(R_EARTH_KM + float(altitude_km))


def chief_lvlh_to_eci_matrix(theta: float) -> np.ndarray:
    """Rotation R such that r_eci = R @ r_lvlh (columns are LVLH basis vectors in ECI)."""
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x_hat = np.array([c, s, 0.0], dtype=np.float64)
    y_hat = np.array([-s, c, 0.0], dtype=np.float64)
    z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.column_stack([x_hat, y_hat, z_hat])


def chief_circular_eci(
    t_s: float,
    a_km: float,
    n_rad_s: float,
    theta0_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chief position (km), velocity (km/s), and LVLH-to-ECI matrix at time 't_s'."""
    theta = n_rad_s * float(t_s) + float(theta0_rad)
    R = chief_lvlh_to_eci_matrix(theta)
    r_c = a_km * np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    v_c = (n_rad_s * a_km) * np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=np.float64)
    return r_c, v_c, R


def deputy_eci_from_cw(
    t_s: float,
    x_lvlh_m: np.ndarray,
    a_km: float,
    n_rad_s: float,
    theta0_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deputy ECI position (km) and velocity (km/s) from CW LVLH state at the same 't_s'."""
    r_c, v_c, R = chief_circular_eci(t_s, a_km, n_rad_s, theta0_rad)
    x_lvlh_m = np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)
    r_m = x_lvlh_m[:3]
    v_m_s = x_lvlh_m[3:6]
    rho_km = (R @ r_m) / 1000.0
    omega = np.array([0.0, 0.0, float(n_rad_s)], dtype=np.float64)
    rho_dot_km_s = (R @ v_m_s) / 1000.0 + np.cross(omega, rho_km)
    r_d = r_c + rho_km
    v_d = v_c + rho_dot_km_s
    return r_d, v_d


def body_triad_velocity_normal(
    v_eci_km_s: np.ndarray,
    orbit_normal: np.ndarray | None = None,
) -> np.ndarray:
    """Right-handed body frame in ECI: +x along 'v', +z ~ orbit normal."""
    v = np.asarray(v_eci_km_s, dtype=np.float64).reshape(3)
    h = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if orbit_normal is None
        else np.asarray(orbit_normal, dtype=np.float64).reshape(3)
    )
    h = h / max(np.linalg.norm(h), 1e-15)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(3, dtype=np.float64)
    x = v / nv
    z = np.cross(h, x)
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        z = np.cross(ref, x)
        zn = np.linalg.norm(z)
    z = z / zn
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def build_eci_ephemeris(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    sample_times_s: np.ndarray,
    altitude_km: float = 400.0,
    theta0_rad: float = 0.0,
    a_km: float | None = None,
) -> dict[str, np.ndarray]:
    """Precompute chief/deputy ECI states and body triads for each sample time."""
    n_rad_s = float(plant.n)
    if a_km is None:
        a_km = circular_orbit_radius_km(altitude_km)
    a_km = float(a_km)
    times = np.asarray(sample_times_s, dtype=np.float64).reshape(-1)
    states_lvlh = propagate_coast_samples(plant, x0_lvlh_m, times)

    n_frames = times.shape[0]
    r_c = np.zeros((n_frames, 3), dtype=np.float64)
    r_d = np.zeros((n_frames, 3), dtype=np.float64)
    v_c = np.zeros((n_frames, 3), dtype=np.float64)
    v_d = np.zeros((n_frames, 3), dtype=np.float64)
    R_chief = np.zeros((n_frames, 3, 3), dtype=np.float64)
    R_dep = np.zeros((n_frames, 3, 3), dtype=np.float64)

    for k, t in enumerate(times):
        r_ckm, v_ckm, Rk = chief_circular_eci(float(t), a_km, n_rad_s, theta0_rad)
        r_dkm, v_dkm = deputy_eci_from_cw(float(t), states_lvlh[k], a_km, n_rad_s, theta0_rad)
        r_c[k] = r_ckm
        r_d[k] = r_dkm
        v_c[k] = v_ckm
        v_d[k] = v_dkm
        R_chief[k] = Rk
        R_dep[k] = body_triad_velocity_normal(v_dkm)

    return {
        "times_s": times,
        "states_lvlh_m": states_lvlh,
        "r_chief_km": r_c,
        "r_deputy_km": r_d,
        "v_chief_km_s": v_c,
        "v_deputy_km_s": v_d,
        "R_chief_body_to_eci": R_chief,
        "R_deputy_body_to_eci": R_dep,
        "a_km": np.array([a_km]),
        "n_rad_s": np.array([n_rad_s]),
    }


def build_eci_ephemeris_from_segments(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    sample_times_s: np.ndarray,
    *,
    altitude_km: float = 400.0,
    theta0_rad: float = 0.0,
    a_km: float | None = None,
) -> dict[str, np.ndarray]:
    """ECI ephemeris along an impulsive maneuver plan (not free drift only)."""
    n_rad_s = float(plant.n)
    if a_km is None:
        a_km = circular_orbit_radius_km(altitude_km)
    a_km = float(a_km)
    times = np.asarray(sample_times_s, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0_lvlh_m, dtype=np.float64).reshape(6)
    states_lvlh = np.stack(
        [state_at_maneuver_elapsed_time(plant, x0, segments, float(t)) for t in times],
        axis=0,
    )

    n_frames = times.shape[0]
    r_c = np.zeros((n_frames, 3), dtype=np.float64)
    r_d = np.zeros((n_frames, 3), dtype=np.float64)
    v_c = np.zeros((n_frames, 3), dtype=np.float64)
    v_d = np.zeros((n_frames, 3), dtype=np.float64)
    R_chief = np.zeros((n_frames, 3, 3), dtype=np.float64)
    R_dep = np.zeros((n_frames, 3, 3), dtype=np.float64)

    for k, t in enumerate(times):
        r_ckm, v_ckm, Rk = chief_circular_eci(float(t), a_km, n_rad_s, theta0_rad)
        r_dkm, v_dkm = deputy_eci_from_cw(float(t), states_lvlh[k], a_km, n_rad_s, theta0_rad)
        r_c[k] = r_ckm
        r_d[k] = r_dkm
        v_c[k] = v_ckm
        v_d[k] = v_dkm
        R_chief[k] = Rk
        R_dep[k] = body_triad_velocity_normal(v_dkm)

    return {
        "times_s": times,
        "states_lvlh_m": states_lvlh,
        "r_chief_km": r_c,
        "r_deputy_km": r_d,
        "v_chief_km_s": v_c,
        "v_deputy_km_s": v_d,
        "R_chief_body_to_eci": R_chief,
        "R_deputy_body_to_eci": R_dep,
        "a_km": np.array([a_km]),
        "n_rad_s": np.array([n_rad_s]),
    }
