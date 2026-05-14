"""KOZ-entry collision BRT using vendored :mod:`hj_reachability` (Hamilton–Jacobi).

**Option 1 (primary):** failure set is deputy position inside the inner ellipsoidal KOZ.
The default solver is **6D** CW state ``(x,y,z,v_x,v_y,v_z)`` with bounded thrust and
optional disturbance (Option 3). Initial value is ellipsoid ``s(r)-1`` (negative inside
the KOZ), velocity-independent. After backward integration with ``backwards_reachable_tube``,
**``value(x) ≤ 0``** is inside the discriminating BRT.

:class:`KozHJTable6D` uses :class:`scipy.interpolate.RegularGridInterpolator` for fast
queries (e.g. animation + marching-cubes shells).

**Option 2** (passive drift) lives in :mod:`simulation.passive_safety`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_HJ_ROOT = _ROOT / "hj_reachability"
if _HJ_ROOT.is_dir():
    _p = str(_HJ_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import jax
    import jax.numpy as jnp
    import hj_reachability as hj

    from simulation.hj_cw_6d_dynamics import CwFull6DHJDynamics

    HJ_AVAILABLE = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    hj = None  # type: ignore[assignment]
    CwFull6DHJDynamics = None  # type: ignore[assignment]
    HJ_AVAILABLE = False


@dataclass
class KozBRTResult6D:
    times_s: np.ndarray
    values: np.ndarray
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    grid_shape: tuple[int, int, int, int, int, int]
    n_rad_s: float
    u_max_m_s2: float
    d_max_m_s2: float


def koz6d_initial_values_from_metric(
    grid: Any,
    *,
    center_m: np.ndarray,
    E_metric: np.ndarray,
) -> Any:
    """``(r-c)^T E (r-c) - 1`` on grid (negative inside KOZ); velocity dims ignored."""
    if jnp is None:
        raise RuntimeError("jax not available")
    c = jnp.array(np.asarray(center_m, dtype=np.float64).reshape(3), dtype=jnp.float64)
    E = jnp.array(np.asarray(E_metric, dtype=np.float64).reshape(3, 3), dtype=jnp.float64)
    r = grid.states[..., :3]
    d = r - c
    s = jnp.einsum("...i,ij,...j->...", d, E, d)
    return s - 1.0


class KozHJTable6D:
    """6D HJ value via :class:`scipy.interpolate.RegularGridInterpolator` (fast batch queries)."""

    def __init__(self, result: KozBRTResult6D, *, use_final_time: bool = True) -> None:
        from scipy.interpolate import RegularGridInterpolator

        if not HJ_AVAILABLE:
            raise RuntimeError("jax / hj_reachability not available")
        self._lo = np.asarray(result.domain_lo, dtype=np.float64).reshape(6)
        self._hi = np.asarray(result.domain_hi, dtype=np.float64).reshape(6)
        self._shape = tuple(int(x) for x in result.grid_shape)
        idx = -1 if use_final_time else 0
        vals = np.asarray(result.values[idx], dtype=np.float64)
        axes = [np.linspace(self._lo[d], self._hi[d], self._shape[d], dtype=np.float64) for d in range(6)]
        self._interp = RegularGridInterpolator(axes, vals, bounds_error=False, fill_value=np.nan)

    @property
    def domain_lo(self) -> np.ndarray:
        return self._lo.copy()

    @property
    def domain_hi(self) -> np.ndarray:
        return self._hi.copy()

    @property
    def grid_shape(self) -> tuple[int, int, int, int, int, int]:
        return self._shape

    def value_batch(self, x6: np.ndarray) -> np.ndarray:
        """``x6`` shaped ``(N, 6)`` → ``(N,)``."""
        p = np.asarray(x6, dtype=np.float64).reshape(-1, 6)
        return np.asarray(self._interp(p), dtype=np.float64).reshape(-1)

    def value(self, x_lvlh_m: np.ndarray) -> float:
        v = float(self.value_batch(np.asarray(x_lvlh_m, dtype=np.float64).reshape(1, 6))[0])
        return v

    def is_unsafe(self, x_lvlh_m: np.ndarray) -> bool:
        v = self.value(x_lvlh_m)
        if np.isnan(v):
            return True
        return v <= 0.0


def solve_koz_collision_brt_6d(
    n_rad_s: float,
    *,
    inner_metric_E: np.ndarray,
    inner_center_m: np.ndarray,
    u_max_m_s2: float = 0.05,
    d_max_m_s2: float = 0.0,
    horizon_s: float = 90.0,
    n_time_nodes: int = 4,
    domain_lo: np.ndarray | None = None,
    domain_hi: np.ndarray | None = None,
    grid_shape: tuple[int, int, int, int, int, int] | None = None,
    accuracy: str = "low",
    progress_bar: bool = False,
) -> KozBRTResult6D:
    """Backward HJ collision BRT on full 6D CW state (Option 1)."""
    if not HJ_AVAILABLE or hj is None or CwFull6DHJDynamics is None:
        raise RuntimeError(
            "solve_koz_collision_brt_6d requires jax, flax, tqdm, hj_reachability, and scipy "
            "(install requirements-brt.txt and `pip install -e hj_reachability`)."
        )
    if domain_lo is None:
        domain_lo = np.array([-2200.0, -6000.0, -900.0, -3.0, -3.0, -3.0], dtype=np.float64)
    if domain_hi is None:
        domain_hi = np.array([2200.0, 7000.0, 900.0, 3.0, 3.0, 3.0], dtype=np.float64)
    if grid_shape is None:
        grid_shape = (7, 7, 5, 5, 5, 5)

    dynamics = CwFull6DHJDynamics(
        float(n_rad_s),
        float(u_max_m_s2),
        float(d_max_m_s2),
        control_mode="min",
        disturbance_mode="max",
    )
    lo = jnp.array(domain_lo, dtype=jnp.float64)
    hi = jnp.array(domain_hi, dtype=jnp.float64)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lo, hi),
        tuple(int(x) for x in grid_shape),
        periodic_dims=None,
    )
    initial = koz6d_initial_values_from_metric(
        grid,
        center_m=np.asarray(inner_center_m, dtype=np.float64),
        E_metric=np.asarray(inner_metric_E, dtype=np.float64),
    )
    times = np.linspace(0.0, -float(abs(horizon_s)), int(n_time_nodes), dtype=np.float64)
    settings = hj.SolverSettings.with_accuracy(
        accuracy,
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )
    all_v = hj.solve(
        settings,
        dynamics,
        grid,
        jnp.array(times),
        initial,
        progress_bar=progress_bar,
    )
    all_np = np.asarray(jax.device_get(all_v), dtype=np.float64)
    return KozBRTResult6D(
        times_s=times,
        values=all_np,
        domain_lo=np.asarray(domain_lo, dtype=np.float64),
        domain_hi=np.asarray(domain_hi, dtype=np.float64),
        grid_shape=tuple(int(x) for x in grid_shape),
        n_rad_s=float(n_rad_s),
        u_max_m_s2=float(u_max_m_s2),
        d_max_m_s2=float(d_max_m_s2),
    )


def save_koz_brt_6d_npz(path: str, result: KozBRTResult6D) -> None:
    np.savez_compressed(
        path,
        times_s=result.times_s,
        values=result.values,
        domain_lo=result.domain_lo,
        domain_hi=result.domain_hi,
        grid_shape=np.array(result.grid_shape, dtype=np.int32),
        n_rad_s=result.n_rad_s,
        u_max_m_s2=result.u_max_m_s2,
        d_max_m_s2=result.d_max_m_s2,
    )


def load_koz_brt_6d_npz(path: str) -> KozBRTResult6D:
    z = np.load(path)
    shape = tuple(int(x) for x in z["grid_shape"])
    return KozBRTResult6D(
        times_s=np.asarray(z["times_s"]),
        values=np.asarray(z["values"]),
        domain_lo=np.asarray(z["domain_lo"]),
        domain_hi=np.asarray(z["domain_hi"]),
        grid_shape=shape,
        n_rad_s=float(z["n_rad_s"]),
        u_max_m_s2=float(z["u_max_m_s2"]),
        d_max_m_s2=float(z["d_max_m_s2"]),
    )
