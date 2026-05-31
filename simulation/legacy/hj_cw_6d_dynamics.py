"""Full 6D Hill–Clohessy state for hj_reachability: ``[x,y,z,vx,vy,vz]`` with thrust acceleration ``u∈ℝ³``."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_HJ_ROOT = _ROOT / "hj_reachability"
if _HJ_ROOT.is_dir():
    _p = str(_HJ_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hj_reachability import dynamics, sets

from simulation.cw_dynamics import CWDynamics


def cw_A_jax(n: float) -> jnp.ndarray:
    """Same 6×6 drift matrix as :class:`~simulation.cw_dynamics.CWDynamics`."""
    plant = CWDynamics(float(n))
    return jnp.array(plant.continuous_A(), dtype=jnp.float64)


class CwFull6DHJDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    """``ẋ = A x + B u + G d`` with ``B = G = [0_{3×3}; I₃]`` (acceleration in LVLH).

    Same min–max game as the planar model: control ``min``, disturbance ``max``.
    """

    def __init__(
        self,
        n: float,
        u_max_m_s2: float,
        d_max_m_s2: float = 0.0,
        *,
        control_mode: str = "min",
        disturbance_mode: str = "max",
    ) -> None:
        self._A = cw_A_jax(float(n))
        u = float(abs(u_max_m_s2))
        d = float(abs(d_max_m_s2))
        control_space = sets.Box(
            jnp.array([-u, -u, -u], dtype=jnp.float64),
            jnp.array([u, u, u], dtype=jnp.float64),
        )
        disturbance_space = sets.Box(
            jnp.array([-d, -d, -d], dtype=jnp.float64),
            jnp.array([d, d, d], dtype=jnp.float64),
        )
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        del time
        return jnp.einsum("ij,...j->...i", self._A, state)

    def control_jacobian(self, state, time):
        del state, time
        return jnp.vstack(
            [
                jnp.zeros((3, 3), dtype=jnp.float64),
                jnp.eye(3, dtype=jnp.float64),
            ]
        )

    def disturbance_jacobian(self, state, time):
        return self.control_jacobian(state, time)
