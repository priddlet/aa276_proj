"""Ellipsoidal keep-out zone (KOZ) in relative position space.

Unsafe / failure: deputy position inside the closed ellipsoid (inclusive boundary).
The safety filter (BRT) will treat this set as the avoid set; here we only provide
geometry checks on relative position r = [x, y, z].
"""

from __future__ import annotations

import numpy as np


class EllipsoidKeepOut:
    """
    Interior of ellipsoid in chief LVLH frame (same axes as CW position).

    Default center at origin (target at chief). Generalized inequality:
        (r - c)^T E (r - c) <= 1
    with E SPD. Equivalent to axes-aligned ellipsoid with semi-axes a,b,c when
    E = diag(1/a^2, 1/b^2, 1/c^2) and c = 0.
    """

    def __init__(
        self,
        semi_axes: np.ndarray,
        center: np.ndarray | None = None,
        rotation_lvlh: np.ndarray | None = None,
    ) -> None:
        """
        semi_axes: (3,) positive semi-axis lengths (meters).
        center: (3,) offset of ellipsoid center in LVLH (meters).
        rotation_lvlh: (3,3) rotation R maps ellipsoid principal frame to LVLH:
            unsafe iff || diag(1/a) R^T (r-c) ||_2 <= 1
            i.e. E = R diag(1/a^2) R^T.
        """
        a = np.asarray(semi_axes, dtype=np.float64).reshape(3)
        if np.any(a <= 0):
            raise ValueError("semi_axes must be strictly positive")
        self._semi = a.copy()
        self._center = (
            np.zeros(3, dtype=np.float64)
            if center is None
            else np.asarray(center, dtype=np.float64).reshape(3)
        )
        if rotation_lvlh is None:
            self._R = np.eye(3, dtype=np.float64)
        else:
            R = np.asarray(rotation_lvlh, dtype=np.float64).reshape(3, 3)
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=1e-9):
                raise ValueError("rotation_lvlh must be orthogonal")
            self._R = R
        inv_s = 1.0 / self._semi
        self._E = self._R @ np.diag(inv_s * inv_s) @ self._R.T

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def semi_axes(self) -> np.ndarray:
        return self._semi.copy()

    def metric(self) -> np.ndarray:
        """SPD matrix E such that interior is (r-c)^T E (r-c) <= 1."""
        return self._E.copy()

    def shape_value(self, r: np.ndarray) -> float:
        """s(r) = (r-c)^T E (r-c). Interior (unsafe) iff s <= 1."""
        r = np.asarray(r, dtype=np.float64).reshape(3)
        d = r - self._center
        return float(d.T @ self._E @ d)

    def is_inside(self, r: np.ndarray) -> bool:
        """True if relative position is in the closed keep-out ellipsoid."""
        return self.shape_value(r) <= 1.0

    def is_unsafe_position_from_state(self, x6: np.ndarray) -> bool:
        """Convenience: first three components of CW state are position."""
        x6 = np.asarray(x6, dtype=np.float64).reshape(6)
        return self.is_inside(x6[:3])

    def surface_mesh(self, nu: int = 48, nv: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parametric ellipsoid surface in LVLH (meters) for plotting.

        Returns (X, Y, Z) each shaped (nv, nu) on the ellipsoid boundary.
        """
        u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
        v = np.linspace(0.0, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="xy")
        px = np.cos(U) * np.sin(V)
        py = np.sin(U) * np.sin(V)
        pz = np.cos(V)
        P = np.stack([px, py, pz], axis=0)
        scaled = self._semi[:, np.newaxis, np.newaxis] * P
        lvlh = self._center[:, np.newaxis, np.newaxis] + np.einsum("ij,jkl->ikl", self._R, scaled)
        return lvlh[0], lvlh[1], lvlh[2]


class EllipsoidMaxSeparation:
    """Upper bound on relative separation (formation / orbit-envelope proxy).

    **Safe** set is the *interior* '(r-c)^T E (r-c) <= 1' (same ellipsoid metric as
    'EllipsoidKeepOut'). **Unsafe** if the deputy is **outside** that ellipsoid
    '(s > 1)': interpreted as drifting too far from the chief to still be in a
    controlled proximity / orbit-maintenance context for this demo.

    This is a coarse LVLH proxy—not a full altitude-keeping or ROE budget model.
    """

    def __init__(
        self,
        semi_axes: np.ndarray,
        center: np.ndarray | None = None,
        rotation_lvlh: np.ndarray | None = None,
    ) -> None:
        self._geom = EllipsoidKeepOut(semi_axes, center=center, rotation_lvlh=rotation_lvlh)

    def shape_value(self, r: np.ndarray) -> float:
        return self._geom.shape_value(r)

    def is_unsafe_far(self, r: np.ndarray) -> bool:
        """True if position is strictly outside the closed safe ellipsoid."""
        return self.shape_value(r) > 1.0

    def is_unsafe_position_from_state(self, x6: np.ndarray) -> bool:
        x6 = np.asarray(x6, dtype=np.float64).reshape(6)
        return self.is_unsafe_far(x6[:3])

    def surface_mesh(self, nu: int = 48, nv: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._geom.surface_mesh(nu=nu, nv=nv)


class RendezvousSafetyZones:
    """Combined inner proximity KOZ + outer max-separation corridor."""

    def __init__(self, inner: EllipsoidKeepOut, outer: EllipsoidMaxSeparation) -> None:
        self.inner = inner
        self.outer = outer

    def is_unsafe_position(self, r: np.ndarray) -> bool:
        r = np.asarray(r, dtype=np.float64).reshape(3)
        return bool(self.inner.is_inside(r) or self.outer.is_unsafe_far(r))

    def is_unsafe_state(self, x6: np.ndarray) -> bool:
        x6 = np.asarray(x6, dtype=np.float64).reshape(6)
        return self.is_unsafe_position(x6[:3])
