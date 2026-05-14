"""Closed-form CW/Hill solution (position + velocity).

Adapted from the uploaded ``Clohessy-Wiltshire/cw2.py`` (Bonar, 2018), expressed in
the same LVLH convention as ``cw_dynamics``: x radial outward from Earth, y along-track,
z cross-track, ``n`` mean motion (rad/s), SI meters and m/s.

This is mathematically equivalent to :math:`x(t)=\\exp(A t)x_0` for the standard CW
``A`` matrix; it removes per-step ``expm`` cost for long coast sampling.
"""

from __future__ import annotations

import numpy as np


def cw_state_closed_form(x0: np.ndarray, n: float, t: float) -> np.ndarray:
    """Return full state x(t) = [x,y,z,vx,vy,vz] from x0 at t >= 0."""
    x0 = np.asarray(x0, dtype=np.float64).reshape(6)
    n = float(n)
    t = float(t)
    x0p, y0p, z0p = x0[0], x0[1], x0[2]
    xdot0, ydot0, zdot0 = x0[3], x0[4], x0[5]

    sn = np.sin(n * t)
    cs = np.cos(n * t)

    xt = (4.0 * x0p + (2.0 * ydot0) / n) + (xdot0 / n) * sn - (3.0 * x0p + (2.0 * ydot0) / n) * cs
    yt = (
        (y0p - (2.0 * xdot0) / n)
        + ((2.0 * xdot0) / n) * cs
        + (6.0 * x0p + (4.0 * ydot0) / n) * sn
        - (6.0 * n * x0p + 3.0 * ydot0) * t
    )
    zt = z0p * cs + (zdot0 / n) * sn

    xdott = (3.0 * n * x0p + 2.0 * ydot0) * sn + xdot0 * cs
    ydott = (6.0 * n * x0p + 4.0 * ydot0) * cs - 2.0 * xdot0 * sn - (6.0 * n * x0p + 3.0 * ydot0)
    zdott = zdot0 * cs - z0p * n * sn

    return np.array([xt, yt, zt, xdott, ydott, zdott], dtype=np.float64)
