"""Simple bus + solar-panel wireframe (body frame) for 3D plots."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def bus_and_panel_edges(
    bus_half: tuple[float, float, float],
    panel_length: float,
    panel_half_width: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed segments in body frame: +x ram, panels extend along ±y from bus."""
    hx, hy, hz = bus_half
    edges: list[tuple[np.ndarray, np.ndarray]] = []

    def add_box(hx_: float, hy_: float, hz_: float) -> None:
        c = [
            (-hx_, -hy_, -hz_),
            (+hx_, -hy_, -hz_),
            (+hx_, +hy_, -hz_),
            (-hx_, +hy_, -hz_),
            (-hx_, -hy_, +hz_),
            (+hx_, -hy_, +hz_),
            (+hx_, +hy_, +hz_),
            (-hx_, +hy_, +hz_),
        ]
        c = [np.array(p, dtype=np.float64) for p in c]
        for a, b in (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ):
            edges.append((c[a], c[b]))

    add_box(hx, hy, hz)
    y0 = hy
    pl = panel_length
    pw = panel_half_width
    for sign in (+1.0, -1.0):
        sy = sign * y0
        rect = [
            (-pw, sy, -hz * 0.1),
            (-pw, sy + sign * pl, -hz * 0.1),
            (+pw, sy + sign * pl, -hz * 0.1),
            (+pw, sy, -hz * 0.1),
        ]
        rect = [np.array(p, dtype=np.float64) for p in rect]
        for k in range(4):
            edges.append((rect[k], rect[(k + 1) % 4]))

    return edges


def scale_edges(edges: list[tuple[np.ndarray, np.ndarray]], scale: float) -> list[tuple[np.ndarray, np.ndarray]]:
    s = float(scale)
    return [(s * p0.copy(), s * p1.copy()) for p0, p1 in edges]


def edges_to_nan_polyline(
    origin: np.ndarray,
    R_body_to_world: np.ndarray,
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform body-frame edge endpoints to world frame; one broken line for Line3D."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    R = np.asarray(R_body_to_world, dtype=np.float64).reshape(3, 3)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for p0, p1 in edges:
        q0 = o + R @ np.asarray(p0, dtype=np.float64).reshape(3)
        q1 = o + R @ np.asarray(p1, dtype=np.float64).reshape(3)
        xs.extend([q0[0], q1[0], float("nan")])
        ys.extend([q0[1], q1[1], float("nan")])
        zs.extend([q0[2], q1[2], float("nan")])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(zs, dtype=np.float64)
