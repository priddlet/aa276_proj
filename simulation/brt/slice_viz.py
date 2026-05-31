"""2D x–y value-function heatmaps over backward time from DeepReach BRT."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from simulation.brt.config import SLICE_GRID_NX, SLICE_GRID_NY, SLICE_TIME_NODES
from simulation.brt.isosurface import choose_contour_level


def render_brt_xy_value_evolution_gif(
    brt: Any,
    *,
    output_path: str,
    inner_koz: Any | None = None,
    slice_z_m: float = 0.0,
    slice_vx_m_s: float = 0.0,
    slice_vy_m_s: float = 0.0,
    slice_vz_m_s: float = 0.0,
    fps: float = 2.0,
    dpi: int = 140,
    figsize: tuple[float, float] = (8.2, 6.8),
    colormap: str = "RdBu_r",
    vmax_abs: float | None = None,
    hold_last_frames: int = 6,
    deputy_pos_m: np.ndarray | None = None,
    n_time_nodes: int | None = None,
    grid_nx: int | None = None,
    grid_ny: int | None = None,
) -> str | None:
    """Animate ``V(x,y)`` on the x–y plane at fixed ``z, vx, vy, vz`` for each backward-time slice."""
    import matplotlib

    if os.environ.get("MPLBACKEND", "").lower() == "agg" or os.environ.get("CI"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import TwoSlopeNorm

    if not hasattr(brt, "value_batch_at_tau") and not hasattr(brt, "backward_times_s"):
        raise TypeError("brt must be KozDeepReachBRT (value_batch_at_tau)")

    lo = np.asarray(brt.domain_lo, dtype=np.float64).reshape(6)
    hi = np.asarray(brt.domain_hi, dtype=np.float64).reshape(6)
    nt = int(n_time_nodes or os.environ.get("BRT_SLICE_TIME_NODES", str(SLICE_TIME_NODES)))
    nx = int(grid_nx or os.environ.get("BRT_SLICE_NX", str(SLICE_GRID_NX)))
    ny = int(grid_ny or os.environ.get("BRT_SLICE_NY", str(SLICE_GRID_NY)))

    times = brt.backward_times_s(nt)
    xs = np.linspace(float(lo[0]), float(hi[0]), nx, dtype=np.float64)
    ys = np.linspace(float(lo[1]), float(hi[1]), ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]

    slabs: list[np.ndarray] = []
    for tau in times:
        pts = np.stack(
            [
                X.ravel(),
                Y.ravel(),
                np.full(X.size, float(slice_z_m)),
                np.full(X.size, float(slice_vx_m_s)),
                np.full(X.size, float(slice_vy_m_s)),
                np.full(X.size, float(slice_vz_m_s)),
            ],
            axis=-1,
        )
        tau_pos = abs(float(tau))
        v = brt.value_batch_at_tau(pts, tau_pos)
        slabs.append(v.reshape(nx, ny))

    finite = np.concatenate([s[np.isfinite(s)] for s in slabs if np.any(np.isfinite(s))])
    if finite.size == 0:
        warnings.warn("No finite DeepReach values on slice; skipping evolution GIF.", UserWarning)
        return None
    cap = float(vmax_abs) if vmax_abs is not None else float(np.percentile(np.abs(finite), 98))
    cap = max(cap, 1e-3)
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)

    E = c = None
    if inner_koz is not None:
        E = np.asarray(inner_koz.metric(), dtype=np.float64)
        c = np.asarray(inner_koz.center, dtype=np.float64).reshape(3)

    def _koz_shape_xy_plane() -> np.ndarray | None:
        if E is None or c is None:
            return None
        dx = X - c[0]
        dy = Y - c[1]
        dz = float(slice_z_m) - c[2]
        return (
            E[0, 0] * dx * dx
            + E[1, 1] * dy * dy
            + E[2, 2] * dz * dz
            + 2 * E[0, 1] * dx * dy
            + 2 * E[0, 2] * dx * dz
            + 2 * E[1, 2] * dy * dz
        )

    dep_xy = None
    if deputy_pos_m is not None:
        r = np.asarray(deputy_pos_m, dtype=np.float64).reshape(3)
        if float(lo[0]) <= r[0] <= float(hi[0]) and float(lo[1]) <= r[1] <= float(hi[1]):
            dep_xy = (float(r[0]), float(r[1]))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slabs[0].T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=colormap,
        norm=norm,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("V")

    def _clear_contours() -> None:
        while ax.collections:
            ax.collections[0].remove()

    def _set_frame(k: int) -> None:
        k = int(k) % nt
        slab = slabs[k]
        im.set_data(slab.T)
        im.set_norm(norm)
        _clear_contours()
        unsafe_mask = slab <= 0.0
        if np.any(unsafe_mask):
            ax.contourf(
                X,
                Y,
                unsafe_mask.astype(float),
                levels=[0.5, 1.5],
                colors=[(0.55, 0.15, 0.75, 0.18)],
                antialiased=False,
            )
        lvl, _ = choose_contour_level(slab, 0.0)
        ax.contour(X, Y, slab, levels=[lvl], colors="k", linewidths=1.2)
        s = _koz_shape_xy_plane()
        if s is not None:
            ax.contour(X, Y, s, levels=[1.0], colors="crimson", linewidths=1.0, linestyles="--")
        tau = float(times[k])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"BRT (x–y slice), τ = {tau:.0f} s", fontsize=10)
        if dep_xy is not None:
            ax.scatter(
                [dep_xy[0]],
                [dep_xy[1]],
                s=55,
                c="gold",
                edgecolors="0.15",
                linewidths=0.8,
                zorder=6,
            )

    hold = max(0, int(hold_last_frames))
    frame_indices = list(range(nt)) + [nt - 1] * hold

    anim = animation.FuncAnimation(fig, _set_frame, frames=frame_indices, interval=max(1, int(1000.0 / fps)))
    _set_frame(0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    try:
        anim.save(output_path, writer="pillow", fps=float(fps), dpi=int(dpi))
    except Exception as exc:
        warnings.warn(f"BRT evolution GIF failed: {exc}", UserWarning)
        plt.close(fig)
        return None
    plt.close(fig)
    return output_path
