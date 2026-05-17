"""2D x–y value-function heatmaps over backward time (HW2-style evolution) from 6D BRT NPZ or KozBRTResult6D."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from simulation.brt_isosurface import choose_contour_level

if TYPE_CHECKING:
    from simulation.hj_koz_brt import KozBRTResult6D


def _grid_axes(domain_lo: np.ndarray, domain_hi: np.ndarray, grid_shape: tuple[int, ...]) -> list[np.ndarray]:
    """Match ``hj_reachability.Grid`` non-periodic axes: ``linspace(lo, hi, n, endpoint=True)`` per dim."""
    out: list[np.ndarray] = []
    lo = np.asarray(domain_lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(domain_hi, dtype=np.float64).reshape(-1)
    for d in range(len(grid_shape)):
        n = int(grid_shape[d])
        out.append(np.linspace(float(lo[d]), float(hi[d]), n, endpoint=True, dtype=np.float64))
    return out


def _nearest_axis_index(axis: np.ndarray, target: float) -> int:
    a = np.asarray(axis, dtype=np.float64).reshape(-1)
    return int(np.argmin(np.abs(a - float(target))))


def _load_result(obj: Any) -> "KozBRTResult6D":
    if isinstance(obj, str):
        from simulation.hj_koz_brt import load_koz_brt_6d_npz

        return load_koz_brt_6d_npz(obj)
    from simulation.hj_koz_brt import KozBRTResult6D

    if isinstance(obj, KozBRTResult6D):
        return obj
    raise TypeError("brt_source must be KozBRTResult6D or path to .npz")


def render_brt_xy_value_evolution_gif(
    brt_source: Any,
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
) -> str | None:
    """Animate HJ value ``V(x,y)`` on the x–y plane at fixed ``z, vx, vy, vz`` for each backward-time slice.

    ``brt_source`` is :class:`~simulation.hj_koz_brt.KozBRTResult6D` or a path to ``brt_koz_collision_6d.npz``.
    ``values`` layout matches the HJ grid ``(t, nx, ny, nz, nvx, nvy, nvz)`` with ``ij`` meshgrid order
    (first axis = x, second = y).

    Each frame shows ``imshow`` of the slice, ``V=0`` contour (BRT boundary in this slice), and optional
    inner-KOZ ``s=1`` contour in the same slice (``inner_koz``).

    Returns path to written GIF, or ``None`` if fewer than two time slices.
    """
    import matplotlib

    if os.environ.get("MPLBACKEND", "").lower() == "agg" or os.environ.get("CI"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import TwoSlopeNorm

    res = _load_result(brt_source)
    vals = np.asarray(res.values, dtype=np.float64)
    times = np.asarray(res.times_s, dtype=np.float64)
    lo = np.asarray(res.domain_lo, dtype=np.float64).reshape(6)
    hi = np.asarray(res.domain_hi, dtype=np.float64).reshape(6)
    gshape = tuple(int(x) for x in res.grid_shape)
    if vals.ndim != 7:
        raise ValueError(f"expected values shaped (T, nx, ny, nz, nvx, nvy, nvz); got {vals.shape}")
    nt = vals.shape[0]
    if nt < 2:
        warnings.warn("BRT evolution GIF needs at least 2 time slices in values.", UserWarning)
        return None

    axes = _grid_axes(lo, hi, gshape)
    iz = _nearest_axis_index(axes[2], slice_z_m)
    ivx = _nearest_axis_index(axes[3], slice_vx_m_s)
    ivy = _nearest_axis_index(axes[4], slice_vy_m_s)
    ivz = _nearest_axis_index(axes[5], slice_vz_m_s)

    x_coords = axes[0]
    y_coords = axes[1]
    z_at = float(axes[2][iz])
    vx_at = float(axes[3][ivx])
    vy_at = float(axes[4][ivy])
    vz_at = float(axes[5][ivz])

    slabs = np.stack([vals[k, :, :, iz, ivx, ivy, ivz] for k in range(nt)], axis=0)
    if vmax_abs is None:
        vmax_abs = float(np.percentile(np.abs(slabs[np.isfinite(slabs)]), 99.5))
        vmax_abs = max(vmax_abs, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)

    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))
    E = inner_koz.metric() if inner_koz is not None else None
    c = inner_koz.center if inner_koz is not None else None

    def _koz_shape_xy_plane() -> np.ndarray | None:
        if E is None or c is None:
            return None
        Zc = np.full_like(X, z_at, dtype=np.float64)
        pts = np.stack([X.ravel(), Y.ravel(), Zc.ravel()], axis=1) - np.asarray(c, dtype=np.float64).reshape(1, 3)
        return np.einsum("ni,ij,nj->n", pts, np.asarray(E, dtype=np.float64), pts).reshape(X.shape)

    dep_xy: tuple[float, float] | None = None
    if deputy_pos_m is not None:
        r = np.asarray(deputy_pos_m, dtype=np.float64).reshape(3)
        if abs(r[2] - z_at) <= 1.5 * max(
            float(np.diff(axes[2]).max()) if axes[2].size > 1 else 600.0, 1.0
        ):
            dep_xy = (float(r[0]), float(r[1]))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slabs[0].T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=colormap,
        norm=norm,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("HJ value V (≤0 ⟺ inside BRT at this τ)")

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
        lvl, lvl_tag = choose_contour_level(slab, 0.0)
        ax.contour(X, Y, slab, levels=[lvl], colors="k", linewidths=1.2)
        if E is not None and c is not None:
            s = _koz_shape_xy_plane()
            if s is not None:
                ax.contour(X, Y, s, levels=[1.0], colors="crimson", linewidths=1.0, linestyles="--")
        tau = float(times[k])
        frac_u = float(np.mean(unsafe_mask))
        ax.set_xlabel("x LVLH (m)")
        ax.set_ylabel("y LVLH (m)")
        ax.set_title(
            f"Backward BRT on x–y slice  (z={z_at:.0f} m, v≈0)\n"
            f"τ = {tau:.0f} s from terminal  |  black: V={lvl:.3g} ({lvl_tag})  |  "
            f"grid V≤0: {frac_u * 100:.2f}%  |  |τ| grows → BRT grows",
            fontsize=9,
        )
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

    def update(j: int) -> tuple:
        _set_frame(frame_indices[int(j)])
        return (im,)

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=max(1, int(1000.0 / float(fps))),
        blit=False,
    )
    try:
        anim.save(out, writer="pillow", fps=float(fps), dpi=int(dpi))
    finally:
        plt.close(fig)
    return out
