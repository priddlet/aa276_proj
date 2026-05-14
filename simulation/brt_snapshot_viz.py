"""Static PNG / GIF of Option-1 BRT boundary (V≈0) in chief LVLH — zoomed on the target (chief)."""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from simulation.keepout import EllipsoidKeepOut
from simulation.orbit_movie import brt_position_isosurface_lvlh_m
from simulation.spacecraft_wire import bus_and_panel_edges, edges_to_nan_polyline, scale_edges


def _axis_cube_3d(ax, center: np.ndarray, half: float) -> None:
    """Equal square axis limits: ``center ± half`` on each axis."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = float(max(half, 1.0))
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render_brt_lvlh_snapshot(
    hj_tab: Any,
    inner_koz: EllipsoidKeepOut,
    x_deputy_lvlh_m: np.ndarray,
    *,
    output_path_png: str,
    output_path_gif: str | None = None,
    chief_box_half_m: float = 380.0,
    iso_resolution: tuple[int, int, int] = (44, 44, 32),
    dpi: int = 150,
    gif_frames: int = 40,
    gif_fps: float = 10.0,
) -> tuple[str | None, str | None]:
    """LVLH plot (meters): inner KOZ + V=0 BRT shell at deputy velocity, **chief (target) only** in frame.

    Marching cubes samples ``[-chief_box_half_m, +chief_box_half_m]^3`` in position (intersected with the
    HJ grid), centered on the chief at LVLH origin. Axis limits are tight around the origin from KOZ size
    and BRT vertices near the target (deputy mesh / markers omitted so the view stays zoomed on the KOZ).

    Returns ``(png_path_or_none, gif_path_or_none)``. GIF omitted if ``output_path_gif`` is ``None`` or
    ``gif_frames < 2``. Requires ``scikit-image`` for marching cubes.
    """
    import matplotlib

    if os.environ.get("MPLBACKEND", "").lower() == "agg" or os.environ.get("CI"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x6 = np.asarray(x_deputy_lvlh_m, dtype=np.float64).reshape(6)
    v_d = x6[3:6].copy()
    lo6 = np.asarray(getattr(hj_tab, "domain_lo"), dtype=np.float64).reshape(6)
    hi6 = np.asarray(getattr(hj_tab, "domain_hi"), dtype=np.float64).reshape(6)
    half_box = float(max(50.0, chief_box_half_m))

    o0 = np.zeros(3, dtype=np.float64)
    x_lo = max(lo6[0], float(o0[0] - half_box))
    x_hi = min(hi6[0], float(o0[0] + half_box))
    y_lo = max(lo6[1], float(o0[1] - half_box))
    y_hi = min(hi6[1], float(o0[1] + half_box))
    z_lo = max(lo6[2], float(o0[2] - half_box))
    z_hi = min(hi6[2], float(o0[2] + half_box))
    if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
        warnings.warn("BRT snapshot chief box vs HJ domain collapsed; using inner KOZ extent only.", UserWarning)
        smax = float(np.max(inner_koz.semi_axes))
        x_lo, x_hi = -2.5 * smax, 2.5 * smax
        y_lo, y_hi = -2.5 * smax, 2.5 * smax
        z_lo, z_hi = -2.5 * smax, 2.5 * smax

    nx, ny, nz = (max(10, int(iso_resolution[i])) for i in range(3))
    iso = None
    if hasattr(hj_tab, "value_batch"):
        try:
            iso = brt_position_isosurface_lvlh_m(
                hj_tab.value_batch,
                float(x6[3]),
                float(x6[4]),
                float(x6[5]),
                float(x_lo),
                float(x_hi),
                float(y_lo),
                float(y_hi),
                float(z_lo),
                float(z_hi),
                int(nx),
                int(ny),
                int(nz),
                level=0.0,
            )
        except Exception as exc:
            warnings.warn(f"BRT isosurface extraction failed: {exc}", UserWarning)

    chief_edges = scale_edges(bus_and_panel_edges((10.0, 4.0, 4.0), panel_length=22.0, panel_half_width=2.5), 1.0)
    koz_max = float(np.max(inner_koz.semi_axes))
    # Tight zoom on chief: KOZ + nearby BRT shell only (ignore deputy range).
    pad = 1.12
    half_view = max(1.35 * koz_max, 45.0)
    if iso is not None and iso[1].size > 0:
        vm = np.asarray(iso[0], dtype=np.float64)
        d0 = np.linalg.norm(vm, axis=1)
        near = d0 <= (1.35 * half_box + 1e-9)
        if np.any(near):
            vm_n = vm[near]
            extent = float(np.max(np.abs(vm_n)))
            half_view = max(half_view, pad * extent)
        else:
            extent = float(np.max(np.linalg.norm(vm, axis=1)))
            half_view = min(half_box, max(half_view, pad * 0.35 * extent))
    half_view = float(min(half_view, half_box * 1.02))

    def _draw_frame(ax, title_suffix: str = "") -> Poly3DCollection | None:
        ax.clear()
        Xm, Ym, Zm = inner_koz.surface_mesh(nu=48, nv=26)
        ax.plot_wireframe(Xm, Ym, Zm, color="crimson", linewidth=0.65, alpha=0.96, rstride=1, cstride=1)
        brt_coll = None
        if iso is not None:
            verts_m, faces = iso
            if faces.size > 0:
                polys = [verts_m[fk] for fk in faces]
                brt_coll = Poly3DCollection(
                    polys,
                    facecolors=(0.55, 0.15, 0.75, 0.14),
                    edgecolors=(0.45, 0.08, 0.6, 0.4),
                    linewidths=0.18,
                )
                ax.add_collection3d(brt_coll)
        I = np.eye(3, dtype=np.float64)
        xs, ys, zs = edges_to_nan_polyline(o0, I, chief_edges)
        ax.plot(xs, ys, zs, color="0.2", linewidth=1.0, label="Chief (target)")
        ax.scatter([0.0], [0.0], [0.0], color="0.15", s=42, depthshade=True, zorder=5)
        ax.set_xlabel("x LVLH (m)")
        ax.set_ylabel("y LVLH (m)")
        ax.set_zlabel("z LVLH (m)")
        ttl = "Option 1 BRT (V=0) + inner KOZ — zoom on chief (LVLH origin)"
        if iso is None or (iso is not None and iso[1].size == 0):
            ttl += " — BRT mesh missing (scikit-image / res / box half)"
        ax.set_title(ttl + title_suffix, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        _axis_cube_3d(ax, o0, half_view)
        return brt_coll

    fig = plt.figure(figsize=(9.0, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    _draw_frame(ax)
    fig.tight_layout()
    outp = os.path.abspath(output_path_png)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    fig.savefig(outp, dpi=int(dpi))
    png_written = outp

    gif_written: str | None = None
    if output_path_gif and int(gif_frames) >= 2:
        outg = os.path.abspath(output_path_gif)
        os.makedirs(os.path.dirname(outg) or ".", exist_ok=True)
        _draw_frame(ax, "")

        def step(i: int) -> tuple:
            ax.view_init(elev=24.0, azim=float(i) * 360.0 / float(gif_frames))
            return (ax,)

        anim = animation.FuncAnimation(
            fig,
            step,
            frames=int(gif_frames),
            interval=max(1, int(1000.0 / float(gif_fps))),
            blit=False,
        )
        try:
            anim.save(outg, writer="pillow", fps=float(gif_fps), dpi=int(dpi))
            gif_written = outg
        except Exception as exc:
            warnings.warn(f"BRT snapshot GIF failed: {exc}", UserWarning)

    plt.close(fig)
    return png_written, gif_written
