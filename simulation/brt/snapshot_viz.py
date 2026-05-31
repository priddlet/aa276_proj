"""Static PNG / GIF of Option-1 BRT unsafe set (V≤0) in chief LVLH — zoomed on the target (chief)."""

from __future__ import annotations

import os
import time as time_mod
import warnings
from typing import Any

import numpy as np

from simulation.brt.isosurface import extract_brt_v0_near_center
from simulation.keepout import EllipsoidKeepOut
from simulation.spacecraft_wire import bus_and_panel_edges, edges_to_nan_polyline, scale_edges


def _log(msg: str) -> None:
    print(msg, flush=True)


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


def _axis_fit_geometry(
    ax,
    center: np.ndarray,
    *,
    mesh_lo: np.ndarray | None,
    mesh_hi: np.ndarray | None,
    fallback_half: float,
    koz_pad_m: float,
) -> None:
    """Frame axes to the BRT mesh (avoids clipping an elongated tube in a centered cube)."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    pad_min = float(max(koz_pad_m, 40.0))
    if mesh_lo is not None and mesh_hi is not None:
        lo = np.asarray(mesh_lo, dtype=np.float64).reshape(3)
        hi = np.asarray(mesh_hi, dtype=np.float64).reshape(3)
        span = np.maximum(hi - lo, pad_min)
        pad = np.maximum(0.12 * span, pad_min)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
        try:
            ax.set_box_aspect(tuple(float(h - l) for l, h in zip(lo - pad, hi + pad)))
        except Exception:
            pass
        return
    _axis_cube_3d(ax, c, fallback_half)


def _extract_snapshot_surface(brt: Any, o0: np.ndarray, x6: np.ndarray) -> dict[str, Any]:
    lo6 = np.asarray(getattr(brt, "domain_lo"), dtype=np.float64).reshape(6)
    hi6 = np.asarray(getattr(brt, "domain_hi"), dtype=np.float64).reshape(6)
    koz_max = 45.0
    display_r = max(1.35 * koz_max, 45.0)
    chief_half = float(os.environ.get("BRT_SNAPSHOT_CHIEF_HALF_M", "1200"))
    max_search = float(os.environ.get("BRT_SNAPSHOT_MAX_HALF_M", "6000"))
    iso_snap = os.environ.get("BRT_SNAPSHOT_ISO", "28,28,22").strip()
    parts = [int(x) for x in iso_snap.split(",")]
    if len(parts) != 3:
        raise ValueError("BRT_SNAPSHOT_ISO must be three comma-separated integers")

    t0 = time_mod.perf_counter()
    if not hasattr(brt, "value_batch"):
        warnings.warn("brt has no value_batch; BRT surface skipped.", UserWarning)
        surf = {
            "mesh_verts": None,
            "mesh_faces": None,
            "footprint_polys": [],
            "contour_lines": [],
        }
    else:
        _log(
            f"  Extracting BRT shell via DeepReach (iso {parts[0]}×{parts[1]}×{parts[2]}, "
            f"box half up to {max_search:.0f} m)…"
        )
        surf = extract_brt_v0_near_center(
            brt.value_batch,
            o0,
            float(x6[3]),
            float(x6[4]),
            float(x6[5]),
            domain_lo=lo6,
            domain_hi=hi6,
            initial_half_m=max(chief_half, 2.5 * koz_max),
            max_half_m=max_search,
            iso_resolution=(parts[0], parts[1], parts[2]),
            display_radius_m=display_r,
            contour_half_m=display_r,
            contour_n2d=int(os.environ.get("BRT_SNAPSHOT_CONTOUR_N", "36")),
        )
    nf = 0
    if surf.get("mesh_faces") is not None:
        nf = int(np.asarray(surf["mesh_faces"]).shape[0])
    n_loops = len(surf.get("slice_loops_3d") or [])
    backend = str(surf.get("mesh_backend", "?"))
    _log(
        f"  Surface ready in {time_mod.perf_counter() - t0:.1f} s "
        f"(3D backend={backend}, mesh faces={nf}, z-slice loops={n_loops})."
    )
    if backend == "stacked_slices":
        _log("  (Install scikit-image for a solid 3D shell: pip install scikit-image)")
    return surf


def render_brt_lvlh_snapshot(
    brt: Any,
    inner_koz: EllipsoidKeepOut,
    x_deputy_lvlh_m: np.ndarray,
    *,
    output_path_png: str,
    output_path_gif: str | None = None,
    chief_box_half_m: float = 1200.0,
    max_search_half_m: float = 6000.0,
    iso_resolution: tuple[int, int, int] = (28, 28, 22),
    dpi: int = 120,
    gif_frames: int = 16,
    gif_fps: float = 8.0,
) -> tuple[str | None, str | None]:
    """LVLH plot (meters): inner KOZ (terminal) + **physical BRT** ``{V ≤ 0}`` near the chief."""
    import matplotlib

    if os.environ.get("MPLBACKEND", "").lower() == "agg" or os.environ.get("CI"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x6 = np.asarray(x_deputy_lvlh_m, dtype=np.float64).reshape(6)
    o0 = np.zeros(3, dtype=np.float64)
    koz_max = float(np.max(inner_koz.semi_axes))
    display_r = max(1.35 * koz_max, 45.0)

    surf: dict[str, Any] = {
        "mesh_verts": None,
        "mesh_faces": None,
        "footprint_polys": [],
        "contour_lines": [],
    }
    if hasattr(brt, "value_batch"):
        os.environ.setdefault("BRT_SNAPSHOT_CHIEF_HALF_M", str(chief_box_half_m))
        os.environ.setdefault(
            "BRT_SNAPSHOT_ISO",
            f"{iso_resolution[0]},{iso_resolution[1]},{iso_resolution[2]}",
        )
        surf = _extract_snapshot_surface(brt, o0, x6)
    else:
        warnings.warn("brt has no value_batch; BRT surface skipped.", UserWarning)

    chief_edges = scale_edges(bus_and_panel_edges((10.0, 4.0, 4.0), panel_length=22.0, panel_half_width=2.5), 1.0)
    half_view = float(surf.get("view_half_m", display_r * 1.08))
    view_cap = float(os.environ.get("BRT_SNAPSHOT_VIEW_HALF_M", "2800"))
    half_view = min(half_view, view_cap)
    vm = surf.get("mesh_verts")
    if vm is not None and np.asarray(vm).size > 0:
        half_view = min(
            view_cap,
            max(half_view, float(np.max(np.linalg.norm(np.asarray(vm) - o0, axis=1))) * 1.05),
        )

    brt_label_added = False

    def _draw_brt(ax) -> None:
        nonlocal brt_label_added
        for poly in surf.get("footprint_polys") or []:
            p3 = np.asarray(poly, dtype=np.float64)
            if p3.shape[0] >= 3:
                coll_fp = Poly3DCollection(
                    [p3],
                    facecolors=(0.55, 0.15, 0.75, 0.22),
                    edgecolors=(0.4, 0.05, 0.55, 0.55),
                    linewidths=0.35,
                )
                ax.add_collection3d(coll_fp)
                if not brt_label_added:
                    coll_fp.set_label("BRT")
                    brt_label_added = True
        if surf.get("mesh_verts") is not None and surf.get("mesh_faces") is not None:
            verts_m = np.asarray(surf["mesh_verts"], dtype=np.float64)
            faces = np.asarray(surf["mesh_faces"], dtype=np.int64)
            if faces.size > 0:
                polys = [verts_m[fk] for fk in faces]
                coll = Poly3DCollection(
                    polys,
                    facecolors=(0.55, 0.15, 0.75, 0.32),
                    edgecolors=(0.32, 0.04, 0.48, 0.42),
                    linewidths=0.22,
                )
                if not brt_label_added:
                    coll.set_label("BRT")
                    brt_label_added = True
                try:
                    coll.set_zsort("average")
                except Exception:
                    pass
                ax.add_collection3d(coll)
        # With a closed mesh, skip the extra z=0 contour and filled footprints (they look like a cutting plane).
        if surf.get("mesh_verts") is not None:
            loop_lines = list(surf.get("slice_loops_3d") or [])
        else:
            loop_lines = list(surf.get("slice_loops_3d") or []) + list(surf.get("contour_lines") or [])
        for j, line in enumerate(loop_lines):
            ln = np.asarray(line, dtype=np.float64)
            if ln.shape[0] >= 2:
                lbl = (
                    "BRT"
                    if j == 0 and surf.get("mesh_verts") is None and not brt_label_added
                    else None
                )
                ax.plot(
                    ln[:, 0],
                    ln[:, 1],
                    ln[:, 2],
                    color="blueviolet",
                    linewidth=0.85 if surf.get("mesh_verts") is None else 0.55,
                    alpha=0.75 if surf.get("mesh_verts") is None else 0.45,
                    linestyle="-",
                    label=lbl,
                )
                if lbl:
                    brt_label_added = True

    def _draw_frame(ax, title_suffix: str = "") -> None:
        ax.clear()
        Xm, Ym, Zm = inner_koz.surface_mesh(nu=32, nv=18)
        ax.plot_wireframe(
            Xm,
            Ym,
            Zm,
            color="crimson",
            linewidth=0.65,
            alpha=0.96,
            rstride=1,
            cstride=1,
            label="KOZ",
        )
        _draw_brt(ax)
        I = np.eye(3, dtype=np.float64)
        xs, ys, zs = edges_to_nan_polyline(o0, I, chief_edges)
        ax.plot(xs, ys, zs, color="0.2", linewidth=1.0, label="Chief")
        ax.scatter([0.0], [0.0], [0.0], color="0.15", s=42, depthshade=True, zorder=5)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("BRT + KOZ" + title_suffix, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        _axis_fit_geometry(
            ax,
            o0,
            mesh_lo=surf.get("mesh_bounds_lo"),
            mesh_hi=surf.get("mesh_bounds_hi"),
            fallback_half=half_view,
            koz_pad_m=2.5 * koz_max,
        )

    fig = plt.figure(figsize=(9.0, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    _draw_frame(ax)
    fig.tight_layout()
    outp = os.path.abspath(output_path_png)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    _log("  Writing PNG…")
    t_png = time_mod.perf_counter()
    fig.savefig(outp, dpi=int(dpi))
    _log(f"  PNG done in {time_mod.perf_counter() - t_png:.1f} s.")
    png_written = outp

    gif_written: str | None = None
    if output_path_gif and int(gif_frames) >= 2:
        outg = os.path.abspath(output_path_gif)
        os.makedirs(os.path.dirname(outg) or ".", exist_ok=True)
        _draw_frame(ax, "")
        _log(f"  Writing GIF ({int(gif_frames)} frames @ {gif_fps} fps)…")
        t_gif = time_mod.perf_counter()

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
            _log(f"  GIF done in {time_mod.perf_counter() - t_gif:.1f} s.")
        except Exception as exc:
            warnings.warn(f"BRT snapshot GIF failed: {exc}", UserWarning)

    plt.close(fig)
    return png_written, gif_written
