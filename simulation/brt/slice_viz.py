"""x–y value-function slices for learned DeepReach-MPC BRT (full domain or KOZ-centered)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
import time as time_mod
import warnings
from typing import Any

import numpy as np

from simulation.isosurface import choose_contour_level
from simulation.keepout import EllipsoidKeepOut


def default_koz_xy_limits(
    semi_axes_m: tuple[float, float, float] | np.ndarray,
    *,
    x_scale: float = 4.0,
    y_neg_scale: float = 2.5,
    y_pos_scale: float = 12.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """x–y plot window: centered on KOZ, +y room for along-track BRT (m)."""
    a, b, _ = (float(x) for x in np.asarray(semi_axes_m, dtype=np.float64).reshape(3))
    x_half = max(x_scale * a, 120.0)
    y_lo = -max(y_neg_scale * b, 80.0)
    y_hi = max(y_pos_scale * b, 400.0)
    return ((-x_half, x_half), (y_lo, y_hi))


def default_report_koz_xy_limits(
    semi_axes_m: tuple[float, float, float] | np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Tighter LVLH window for report slices: KOZ + along-track approach (~250 m default)."""
    a, b, _ = (float(x) for x in np.asarray(semi_axes_m, dtype=np.float64).reshape(3))
    x_half = max(5.0 * a, 200.0)
    y_lo = -max(2.0 * b, 100.0)
    y_hi = max(8.0 * b, 350.0)
    return ((-x_half, x_half), (y_lo, y_hi))


def parse_xy_limits_from_env(
    semi_axes_m: tuple[float, float, float] | np.ndarray,
    domain_lo: np.ndarray | None = None,
    domain_hi: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``BRT_KOZ_VIZ_X_MIN,X_MAX,Y_MIN,Y_MAX`` override; else KOZ-centered defaults clipped to domain."""
    env_x = os.environ.get("BRT_KOZ_VIZ_X_MIN", "").strip(), os.environ.get("BRT_KOZ_VIZ_X_MAX", "").strip()
    env_y = os.environ.get("BRT_KOZ_VIZ_Y_MIN", "").strip(), os.environ.get("BRT_KOZ_VIZ_Y_MAX", "").strip()
    if all(env_x) and all(env_y):
        xlim = (float(env_x[0]), float(env_x[1]))
        ylim = (float(env_y[0]), float(env_y[1]))
    else:
        xlim, ylim = default_koz_xy_limits(semi_axes_m)
    if domain_lo is not None and domain_hi is not None:
        lo = np.asarray(domain_lo, dtype=np.float64).reshape(6)
        hi = np.asarray(domain_hi, dtype=np.float64).reshape(6)
        xlim = (max(xlim[0], float(lo[0])), min(xlim[1], float(hi[0])))
        ylim = (max(ylim[0], float(lo[1])), min(ylim[1], float(hi[1])))
    return xlim, ylim


def parse_time_slices_s(horizon_s: float, times_s: tuple[float, ...] | None = None) -> tuple[float, ...]:
    """Time panels for the KOZ-centered PNG.

    'BRT_KOZ_VIZ_TIMES=0,450,900,...' overrides count; else 'BRT_KOZ_VIZ_N_TIMES' (default 7).
    """
    if times_s is not None:
        return tuple(float(t) for t in times_s)
    env = os.environ.get("BRT_KOZ_VIZ_TIMES", "").strip()
    if env:
        return tuple(float(x.strip()) for x in env.split(",") if x.strip())
    n = max(2, int(os.environ.get("BRT_KOZ_VIZ_N_TIMES", "7")))
    return tuple(float(t) for t in np.linspace(0.0, float(horizon_s), n, dtype=np.float64))


def _panel_grid(n_panels: int, *, max_cols: int = 4) -> tuple[int, int]:
    ncols = min(max_cols, max(1, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _value_xy_slice(
    brt: Any,
    tau_query_s: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_m: float,
    vx: float,
    vy: float,
    vz: float,
) -> np.ndarray:
    U, V = np.meshgrid(x_coords, y_coords, indexing="ij")
    pts = np.stack(
        [
            U.ravel(),
            V.ravel(),
            np.full(U.size, float(z_m)),
            np.full(U.size, float(vx)),
            np.full(U.size, float(vy)),
            np.full(U.size, float(vz)),
        ],
        axis=-1,
    )
    if hasattr(brt, "value_batch_at_tau"):
        vals = brt.value_batch_at_tau(pts, float(tau_query_s))
    else:
        vals = np.array([float(brt.value(p)) for p in pts], dtype=np.float64)
    return np.asarray(vals, dtype=np.float64).reshape(U.shape)


def _koz_shape_grid(
    inner_koz: EllipsoidKeepOut | None,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_m: float,
) -> np.ndarray | None:
    if inner_koz is None:
        return None
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    E = inner_koz.metric()
    c = np.asarray(inner_koz.center, dtype=np.float64).reshape(3)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_m))], axis=1) - c
    return np.einsum("ni,ij,nj->n", pts, E, pts).reshape(X.shape)


def _resolve_color_mode() -> str:
    return os.environ.get("BRT_KOZ_VIZ_COLOR", "per_panel").strip().lower()


def _low_v_level(sl: np.ndarray, percentile: float = 15.0) -> float:
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, percentile))


def _imshow_slice(
    ax: Any,
    sl: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    color_mode: str,
    cmap: str,
    shared_vmin: float | None = None,
    shared_vmax: float | None = None,
) -> Any:
    """``per_panel``: autoscale each slice (matches DeepReach training PNGs)."""
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif color_mode in ("per_panel", "relative", "training"):
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    elif color_mode == "unsafe_only":
        vmin, vmax = -1.0, 1.0
    else:
        vmin = shared_vmin if shared_vmin is not None else float(np.min(finite))
        vmax = shared_vmax if shared_vmax is not None else float(np.max(finite))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    use_cmap = "coolwarm_r" if color_mode in ("per_panel", "relative", "training") else cmap
    if color_mode == "unsafe_only":
        unsafe = np.where(sl <= 0.0, 1.0, np.nan)
        im = ax.imshow(
            np.ma.masked_invalid(unsafe.T),
            origin="lower",
            extent=extent,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        return im
    return ax.imshow(
        sl.T,
        origin="lower",
        extent=extent,
        cmap=use_cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )


def _add_contours(
    ax: Any,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    sl: np.ndarray,
    koz_s: np.ndarray | None,
    *,
    draw_v0: bool = True,
    draw_low_v: bool = True,
    low_v_percentile: float = 15.0,
) -> str:
    """Draw KOZ (g=0), optional V=0, and low-V envelope; return V=0 bracket tag."""
    tag = ""
    if draw_v0 and np.any(np.isfinite(sl)):
        _, tag = choose_contour_level(sl, 0.0)
        if tag == "zero":
            ax.contour(
                x_coords,
                y_coords,
                sl.T,
                levels=[0.0],
                colors="black",
                linewidths=1.8,
                linestyles="--",
            )
    # Low-V percentile is misleading when V=0 is not bracketed (entire slice negative).
    if draw_low_v and tag != "zero" and np.any(np.isfinite(sl)):
        finite = sl[np.isfinite(sl)]
        if float(np.min(finite)) < 0.0 and float(np.max(finite)) > 0.0:
            lv = _low_v_level(sl, low_v_percentile)
            ax.contour(
                x_coords,
                y_coords,
                sl.T,
                levels=[lv],
                colors="navy",
                linewidths=1.4,
                linestyles="-",
            )
    if koz_s is not None:
        ax.contour(
            x_coords,
            y_coords,
            koz_s.T,
            levels=[1.0],
            colors="saddlebrown",
            linewidths=2.0,
            linestyles="-",
        )
        try:
            ax.contourf(
                x_coords,
                y_coords,
                koz_s.T,
                levels=[0.0, 1.0, 1e6],
                colors=["#ffcccc", "none"],
                alpha=0.35,
            )
        except Exception:
            pass
    unsafe = np.where(sl <= 0.0, 1.0, np.nan)
    if np.any(np.isfinite(unsafe)):
        ax.contourf(
            x_coords,
            y_coords,
            unsafe.T,
            levels=[0.5, 1.5],
            colors=["#c44e52"],
            alpha=0.18,
        )
    return tag


def render_brt_koz_centered_png(
    brt: Any,
    output_path: str,
    *,
    inner_koz: EllipsoidKeepOut | None = None,
    semi_axes_m: tuple[float, float, float] | None = None,
    z_m: float = 0.0,
    vx_m_s: float = 0.0,
    vy_m_s: float = 0.0,
    vz_m_s: float = 0.0,
    times_s: tuple[float, ...] | None = None,
    grid_n: int = 100,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
) -> str:
    """Static x–y panels at selected times, zoomed on the KOZ (chief at origin)."""
    import matplotlib.pyplot as plt

    color_mode = _resolve_color_mode()
    low_pct = float(os.environ.get("BRT_KOZ_VIZ_LOW_V_PERCENTILE", "15"))

    lo6 = np.asarray(getattr(brt, "domain_lo"), dtype=np.float64).reshape(6)
    hi6 = np.asarray(getattr(brt, "domain_hi"), dtype=np.float64).reshape(6)
    horizon = float(getattr(brt, "horizon_s", 1800.0))
    axes = semi_axes_m
    if axes is None and inner_koz is not None:
        axes = tuple(float(x) for x in inner_koz.semi_axes.reshape(3))
    if axes is None:
        axes = (28.0, 45.0, 18.0)

    if xlim is None or ylim is None:
        dx, dy = parse_xy_limits_from_env(axes, lo6, hi6)
        xlim = xlim or dx
        ylim = ylim or dy

    if times_s is None:
        times_s = parse_time_slices_s(horizon)
    else:
        times_s = parse_time_slices_s(horizon, times_s)

    x_coords = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_n), dtype=np.float64)
    y_coords = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_n), dtype=np.float64)
    koz_s = _koz_shape_grid(inner_koz, x_coords, y_coords, z_m)

    slices: list[np.ndarray] = []
    tags: list[str] = []
    vmin, vmax = np.inf, -np.inf
    for tq in times_s:
        sl = _value_xy_slice(brt, float(tq), x_coords, y_coords, z_m, vx_m_s, vy_m_s, vz_m_s)
        slices.append(sl)
        finite = sl[np.isfinite(sl)]
        if finite.size:
            vmin = min(vmin, float(np.min(finite)))
            vmax = max(vmax, float(np.max(finite)))
        tags.append(choose_contour_level(sl, 0.0)[1])

    if not np.isfinite(vmin):
        vmin, vmax = -1.0, 1.0
    extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))

    n_t = len(times_s)
    nrows, ncols = _panel_grid(n_t)
    fig, axes_ax = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    axes_flat = axes_ax.ravel()
    for j in range(n_t, len(axes_flat)):
        axes_flat[j].axis("off")
    for i, (tq, sl, ctag) in enumerate(zip(times_s, slices, tags)):
        ax = axes_flat[i]
        im = _imshow_slice(
            ax,
            sl,
            extent,
            color_mode=color_mode,
            cmap=cmap,
            shared_vmin=vmin,
            shared_vmax=vmax,
        )
        _add_contours(
            ax,
            x_coords,
            y_coords,
            sl,
            koz_s,
            draw_v0=(color_mode != "unsafe_only"),
            low_v_percentile=low_pct,
        )
        ax.plot(0.0, 0.0, "k+", markersize=8, markeredgewidth=1.2)
        ax.set_xlabel("Along-track (m)")
        if i == 0:
            ax.set_ylabel("Radial (m)")
        v0_note = "dashed = V = 0" if ctag == "zero" else "no zero crossing"
        ax.set_title(f"t = {tq:.0f} s remaining  ({v0_note})\nbrown = keep-out; red = unsafe")
    scale_note = "shared scale" if color_mode not in ("per_panel", "relative", "training") else "per-panel scale"
    cbar_label = "unsafe region" if color_mode == "unsafe_only" else f"V  ({scale_note})"
    fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.035, pad=0.04, label=cbar_label)
    fig.suptitle(
        f"Learned value function — x–y slices (z = {z_m:.1f} m, coasting)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def render_brt_xy_value_evolution_gif(
    brt: Any,
    output_path: str,
    *,
    inner_koz: EllipsoidKeepOut | None = None,
    semi_axes_m: tuple[float, float, float] | None = None,
    z_m: float = 0.0,
    vx_m_s: float = 0.0,
    vy_m_s: float = 0.0,
    vz_m_s: float = 0.0,
    n_frames: int | None = None,
    grid_n: int = 80,
    fps: float = 4.0,
    hold_last_frames: int = 2,
    figsize: tuple[float, float] = (7.0, 6.0),
    cmap: str = "RdBu_r",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    koz_centered: bool | None = None,
) -> str:
    """Animate x–y 'V' from tau=0 to tau=-T. Set 'koz_centered=True' (default) to zoom on KOZ."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    color_mode = _resolve_color_mode()
    low_pct = float(os.environ.get("BRT_KOZ_VIZ_LOW_V_PERCENTILE", "15"))

    lo6 = np.asarray(getattr(brt, "domain_lo"), dtype=np.float64).reshape(6)
    hi6 = np.asarray(getattr(brt, "domain_hi"), dtype=np.float64).reshape(6)
    horizon = float(getattr(brt, "horizon_s", 1800.0))

    if koz_centered is None:
        koz_centered = os.environ.get("BRT_SLICE_FULL_DOMAIN", "0").lower() not in ("1", "true", "yes")

    axes_sa = semi_axes_m
    if axes_sa is None and inner_koz is not None:
        axes_sa = tuple(float(x) for x in inner_koz.semi_axes.reshape(3))
    if axes_sa is None:
        axes_sa = (28.0, 45.0, 18.0)

    if koz_centered and (xlim is None or ylim is None):
        dx, dy = parse_xy_limits_from_env(axes_sa, lo6, hi6)
        xlim = xlim or dx
        ylim = ylim or dy
    else:
        xlim = xlim or (float(lo6[0]), float(hi6[0]))
        ylim = ylim or (float(lo6[1]), float(hi6[1]))

    if n_frames is None:
        n_frames = int(os.environ.get("BRT_SLICE_EVOLUTION_FRAMES", "21"))
    n_frames = max(2, int(n_frames))

    x_coords = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_n), dtype=np.float64)
    y_coords = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_n), dtype=np.float64)
    t_query = np.linspace(0.0, horizon, n_frames, dtype=np.float64)
    tau_labels = -t_query
    koz_s = _koz_shape_grid(inner_koz, x_coords, y_coords, z_m)

    zoom_tag = "KOZ-centered" if koz_centered else "full training domain"
    print(
        f"  Slice evolution ({zoom_tag}): τ∈[{tau_labels[0]:.0f},{tau_labels[-1]:.0f}] s, "
        f"grid {grid_n}², x∈[{xlim[0]:.0f},{xlim[1]:.0f}], y∈[{ylim[0]:.0f},{ylim[1]:.0f}]…",
        flush=True,
    )
    t0 = time_mod.perf_counter()
    slices: list[np.ndarray] = []
    vmin, vmax = np.inf, -np.inf
    for tq in t_query:
        sl = _value_xy_slice(brt, float(tq), x_coords, y_coords, z_m, vx_m_s, vy_m_s, vz_m_s)
        finite = sl[np.isfinite(sl)]
        if finite.size:
            vmin = min(vmin, float(np.min(finite)))
            vmax = max(vmax, float(np.max(finite)))
        slices.append(sl)
    if not np.isfinite(vmin):
        vmin, vmax = -1.0, 1.0
    extent = (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1]))
    use_cmap = "coolwarm_r" if color_mode in ("per_panel", "relative", "training") else cmap

    fig, ax = plt.subplots(figsize=figsize)
    sl0 = slices[0]
    fin0 = sl0[np.isfinite(sl0)]
    if color_mode in ("per_panel", "relative", "training") and fin0.size:
        im_vmin, im_vmax = float(np.min(fin0)), float(np.max(fin0))
    else:
        im_vmin, im_vmax = vmin, vmax
    im = ax.imshow(
        sl0.T,
        origin="lower",
        extent=extent,
        cmap=use_cmap,
        vmin=im_vmin,
        vmax=im_vmax if im_vmax > im_vmin else im_vmin + 1.0,
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.plot(0.0, 0.0, "k+", markersize=8)
    contour_sets: list[Any] = []

    def _draw_contours(sl: np.ndarray) -> None:
        nonlocal contour_sets
        for cs in contour_sets:
            try:
                cs.remove()
            except Exception:
                pass
        contour_sets = []
        _, tag = choose_contour_level(sl, 0.0)
        if tag == "zero" and np.any(np.isfinite(sl)):
            contour_sets.append(
                ax.contour(x_coords, y_coords, sl.T, levels=[0.0], colors="black", linewidths=1.5, linestyles="--")
            )
        if np.any(np.isfinite(sl)):
            lv = _low_v_level(sl, low_pct)
            contour_sets.append(
                ax.contour(x_coords, y_coords, sl.T, levels=[lv], colors="navy", linewidths=1.4, linestyles="-")
            )
        if koz_s is not None:
            contour_sets.append(
                ax.contour(x_coords, y_coords, koz_s.T, levels=[1.0], colors="saddlebrown", linewidths=1.2)
            )

    _draw_contours(slices[0])

    def _title(tau: float, tq: float) -> str:
        return (
            f"{zoom_tag} ({color_mode})  z={z_m:.1f} m\n"
            f"τ = {tau:.0f} s; navy = low-V ({low_pct:.0f}%ile); brown = KOZ"
        )

    ax.set_title(_title(float(tau_labels[0]), float(t_query[0])))
    hold = max(0, int(hold_last_frames))
    frame_indices = list(range(n_frames)) + [n_frames - 1] * hold

    def update(j: int) -> tuple:
        k = frame_indices[int(j)]
        sl = slices[k]
        fin = sl[np.isfinite(sl)]
        if color_mode in ("per_panel", "relative", "training") and fin.size:
            im.set_clim(float(np.min(fin)), float(np.max(fin)))
        im.set_data(sl.T)
        _draw_contours(sl)
        ax.set_title(_title(float(tau_labels[k]), float(t_query[k])))
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        anim.save(out, writer="pillow", fps=float(fps), dpi=100)
    except Exception as exc:
        warnings.warn(f"GIF save failed: {exc}", UserWarning)
        out = ""
    plt.close(fig)
    print(f"  Slice evolution done in {time_mod.perf_counter() - t0:.1f} s.", flush=True)
    return out


def main() -> None:
    from simulation.brt.deepreach_mpc_brt import (
        DEEPREACH_MPC_AVAILABLE,
        DEEPREACH_MPC_IMPORT_ERROR,
        default_checkpoint_dir,
        load_or_train_koz_brt,
    )
    from simulation.cw_dynamics import leo_circular_orbit

    if not DEEPREACH_MPC_AVAILABLE:
        print(DEEPREACH_MPC_IMPORT_ERROR or "torch required", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="KOZ-centered BRT x–y slice PNG / evolution GIF.")
    p.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir()))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--png", action="store_true", help="Write static multi-panel KOZ-centered PNG.")
    p.add_argument("--gif", action="store_true", help="Write evolution GIF.")
    p.add_argument(
        "--n-times",
        type=int,
        default=int(os.environ.get("BRT_KOZ_VIZ_N_TIMES", "7")),
        help="Number of evenly spaced time slices for PNG (default 7).",
    )
    p.add_argument(
        "--times",
        type=str,
        default=os.environ.get("BRT_KOZ_VIZ_TIMES", ""),
        help="Explicit times in seconds, comma-separated (overrides --n-times).",
    )
    p.add_argument("--semi-axes", type=str, default="28,45,18")
    p.add_argument("--device", type=str, default=os.environ.get("DEEPREACH_DEVICE", "auto"))
    args = p.parse_args()

    if not args.png and not args.gif:
        args.png = True
        args.gif = True

    os.environ.setdefault("DEEPREACH_AUTO_TRAIN", "0")
    leo = leo_circular_orbit(float(os.environ.get("LEO_ALTITUDE_KM", "400")))
    axes = tuple(float(x) for x in args.semi_axes.split(","))
    inner = EllipsoidKeepOut(np.array(axes))
    brt, _ = load_or_train_koz_brt(
        leo.n_rad_s,
        semi_axes_m=axes,
        checkpoint_dir=args.checkpoint_dir,
        force_train=False,
    )
    out_dir = args.out_dir.strip() or str(Path(args.checkpoint_dir).resolve().parent)
    os.makedirs(out_dir, exist_ok=True)

    if args.png:
        path = os.path.join(out_dir, "brt_koz_centered_xy.png")
        if args.times.strip():
            times = tuple(float(x.strip()) for x in args.times.split(",") if x.strip())
        else:
            os.environ["BRT_KOZ_VIZ_N_TIMES"] = str(args.n_times)
            times = None
        render_brt_koz_centered_png(
            brt, path, inner_koz=inner, semi_axes_m=axes, times_s=times
        )
        print(f"Wrote {path}")
    if args.gif:
        path = os.path.join(out_dir, "brt_koz_centered_xy_evolution.gif")
        render_brt_xy_value_evolution_gif(
            brt, path, inner_koz=inner, semi_axes_m=axes, koz_centered=True
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
