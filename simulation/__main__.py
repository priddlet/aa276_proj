"""Demo: CW trajectory, DeepReach-MPC 6D BRT, sampling safety filter, ECI animation."""

from __future__ import annotations

import csv
import math
import os
import sys
import time as time_mod
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from simulation.brt_safety import simulate_plan_with_brt
from simulation.cw_dynamics import (
    CWDynamics,
    leo_circular_orbit,
    maneuver_total_duration_s,
    simulate_impulsive_segments_dense,
    state_at_maneuver_elapsed_time,
)
from simulation.eci_kinematics import build_eci_ephemeris
from simulation.keepout import EllipsoidKeepOut, EllipsoidMaxSeparation
from simulation.maneuver_plan import burns_to_segments, parse_llm_maneuver_json
from simulation.orbit_movie import render_orbit_eci_animation, sample_uniform_times
from simulation.passive_safety import is_passively_safe_natural_coast


def _default_output_dir(project_root: str | Path | None = None) -> Path:
    env = os.environ.get("SIMULATION_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    return (root / "simulation_output").resolve()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    p = Path(path)
    _ensure_dir(p.parent)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return p


def _segment_index_for_elapsed(
    segments: list[tuple[float, np.ndarray | None]], t_elapsed_s: float
) -> int:
    """Index ``k`` of the segment active at time ``t_elapsed_s`` (0-based); ``-1`` if ``t <= 0``."""
    if t_elapsed_s <= 0.0:
        return -1
    c = 0.0
    for k, (dt, _) in enumerate(segments):
        c += float(dt)
        if t_elapsed_s <= c + 1e-9:
            return k
    return len(segments) - 1


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = _ensure_dir(_default_output_dir(root))

    altitude_km = float(os.environ.get("LEO_ALTITUDE_KM", "400"))
    leo = leo_circular_orbit(altitude_km)
    plant = CWDynamics(leo.n_rad_s)

    inner_axes = np.array(
        [float(x) for x in os.environ.get("KOZ_INNER_SEMIAXES_M", "28,45,18").split(",")],
        dtype=np.float64,
    )
    outer_axes = np.array(
        [float(x) for x in os.environ.get("KOZ_OUTER_SEMIAXES_M", "4800,14000,4000").split(",")],
        dtype=np.float64,
    )
    inner_koz = EllipsoidKeepOut(inner_axes)
    outer_corridor = EllipsoidMaxSeparation(outer_axes)

    y0 = float(os.environ.get("APPROACH_START_ALONGTRACK_M", "3200.0"))
    x0 = np.array([0.0, y0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    dt = float(os.environ.get("DEMO_SEGMENT_DT_S", "45.0"))
    segments = [
        (dt, np.array([-0.015, -0.12, 0.0])),
        (dt, np.array([-0.01, -0.08, 0.0])),
        (dt, np.array([0.0, -0.05, 0.0])),
    ]

    _, states = simulate_impulsive_segments_dense(
        plant, x0, segments, substeps=int(os.environ.get("LVLH_DENSE_SUBSTEPS", "120"))
    )

    print(
        f"LEO (circular): n = {leo.n_rad_s:.6e} rad/s, a = {leo.a_km:.3f} km, "
        f"h = {leo.altitude_km:.1f} km altitude"
    )
    print(f"Output directory: {out_dir}")

    demo_json = os.environ.get(
        "LLM_MANEUVER_JSON",
        '{"maneuvers": ['
        '{"t_s": 45, "dv_m_s": [-0.015, -0.12, 0.0], "kind": "approach"},'
        '{"timestep": 45, "delta_v": [-0.01, -0.08, 0.0], "kind": "approach"},'
        '{"t_s": 60, "dv_m_s": [0, 0, 0], "kind": "coast"}'
        "]}",
    )
    parsed = parse_llm_maneuver_json(demo_json)
    segs, kinds = burns_to_segments(parsed)
    passive_h = float(os.environ.get("PASSIVE_CHECK_HORIZON_S", "7200"))
    log_interval_s = float(os.environ.get("BRT_LOG_INTERVAL_S", "50"))

    from simulation.brt.deepreach_mpc_brt import (
        DEEPREACH_MPC_AVAILABLE,
        DEEPREACH_MPC_IMPORT_ERROR,
        default_checkpoint_dir,
        load_or_train_koz_brt,
    )
    from simulation.brt.config import BRT_HORIZON_S
    from simulation.sampling.safety_filter import filter_maneuver_plan

    if not DEEPREACH_MPC_AVAILABLE:
        print(
            "Option 1 BRT requires DeepReach-MPC (torch + deepreach_MPC/). "
            "Install: pip install -r requirements-deepreach.txt",
            file=sys.stderr,
        )
        if DEEPREACH_MPC_IMPORT_ERROR:
            print(f"  ({DEEPREACH_MPC_IMPORT_ERROR})", file=sys.stderr)
        sys.exit(1)
    try:
        import skimage  # noqa: F401
    except ImportError:
        print(
            "Note: scikit-image not installed — 3D BRT shells use slice contours only. "
            "Run: pip install scikit-image",
            flush=True,
        )

    brt_horizon_s = float(os.environ.get("BRT_HORIZON_S", str(BRT_HORIZON_S)))
    ck_dir = os.environ.get("DEEPREACH_CHECKPOINT_DIR", "").strip() or str(default_checkpoint_dir(root))
    force_train = os.environ.get("DEEPREACH_FORCE_TRAIN", "0").lower() in ("1", "true", "yes")
    print(
        "Option 1 BRT (DeepReach-MPC, 6D CW avoid) for the inner KOZ "
        f"(chief-centered LVLH; semi-axes m = {inner_axes.tolist()}).\n"
        f"  Horizon: {brt_horizon_s:.0f} s. Checkpoints: {ck_dir}"
    )
    t0 = time_mod.perf_counter()
    brt, loaded_brt = load_or_train_koz_brt(
        leo.n_rad_s,
        semi_axes_m=inner_axes,
        center_m=inner_koz.center,
        checkpoint_dir=ck_dir,
        force_train=force_train,
    )
    elapsed = time_mod.perf_counter() - t0
    if loaded_brt:
        print(f"  DeepReach-MPC BRT loaded ({elapsed:.2f} s). Set DEEPREACH_FORCE_TRAIN=1 to retrain.")
    else:
        print(f"  DeepReach-MPC training finished ({elapsed / 60:.1f} min). Checkpoints: {ck_dir}")

    if os.environ.get("SAFETY_FILTER", "1").lower() not in ("0", "false", "no"):
        max_pert = float(os.environ.get("FILTER_MAX_PERTURB_M_S", "0.08"))
        n_sph = int(os.environ.get("FILTER_N_SPHERE", "48"))
        margin = float(os.environ.get("FILTER_BRT_MARGIN", "0.0"))
        from simulation.sampling.safety_filter import default_filter_mode

        fmode = default_filter_mode()
        if fmode == "linesearch":
            print("BRT safety filter (line-search α·Δv toward 0, V(x⁺, t_k), passive from x_k)…")
        else:
            print(
                f"BRT safety filter — sampling mode (max Δv perturb {max_pert} m/s, {n_sph} samples)…"
            )
        segs_filt, filt_results = filter_maneuver_plan(
            plant,
            x0,
            segs,
            brt,
            filter_mode=fmode,
            max_perturb_m_s=max_pert,
            n_sphere_samples=n_sph,
            brt_margin=margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_h,
        )
        n_ok = sum(1 for fr in filt_results if fr.accepted)
        print(f"  Filter: {len(filt_results)} burns, {n_ok} accepted safe perturbations.")
        for i, fr in enumerate(filt_results):
            print(
                f"    burn {i}: accepted={fr.accepted} α={fr.scale_alpha:.3f} "
                f"|Δv|_res={fr.residual_norm:.4f} m/s V(x⁺,t={fr.time_s:.1f})={fr.brt_value:.3f}"
            )
        segs = segs_filt

    print("Evaluating BRT (learned V; unsafe if V ≤ 0) at maneuver segment boundaries…")
    _, _, boundary_steps = simulate_plan_with_brt(
        plant,
        x0,
        segs,
        brt,
        burn_kinds=kinds,
        passive_inner_koz=inner_koz,
        passive_horizon_s=passive_h,
    )
    n_unsafe_b = sum(1 for s in boundary_steps if s.unsafe)
    print(f"  {len(boundary_steps)} boundary nodes, {n_unsafe_b} unsafe under Option 1 BRT.")
    print("  (Burn kinds, passive flags, and sampled trajectory → CSV.)")

    T_plan = maneuver_total_duration_s(segs)
    sample_times = np.arange(0.0, T_plan + 1e-9, log_interval_s, dtype=np.float64)
    if sample_times.size == 0 or abs(float(sample_times[-1]) - T_plan) > 1e-3 * max(1.0, T_plan):
        sample_times = np.unique(np.append(sample_times, T_plan))
    boundary_times = np.array([float(s.time_s) for s in boundary_steps], dtype=np.float64)
    log_times = np.unique(np.concatenate([sample_times, boundary_times]))

    csv_rows: list[dict[str, Any]] = []
    passive_samples = int(os.environ.get("PASSIVE_LOG_SAMPLES", "256"))
    tol = 1e-6 * max(1.0, T_plan)
    for tq in log_times:
        tqf = float(tq)
        xq = state_at_maneuver_elapsed_time(plant, x0, segs, tqf)
        seg_i = _segment_index_for_elapsed(segs, tqf)
        kind = "" if seg_i < 0 else (kinds[seg_i] or "")
        ps = is_passively_safe_natural_coast(
            plant, xq, inner_koz, passive_h, n_samples=passive_samples
        )
        v_brt = float(brt.value(xq))
        u_brt = bool(brt.is_unsafe(xq))
        r = xq[:3]
        on_bdry = bool(np.any(np.abs(boundary_times - tqf) <= tol))
        csv_rows.append(
            {
                "time_s": tqf,
                "segment_index": seg_i,
                "burn_kind": kind,
                "x_m": float(xq[0]),
                "y_m": float(xq[1]),
                "z_m": float(xq[2]),
                "vx_m_s": float(xq[3]),
                "vy_m_s": float(xq[4]),
                "vz_m_s": float(xq[5]),
                "brt_value": v_brt,
                "brt_unsafe": int(u_brt),
                "passive_safe": int(ps),
                "inner_koz_inside": int(inner_koz.is_inside(r)),
                "outer_corridor_unsafe_far": int(outer_corridor.is_unsafe_far(r)),
                "row_kind": "segment_boundary" if on_bdry else f"sample_{log_interval_s:g}s",
            }
        )

    fieldnames = [
        "time_s",
        "segment_index",
        "burn_kind",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "brt_value",
        "brt_unsafe",
        "passive_safe",
        "inner_koz_inside",
        "outer_corridor_unsafe_far",
        "row_kind",
    ]
    _write_csv(os.path.join(str(out_dir), "brt_trajectory_log.csv"), csv_rows, fieldnames=fieldnames)
    print(
        f"Wrote brt_trajectory_log.csv ({len(csv_rows)} rows: every ~{log_interval_s:g} s "
        f"and maneuver segment boundaries, de-duplicated by time_s)."
    )

    if os.environ.get("BRT_KOZ_CENTERED_VIZ", "1").lower() not in ("0", "false", "no"):
        from simulation.brt.slice_viz import parse_time_slices_s, render_brt_koz_centered_png

        koz_png = os.path.join(str(out_dir), "brt_koz_centered_xy.png")
        times = parse_time_slices_s(brt.horizon_s)
        print(
            f"Rendering KOZ-centered BRT x–y slices ({len(times)} times: "
            f"{', '.join(f'{t:.0f}' for t in times)} s)…"
        )
        render_brt_koz_centered_png(
            brt,
            koz_png,
            inner_koz=inner_koz,
            semi_axes_m=tuple(float(x) for x in inner_axes),
            times_s=times,
            z_m=float(os.environ.get("BRT_SLICE_Z_M", "0")),
            vx_m_s=float(os.environ.get("BRT_SLICE_VX_M_S", "0")),
            vy_m_s=float(os.environ.get("BRT_SLICE_VY_M_S", "0")),
            vz_m_s=float(os.environ.get("BRT_SLICE_VZ_M_S", "0")),
            grid_n=int(os.environ.get("BRT_KOZ_VIZ_GRID_N", "100")),
        )
        print(f"  Wrote {koz_png}")

    if os.environ.get("BRT_VALUE_EVOLUTION_GIF", "1").lower() not in ("0", "false", "no"):
        from simulation.brt.slice_viz import render_brt_xy_value_evolution_gif

        evo_path = os.path.join(str(out_dir), "brt_koz_centered_xy_evolution.gif")
        z_sl = float(os.environ.get("BRT_SLICE_Z_M", "0"))
        vxs = float(os.environ.get("BRT_SLICE_VX_M_S", "0"))
        vys = float(os.environ.get("BRT_SLICE_VY_M_S", "0"))
        vzs = float(os.environ.get("BRT_SLICE_VZ_M_S", "0"))
        n_evo = int(os.environ.get("BRT_SLICE_EVOLUTION_FRAMES", "21"))
        grid_n = int(os.environ.get("BRT_SLICE_GRID_N", "80"))
        evo_fps = float(os.environ.get("BRT_SLICE_EVOLUTION_FPS", "4"))
        print("Rendering x–y BRT value slice evolution (τ = 0 → -T)…")
        written_evo = render_brt_xy_value_evolution_gif(
            brt,
            evo_path,
            inner_koz=inner_koz,
            semi_axes_m=tuple(float(x) for x in inner_axes),
            z_m=z_sl,
            vx_m_s=vxs,
            vy_m_s=vys,
            vz_m_s=vzs,
            n_frames=n_evo,
            grid_n=grid_n,
            fps=evo_fps,
            koz_centered=True,
        )
        if written_evo:
            print(f"  Wrote {written_evo}")

    if os.environ.get("BRT_SNAPSHOT", "1").lower() not in ("0", "false", "no"):
        from simulation.snapshot_viz import render_brt_lvlh_snapshot

        snap_png = os.path.join(str(out_dir), "brt_formation_lvlh.png")
        snap_gif: str | None = None
        if os.environ.get("BRT_SNAPSHOT_GIF", "1").lower() not in ("0", "false", "no"):
            snap_gif = os.path.join(str(out_dir), "brt_formation_lvlh.gif")
        # Half-width of LVLH position cube centered on chief (m). Smaller = tighter zoom on target.
        chief_half = float(os.environ.get("BRT_SNAPSHOT_CHIEF_HALF_M", "1200"))
        max_search = float(os.environ.get("BRT_SNAPSHOT_MAX_HALF_M", "6000"))
        iso_snap = os.environ.get("BRT_SNAPSHOT_ISO", "32,32,26").strip()
        parts = [int(x) for x in iso_snap.split(",")]
        if len(parts) != 3:
            raise ValueError("BRT_SNAPSHOT_ISO must be three comma-separated integers")
        rx, ry, rz = parts
        gif_n = int(os.environ.get("BRT_SNAPSHOT_GIF_FRAMES", "16"))
        gif_f = float(os.environ.get("BRT_SNAPSHOT_GIF_FPS", "8"))
        dpi_snap = int(os.environ.get("BRT_SNAPSHOT_DPI", "120"))
        print("Rendering BRT + KOZ snapshot (LVLH, zoomed on chief / target)…")
        p_out, g_out = render_brt_lvlh_snapshot(
            brt,
            inner_koz,
            x0,
            output_path_png=snap_png,
            output_path_gif=snap_gif,
            chief_box_half_m=chief_half,
            max_search_half_m=max_search,
            iso_resolution=(rx, ry, rz),
            dpi=dpi_snap,
            gif_frames=gif_n,
            gif_fps=gif_f,
        )
        if p_out:
            print(f"  Wrote {p_out}")
        if g_out:
            print(f"  Wrote {g_out}")

    print("Sample | pos (m)                    | inner KOZ | outer corridor")
    for k in (0, len(states) // 2, len(states) - 1):
        r = states[k, :3]
        inner_hit = inner_koz.is_inside(r)
        outer_hit = outer_corridor.is_unsafe_far(r)
        print(
            f" {k:5d} | {r[0]:8.2f} {r[1]:8.2f} {r[2]:8.2f} | "
            f"{'INSIDE' if inner_hit else 'clear':^9} | {'OUTSIDE' if outer_hit else 'inside':^14}"
        )

    x_coast0 = x0.copy()
    period_s = 2.0 * math.pi / leo.n_rad_s
    duration_s = 3.0 * period_s
    n_frames = int(os.environ.get("ORBIT_ANIM_FRAMES", "540"))
    times = sample_uniform_times(duration_s, n_frames)
    ephem = build_eci_ephemeris(
        plant, x_coast0, times, altitude_km=altitude_km, theta0_rad=0.0, a_km=leo.a_km
    )

    n_trail = int(os.environ.get("ORBIT_TRAIL_SAMPLES", "4000"))
    times_trail = sample_uniform_times(duration_s, n_trail)
    trail_eph = build_eci_ephemeris(
        plant, x_coast0, times_trail, altitude_km=altitude_km, theta0_rad=0.0, a_km=leo.a_km
    )

    out_fmt = os.environ.get("ORBIT_MOVIE_FORMAT", "gif").lower().lstrip(".")
    movie_path = os.path.join(str(out_dir), f"orbit_eci_3rev.{out_fmt}")
    fps = float(os.environ.get("ORBIT_ANIM_FPS", "20"))

    iso = os.environ.get("FORMATION_BRT_ISO", "18,18,14")
    nx, ny, nz = (int(x) for x in iso.split(","))
    mrg = os.environ.get("FORMATION_BRT_MARGIN_M", "900,900,500")
    mx, my, mz = (float(x) for x in mrg.split(","))

    frame_log: list[dict[str, object]] = []
    print(
        f"Rendering {duration_s/60:.1f} min ({duration_s/period_s:.1f} chief orbits), "
        f"{n_frames} anim frames, {n_trail} trail samples @ {fps} fps -> {movie_path}"
    )
    written = render_orbit_eci_animation(
        ephem,
        trail_r_chief_km=trail_eph["r_chief_km"],
        trail_r_deputy_km=trail_eph["r_deputy_km"],
        output_path=movie_path,
        fps=fps,
        dpi=int(os.environ.get("ORBIT_ANIM_DPI", "100")),
        show=False,
        brt_option1=brt,
        inner_koz_formation=inner_koz,
        frame_log_rows=frame_log,
        formation_brt_iso_resolution=(nx, ny, nz),
        formation_brt_margin_m=(mx, my, mz),
    )
    if frame_log:
        _write_csv(
            os.path.join(str(out_dir), "orbit_frame_brt_log.csv"),
            frame_log,
            fieldnames=[
                "frame",
                "time_s",
                "x_m",
                "y_m",
                "z_m",
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
                "sep_km",
                "earth_koz_violation_chief",
                "earth_koz_violation_deputy",
                "inner_koz_shape",
                "inner_koz_inside",
                "brt_value",
                "brt_unsafe",
            ],
        )
    if written:
        print(f"Wrote {written}")
    else:
        print("Animation export did not complete (check warnings / ffmpeg for mp4).")


if __name__ == "__main__":
    main()
