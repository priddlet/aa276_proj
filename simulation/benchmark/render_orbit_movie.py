#!/usr/bin/env python3
"""Render chief + deputy ECI orbit animation (GIF/MP4) and static snapshot.

Default mode: **initial formation** - free drift from the scenario start state
(no burns, no plan). Use ``--plan-id`` to visualize a maneuver plan instead.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.cw_dynamics import CWDynamics, leo_circular_orbit, maneuver_total_duration_s
from simulation.eci_kinematics import (
    build_eci_ephemeris,
    build_eci_ephemeris_from_segments,
    circular_orbit_radius_km,
)
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import LLMScenario, default_llm_dir, load_llm_bundle, load_llm_plans
from simulation.orbit_movie import render_orbit_eci_animation, sample_uniform_times


def render_orbit_static_png(
    trail_eph: dict[str, np.ndarray],
    output_path: Path,
    *,
    inner_koz: EllipsoidKeepOut | None = None,
    title: str = "Chief and deputy orbits (400 km LEO)",
    dpi: int = 150,
) -> Path:
    """Single-frame global + formation view (matches animation layout)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    from simulation.cw_dynamics import R_EARTH_KM
    from simulation.orbit_movie import _earth_wireframe_km, inner_koz_wireframe_eci_km

    r_c = trail_eph["r_chief_km"]
    r_d = trail_eph["r_deputy_km"]
    a_km = float(trail_eph["a_km"][0])
    k = len(r_c) // 2
    Ri = trail_eph["R_chief_body_to_eci"][k]

    fig = plt.figure(figsize=(13, 5.5))
    ax_g = fig.add_subplot(121, projection="3d")
    ax_f = fig.add_subplot(122, projection="3d")

    xe, ye, ze = _earth_wireframe_km(R_EARTH_KM, nu=48, nv=24)
    for ax in (ax_g, ax_f):
        ax.plot_surface(xe, ye, ze, color="#d4e4f7", alpha=0.35, linewidth=0, shade=True)

    ax_g.plot(r_c[:, 0], r_c[:, 1], r_c[:, 2], color="0.25", lw=1.2, label="Chief")
    ax_g.plot(r_d[:, 0], r_d[:, 1], r_d[:, 2], color="tab:orange", lw=1.4, label="Deputy")
    ax_g.scatter(*r_c[0], c="0.2", s=36, depthshade=True, label="Start")
    ax_g.scatter(*r_d[0], c="tab:blue", s=36, depthshade=True)
    ax_g.set_title("Global view")
    ax_g.set_xlabel("x (km)")
    ax_g.set_ylabel("y (km)")
    ax_g.set_zlabel("z (km)")
    lim = a_km * 1.35
    ax_g.set_xlim(-lim, lim)
    ax_g.set_ylim(-lim, lim)
    ax_g.set_zlim(-lim * 0.4, lim * 0.4)
    ax_g.legend(loc="upper right", fontsize=8)

    mid_c = r_c[k]
    mid_d = r_d[k]
    ax_f.plot(r_c[:, 0], r_c[:, 1], r_c[:, 2], color="0.35", lw=0.9, alpha=0.7, label="Chief")
    ax_f.plot(r_d[:, 0], r_d[:, 1], r_d[:, 2], color="tab:orange", lw=1.6, label="Deputy")
    ax_f.scatter(*r_c[0], c="0.15", s=44, label="Chief (t=0)")
    ax_f.scatter(*r_d[0], c="tab:blue", s=44, label="Deputy (t=0)")
    if inner_koz is not None:
        Xk, Yk, Zk = inner_koz_wireframe_eci_km(inner_koz, mid_c, Ri, nu=24, nv=14)
        ax_f.plot_wireframe(Xk, Yk, Zk, color="crimson", linewidth=0.5, alpha=0.65)
    center = 0.5 * (mid_c + mid_d)
    span = max(0.25, float(np.linalg.norm(mid_d - mid_c)) * 4.0)
    ax_f.set_xlim(center[0] - span, center[0] + span)
    ax_f.set_ylim(center[1] - span, center[1] + span)
    ax_f.set_zlim(center[2] - span * 0.6, center[2] + span * 0.6)
    ax_f.set_title("Formation (zoom)")
    ax_f.set_xlabel("x (km)")
    ax_f.set_ylabel("y (km)")
    ax_f.set_zlabel("z (km)")
    ax_f.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=11, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_scenario(llm_dir: str) -> LLMScenario:
    scenario, _ = load_llm_bundle(llm_dir)
    return scenario


def _resolve_start_state(scenario: LLMScenario, start_y_m: float | None) -> np.ndarray:
    if start_y_m is not None:
        y = float(start_y_m)
    else:
        y = float(scenario.reference_start_y_m)
    return np.array([0.0, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Chief + deputy ECI orbit GIF/PNG (initial formation by default)."
    )
    p.add_argument("--llm-dir", type=str, default=str(default_llm_dir(ROOT)))
    p.add_argument(
        "--plan-id",
        type=str,
        default="",
        help="If set, animate that maneuver plan instead of free-drift initial orbit.",
    )
    p.add_argument(
        "--start-y-m",
        type=float,
        default=None,
        help="Deputy along-track start (m). Default: scenario reference (90 m for boundary bundle).",
    )
    p.add_argument(
        "--orbits",
        type=float,
        default=float(os.environ.get("ORBIT_CHIEF_ORBITS", "1.5")),
        help="Chief orbit revolutions to animate (initial-orbit mode only).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "simulation_output" / "report" / "figures"),
    )
    p.add_argument("--gif-name", type=str, default="spacecraft_orbits_eci.gif")
    p.add_argument("--png-name", type=str, default="spacecraft_orbits_eci.png")
    p.add_argument("--duration-s", type=float, default=None, help="Override animation duration (s).")
    p.add_argument("--frames", type=int, default=int(os.environ.get("ORBIT_ANIM_FRAMES", "180")))
    p.add_argument("--trail-samples", type=int, default=int(os.environ.get("ORBIT_TRAIL_SAMPLES", "2000")))
    p.add_argument("--fps", type=float, default=float(os.environ.get("ORBIT_ANIM_FPS", "20")))
    p.add_argument("--dpi", type=int, default=int(os.environ.get("ORBIT_ANIM_DPI", "100")))
    p.add_argument(
        "--eci-view",
        type=str,
        default=os.environ.get("ORBIT_ECI_VIEW", "both"),
        choices=("both", "global", "formation"),
    )
    p.add_argument("--altitude-km", type=float, default=400.0)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()

    scenario = _load_scenario(args.llm_dir)
    leo = leo_circular_orbit(args.altitude_km)
    plant = CWDynamics(scenario.mean_motion_rad_s if scenario.mean_motion_rad_s > 0 else leo.n_rad_s)
    inner = EllipsoidKeepOut(np.array(scenario.semi_axes_m, dtype=np.float64))
    period_s = 2.0 * math.pi / plant.n
    a_km = circular_orbit_radius_km(args.altitude_km)

    if args.plan_id.strip():
        _, plans = load_llm_plans(args.llm_dir)
        plan = next((pl for pl in plans if pl.plan_id == args.plan_id.strip()), None)
        if plan is None:
            print(f"plan_id not found: {args.plan_id}", file=sys.stderr)
            sys.exit(1)
        x0 = plan.x0(scenario)
        segments = plan.segments
        plan_dur = maneuver_total_duration_s(segments)
        duration_s = float(args.duration_s if args.duration_s is not None else max(plan_dur, 1200.0))
        title = f"Maneuver: {plan.plan_id.replace('_', ' ')}"
        print(f"Plan {plan.plan_id}: y0={x0[1]:.0f} m, duration={duration_s:.0f} s ({duration_s/period_s:.2f} orbits)")

        def _ephem(times: np.ndarray) -> dict[str, np.ndarray]:
            return build_eci_ephemeris_from_segments(
                plant, x0, segments, times, altitude_km=args.altitude_km, a_km=a_km
            )
    else:
        x0 = _resolve_start_state(scenario, args.start_y_m)
        duration_s = float(
            args.duration_s if args.duration_s is not None else float(args.orbits) * period_s
        )
        title = f"Initial formation - deputy {x0[1]:.0f} m downrange (free drift, no burns)"
        print(
            f"Initial orbit: y0={x0[1]:.0f} m, duration={duration_s:.0f} s "
            f"({duration_s/period_s:.2f} chief orbits), no maneuvers"
        )

        def _ephem(times: np.ndarray) -> dict[str, np.ndarray]:
            return build_eci_ephemeris(
                plant, x0, times, altitude_km=args.altitude_km, a_km=a_km
            )

    times = sample_uniform_times(duration_s, int(args.frames))
    times_trail = sample_uniform_times(duration_s, int(args.trail_samples))
    ephem = _ephem(times)
    trail_eph = _ephem(times_trail)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_png:
        png_path = render_orbit_static_png(
            trail_eph,
            out_dir / args.png_name,
            inner_koz=inner,
            title=title,
        )
        print(f"Wrote {png_path}")

    if not args.no_gif:
        gif_path = out_dir / args.gif_name
        written = render_orbit_eci_animation(
            ephem,
            trail_r_chief_km=trail_eph["r_chief_km"],
            trail_r_deputy_km=trail_eph["r_deputy_km"],
            output_path=str(gif_path),
            fps=float(args.fps),
            dpi=int(args.dpi),
            eci_view=args.eci_view,
            inner_koz_formation=inner,
            show=False,
        )
        if written:
            print(f"Wrote {written}")


if __name__ == "__main__":
    main()
