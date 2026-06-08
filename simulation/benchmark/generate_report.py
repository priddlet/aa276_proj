"""Build report figures and tables from benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from simulation.benchmark.evaluate import EvalCondition, PlanEvalResult, summarize_results
from simulation.brt.config import BRT_HORIZON_S
from simulation.brt.deepreach_mpc_brt import default_checkpoint_dir
from simulation.cw_dynamics import CWDynamics, simulate_impulsive_segments_dense
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import LLMPlan, LLMScenario, default_llm_dir, load_llm_plans
from simulation.sampling.safety_filter import FilterResult, filter_maneuver_plan


def _load_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _results_to_plan_eval(rows: list[dict[str, str]]) -> list[PlanEvalResult]:
    """Rehydrate enough for summarize_results (numeric fields only)."""
    out: list[PlanEvalResult] = []
    for r in rows:
        def _b(k: str) -> bool:
            return r.get(k, "") in ("1", "True", "true")

        def _f(k: str) -> float:
            v = r.get(k, "")
            return float(v) if v not in ("", None) else 0.0

        def _i(k: str) -> int:
            v = r.get(k, "")
            return int(v) if v not in ("", None) else 0

        out.append(
            PlanEvalResult(
                plan_id=str(r["plan_id"]),
                condition=str(r["condition"]),
                category=str(r.get("category", "")),
                approach_angle=str(r.get("approach_angle", "")),
                urgency=str(r.get("urgency", "")),
                n_burns=_i("n_burns"),
                max_dv_nom_m_s=_f("max_dv_nom_m_s"),
                intercepted=_b("intercepted"),
                mission_success=_b("mission_success"),
                mission_success_tier_b=_b("mission_success_tier_b"),
                success_kind=str(r.get("success_kind", "")),
                range_closed_m=_f("range_closed_m"),
                eval_time_s=_f("eval_time_s"),
                range_at_eval_m=_f("range_at_eval_m"),
                mean_dv_overhead_m_s=_f("mean_dv_overhead_m_s"),
                n_burns_corrected=_i("n_burns_corrected"),
                final_range_m=_f("final_range_m"),
                min_koz_shape_value=_f("min_koz_shape_value"),
                brt_unsafe_any_post_burn=_b("brt_unsafe_any_post_burn"),
                filter_n_accepted=_i("filter_n_accepted"),
                filter_n_burns=_i("filter_n_burns"),
                llm_unsafe_nominal=_b("llm_unsafe_nominal"),
                nominal_intervention_reasons=str(r.get("nominal_intervention_reasons", "")),
                brt_intervened=_b("brt_intervened"),
                n_burns_intervened=_i("n_burns_intervened"),
                n_burns_suppressed=_i("n_burns_suppressed"),
                n_burns_scaled=_i("n_burns_scaled"),
                expected_intervention=_i("expected_intervention") if r.get("expected_intervention", "") != "" else None,
                requires_intervention_rollout=_b("requires_intervention_rollout"),
                label_match=_b("label_match"),
                intervention_reasons=str(r.get("intervention_reasons", "")),
                mission_success_safe=_b("mission_success_safe"),
            )
        )
    return out


def _plot_koz_along_track(
    ax,
    inner: EllipsoidKeepOut,
    *,
    facecolor: str = "#ffcccc",
    alpha: float = 0.35,
) -> None:
    """z=0 slice in along-track frame: horizontal = y (semi-axis b), vertical = x (semi-axis a)."""
    th = np.linspace(0, 2 * np.pi, 200)
    a, b = float(inner.semi_axes[0]), float(inner.semi_axes[1])
    y0, x0 = float(inner.center[1]), float(inner.center[0])
    ax.fill(
        y0 + b * np.cos(th),
        x0 + a * np.sin(th),
        facecolor=facecolor,
        edgecolor="#8b0000",
        linewidth=1.2,
        alpha=alpha,
        zorder=1,
        label="Keep-out zone",
    )


def _lvlh_to_along_track_xy(pos: np.ndarray, vel: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """Map LVLH (x radial, y along-track) to plot (horizontal=y, vertical=x)."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    h, v = float(p[1]), float(p[0])
    if vel is None:
        return h, v, 0.0, 0.0
    u = np.asarray(vel, dtype=np.float64).reshape(3)
    return h, v, float(u[1]), float(u[0])


def _draw_craft_along_track(
    ax,
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    scale_m: float = 80.0,
    color: str = "C0",
    label: str | None = None,
    zorder: int = 5,
    fixed_along_track: bool = False,
) -> None:
    """Spacecraft glyph with body long axis along along-track / velocity in (y, x) plot."""
    h, v, vh, vv = _lvlh_to_along_track_xy(pos, vel)
    s = float(scale_m)
    if fixed_along_track:
        ux, uy = 1.0, 0.0
    else:
        spd = float(np.hypot(vh, vv))
        if spd > 1e-6:
            ux, uy = vh / spd, vv / spd
        else:
            ux, uy = -1.0, 0.0 if float(pos[1]) > 0 else 1.0
    # Body: long along u, short perpendicular
    long_h, short_v = 0.55 * s, 0.18 * s
    corners = np.array(
        [
            [-long_h, -short_v],
            [long_h, -short_v],
            [long_h, short_v],
            [-long_h, short_v],
            [-long_h, -short_v],
        ],
        dtype=np.float64,
    )
    rot = np.array([[ux, -uy], [uy, ux]], dtype=np.float64)
    body = corners @ rot.T + np.array([h, v])
    ax.plot(body[:, 0], body[:, 1], color=color, linewidth=1.4, zorder=zorder)
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.25, zorder=zorder - 1)
    if not fixed_along_track and float(np.hypot(vh, vv)) > 0.05:
        ax.annotate(
            "",
            xy=(h + 1.6 * s * ux, v + 1.6 * s * uy),
            xytext=(h, v),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
            zorder=zorder + 1,
        )
    if label:
        ax.text(h, v + 1.1 * s, label, ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")


def _plot_koz_xy(
    ax,
    inner: EllipsoidKeepOut,
    *,
    facecolor: str = "#ffcccc",
    alpha: float = 0.35,
    annotate_axes: bool = False,
) -> None:
    """LVLH x–y cross-section (z=0): semi-axis a along radial x, b along along-track y."""
    th = np.linspace(0, 2 * np.pi, 200)
    a, b = float(inner.semi_axes[0]), float(inner.semi_axes[1])
    cx, cy = float(inner.center[0]), float(inner.center[1])
    ax.fill(
        cx + a * np.cos(th),
        cy + b * np.sin(th),
        facecolor=facecolor,
        edgecolor="#8b0000",
        linewidth=1.2,
        alpha=alpha,
        zorder=1,
        label="Keep-out zone",
    )
    if annotate_axes:
        ax.plot([cx, cx + a], [cy, cy], color="#8b0000", linewidth=1.0, zorder=2)
        ax.plot([cx, cx], [cy, cy + b], color="#8b0000", linewidth=1.0, zorder=2)
        ax.text(cx + 0.55 * a, cy - 0.08 * b, f"{a:.0f} m\n(radial)", ha="center", fontsize=7, color="#8b0000")
        ax.text(cx - 0.12 * a, cy + 0.55 * b, f"{b:.0f} m\n(along-track)", ha="center", fontsize=7, color="#8b0000")


def _draw_craft_2d(
    ax,
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    scale_m: float = 80.0,
    color: str = "C0",
    label: str | None = None,
    zorder: int = 5,
) -> None:
    """Simple 2D spacecraft glyph: body rectangle + velocity arrow."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    v = np.asarray(vel, dtype=np.float64).reshape(3)
    x, y = float(p[0]), float(p[1])
    s = float(scale_m)
    body = np.array([[-0.35, -0.2], [0.35, -0.2], [0.35, 0.2], [-0.35, 0.2], [-0.35, -0.2]]) * s + np.array([x, y])
    ax.plot(body[:, 0], body[:, 1], color=color, linewidth=1.4, zorder=zorder)
    ax.fill(body[:, 0], body[:, 1], color=color, alpha=0.25, zorder=zorder - 1)
    spd = float(np.hypot(v[0], v[1]))
    if spd > 1e-6:
        u, w = v[0] / spd, v[1] / spd
        ax.annotate(
            "",
            xy=(x + 1.8 * s * u, y + 1.8 * s * w),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
            zorder=zorder + 1,
        )
    if label:
        ax.text(x, y + 1.2 * s, label, ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")


def _rollout_xy(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    *,
    substeps: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray | None]]]:
    """Return times, xy path (N,2), burn events (t, pos, dv_or_none)."""
    times, states = simulate_impulsive_segments_dense(plant, x0, segments, substeps=substeps)
    xy = np.asarray(states[:, :2], dtype=np.float64)
    burns: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    x = np.asarray(x0, dtype=np.float64).reshape(6).copy()
    t = 0.0
    for dt, dv in segments:
        if dv is not None:
            burns.append((t, x[:3].copy(), np.asarray(dv, dtype=np.float64).reshape(3).copy()))
            x = plant.apply_impulsive_dv(x, dv)
        x = plant.propagate(x, float(dt))
        t += float(dt)
    return times, xy, burns


def _deputy_eci_path_km(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    *,
    altitude_km: float = 400.0,
    substeps: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Deputy ECI positions (N,3) in km and time samples (s)."""
    from simulation.eci_kinematics import circular_orbit_radius_km, deputy_eci_from_cw

    times, states = simulate_impulsive_segments_dense(plant, x0, segments, substeps=substeps)
    a_km = circular_orbit_radius_km(altitude_km)
    n = float(plant.n)
    path = np.zeros((len(states), 3), dtype=np.float64)
    for k, (t, x6) in enumerate(zip(times, states)):
        r_d, _ = deputy_eci_from_cw(float(t), x6, a_km, n)
        path[k] = r_d
    return path, times


def render_scenario_orbit_png(
    plant: CWDynamics,
    scenario: LLMScenario,
    inner: EllipsoidKeepOut,
    plan: LLMPlan,
    output_path: Path,
    *,
    altitude_km: float = 400.0,
) -> Path:
    """LVLH along-track panel + ECI orbit panel: LLM-planned deputy path."""
    import matplotlib.pyplot as plt
    from simulation.cw_dynamics import R_EARTH_KM
    from simulation.eci_kinematics import circular_orbit_radius_km

    x0 = plan.x0(scenario)
    _, states_llm = simulate_impulsive_segments_dense(plant, x0, plan.segments, substeps=8)
    xy_llm = np.asarray(states_llm[:, :2], dtype=np.float64)
    _, _, burns = _rollout_xy(plant, x0, plan.segments)

    def _at(hv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return hv[:, 1], hv[:, 0]

    eci_llm, _ = _deputy_eci_path_km(plant, x0, plan.segments, altitude_km=altitude_km)
    a_km = circular_orbit_radius_km(altitude_km)

    fig, (ax_lv, ax_eci) = plt.subplots(1, 2, figsize=(13, 5.2))

    _plot_koz_along_track(ax_lv, inner)
    hl, vl = _at(xy_llm)
    ax_lv.plot(hl, vl, color="tab:orange", linewidth=2.0, label="Planned path (LLM)", zorder=3)
    for t_b, pos, dv in burns:
        if dv is None or np.linalg.norm(dv) < 1e-9:
            continue
        h, v = float(pos[1]), float(pos[0])
        ax_lv.scatter(h, v, s=55, c="tab:red", edgecolors="k", linewidths=0.4, zorder=6)
        ax_lv.annotate(f"burn @ {t_b:.0f} s", (h, v), textcoords="offset points", xytext=(6, 6), fontsize=7)

    _draw_craft_along_track(ax_lv, np.zeros(3), np.zeros(3), scale_m=50, color="0.15", label="Chief", fixed_along_track=True)
    _draw_craft_along_track(ax_lv, x0[:3], x0[3:6], scale_m=42, color="tab:blue", label="Deputy (start)")
    _draw_craft_along_track(ax_lv, states_llm[-1, :3], states_llm[-1, 3:6], scale_m=42, color="tab:orange", label="Deputy (end)")

    ax_lv.set_xlabel("Along-track separation (m)")
    ax_lv.set_ylabel("Radial separation (m)")
    ax_lv.set_title("Relative motion (LVLH frame)")
    ax_lv.set_aspect("equal", adjustable="box")
    ax_lv.grid(True, alpha=0.25)
    ax_lv.legend(loc="best", fontsize=7)

    th = np.linspace(0, 2 * np.pi, 360)
    ax_eci.plot(a_km * np.cos(th), a_km * np.sin(th), color="0.35", linestyle=":", linewidth=1.0, label="Chief orbit")
    earth = plt.Circle((0, 0), R_EARTH_KM, facecolor="#d4e4f7", edgecolor="0.4", linewidth=0.8, zorder=0)
    ax_eci.add_patch(earth)
    ax_eci.plot(eci_llm[:, 0], eci_llm[:, 1], color="tab:orange", linewidth=1.8, label="Deputy path", zorder=3)
    ax_eci.scatter(eci_llm[0, 0], eci_llm[0, 1], s=40, c="tab:blue", zorder=5, label=f"Start ({float(np.linalg.norm(x0[:3])):.0f} m behind)")
    ax_eci.scatter(eci_llm[-1, 0], eci_llm[-1, 1], s=40, c="tab:orange", zorder=5)

    zoom = a_km * 0.012
    cx, cy = float(eci_llm[0, 0]), float(eci_llm[0, 1])
    ax_eci.set_xlim(cx - zoom, cx + zoom)
    ax_eci.set_ylim(cy - zoom, cy + zoom)
    ax_eci.set_xlabel("ECI x (km)")
    ax_eci.set_ylabel("ECI y (km)")
    ax_eci.set_title(f"Inertial view ({altitude_km:.0f} km orbit, zoomed on formation)")
    ax_eci.set_aspect("equal", adjustable="box")
    ax_eci.grid(True, alpha=0.25)
    ax_eci.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Example maneuver: {plan.plan_id.replace('_', ' ')}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_koz_schematic_png(
    inner: EllipsoidKeepOut,
    x0: np.ndarray,
    scenario: LLMScenario,
    output_path: Path,
) -> Path:
    """Standalone KOZ ellipsoid (LVLH x–y slice) with deputy start and axis labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    _plot_koz_xy(ax, inner, annotate_axes=True)
    p0 = np.asarray(x0, dtype=np.float64).reshape(6)
    ax.scatter(
        float(p0[0]),
        float(p0[1]),
        s=90,
        c="tab:blue",
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
        label=f"Deputy start ({p0[1]:.0f} m downrange)",
    )
    ax.scatter(0.0, 0.0, s=70, c="0.15", marker="s", zorder=5, label="Chief")
    ax.annotate(
        "",
        xy=(float(p0[0]), float(p0[1])),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.2, linestyle="--"),
        zorder=4,
    )
    ax.set_xlabel("Radial offset (m)")
    ax.set_ylabel("Along-track offset (m)")
    ax.set_title(
        f"Keep-out zone around the chief\n"
        f"Ellipsoid semi-axes: {int(inner.semi_axes[0])} × {int(inner.semi_axes[1])} × {int(inner.semi_axes[2])} m"
    )
    pad = max(40.0, float(p0[1]) * 0.08)
    ax.set_xlim(-inner.semi_axes[0] - pad, inner.semi_axes[0] + pad)
    ax.set_ylim(-pad, float(p0[1]) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_benchmark_barchart_png(
    summary: dict[str, Any],
    output_path: Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Grouped bar chart: key filter / mission metrics (no_filter vs brt_filter)."""
    import matplotlib.pyplot as plt

    nf = summary.get("by_condition", {}).get("no_filter", {})
    bf = summary.get("by_condition", {}).get("brt_filter", {})
    if not nf and not bf:
        return None

    specs: list[tuple[str, str, bool]] = [
        ("llm_unsafe_rate", "Flagged\nunsafe", False),
        ("brt_unsafe_nominal_rate", "Nominal\nV unsafe", False),
        ("post_filter_unsafe_rate", "Still unsafe\nafter filter", False),
        ("filter_safety_success_rate", "Passes\nsafety checks", False),
        ("mission_success_rate", "Capture\nsuccess", False),
        ("mission_success_tier_b_rate", "Approach\nprogress", False),
        ("interception_rate", "Entered\nkeep-out", False),
        ("brt_intervention_rate", "Filter\nchanged burns", True),
    ]

    labels: list[str] = []
    vals_nf: list[float] = []
    vals_bf: list[float] = []
    for key, label, brt_only in specs:
        if brt_only and key not in bf:
            continue
        if not brt_only and key not in nf and key not in bf:
            continue
        labels.append(label)
        vals_nf.append(100.0 * float(nf.get(key, 0.0)) if not brt_only else 0.0)
        vals_bf.append(100.0 * float(bf.get(key, 0.0)))

    if not labels:
        return None

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 1.05), 5.0))
    ax.bar(x - w / 2, vals_nf, width=w, label="No filter", color="tab:gray", alpha=0.85)
    ax.bar(x + w / 2, vals_bf, width=w, label="With BRT filter", color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Share of plans (%)")
    ax.set_ylim(0, 105.0)
    n_plans = int(nf.get("n_plans", bf.get("n_plans", 0)))
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"LLM maneuver benchmark ({n_plans} plans, 400 km LEO)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _category_label(category: str) -> str:
    return category.replace("_", " ").strip() or "maneuver"


def render_case_study_png(
    plant: CWDynamics,
    inner: EllipsoidKeepOut,
    brt: Any,
    plan: LLMPlan,
    x0: np.ndarray,
    output_path: Path,
    *,
    passive_horizon_s: float,
    dv_cap_m_s: float | None,
) -> Path:
    """Three-panel case study: trajectories, burns, post-burn V."""
    import matplotlib.pyplot as plt

    nominal = plan.segments
    filtered, filt_results = filter_maneuver_plan(
        plant,
        x0,
        nominal,
        brt,
        inner_koz=inner,
        passive_horizon_s=passive_horizon_s,
        dv_cap_m_s=dv_cap_m_s,
    )

    _, xy_nom, _ = _rollout_xy(plant, x0, nominal)
    _, xy_filt, _ = _rollout_xy(plant, x0, filtered)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    ax = axes[0]
    _plot_koz_xy(ax, inner, alpha=0.25)
    ax.plot(xy_nom[:, 0], xy_nom[:, 1], "r--", linewidth=1.8, label="LLM plan")
    ax.plot(xy_filt[:, 0], xy_filt[:, 1], "g-", linewidth=2.0, label="After filter")
    for fr in filt_results:
        pos = np.asarray(x0, dtype=np.float64).reshape(6).copy()
        tb = 0.0
        for dt, dv in nominal:
            if abs(tb - fr.time_s) < 1e-6:
                break
            if dv is not None:
                pos = plant.apply_impulsive_dv(pos, dv)
            pos = plant.propagate(pos, float(dt))
            tb += float(dt)
        ax.scatter(pos[0], pos[1], s=40, c="tab:red", marker="x", zorder=5)
    for fr in filt_results:
        if np.linalg.norm(fr.dv_applied) > 1e-9:
            pos = np.asarray(x0, dtype=np.float64).reshape(6).copy()
            tb = 0.0
            for dt, dv in filtered:
                if abs(tb - fr.time_s) < 1e-6:
                    break
                if dv is not None:
                    pos = plant.apply_impulsive_dv(pos, dv)
                pos = plant.propagate(pos, float(dt))
                tb += float(dt)
            ax.scatter(pos[0], pos[1], s=50, facecolors="none", edgecolors="tab:green", linewidths=1.5, zorder=6)
    _draw_craft_2d(ax, np.zeros(3), np.zeros(3), scale_m=40, color="0.2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Radial (m)")
    ax.set_ylabel("Along-track (m)")
    ax.set_title("Relative trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    burn_idx = np.arange(len(filt_results))
    dv_nom = [float(np.linalg.norm(fr.dv_nominal)) for fr in filt_results]
    dv_app = [float(np.linalg.norm(fr.dv_applied)) for fr in filt_results]
    times_b = [fr.time_s for fr in filt_results]
    w = 0.35
    ax.bar(burn_idx - w / 2, dv_nom, width=w, label="Requested", color="tab:red", alpha=0.7)
    ax.bar(burn_idx + w / 2, dv_app, width=w, label="Applied", color="tab:green", alpha=0.7)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=1, label="0.5 m/s cap")
    for i, fr in enumerate(filt_results):
        tag = "dropped" if np.linalg.norm(fr.dv_applied) < 1e-9 else ("scaled" if fr.residual_norm > 1e-9 else "ok")
        ax.text(i, max(dv_nom[i], dv_app[i]) + 0.03, tag, ha="center", fontsize=7)
    ax.set_xticks(burn_idx)
    ax.set_xticklabels([f"{t:.0f}s" for t in times_b], fontsize=8)
    ax.set_xlabel("Burn time (s)")
    ax.set_ylabel("Burn size |Δv| (m/s)")
    ax.set_title("What the filter did to each burn")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2]
    v_nom, v_filt, labels = [], [], []
    for fr in filt_results:
        labels.append(f"{fr.time_s:.0f}s")
        x_pre = np.asarray(x0, dtype=np.float64).reshape(6).copy()
        t = 0.0
        for dt, dv in nominal:
            if abs(t - fr.time_s) < 1e-6:
                break
            if dv is not None:
                x_pre = plant.apply_impulsive_dv(x_pre, dv)
            x_pre = plant.propagate(x_pre, float(dt))
            t += float(dt)
        v_nom.append(float(brt.value_at_tau(plant.apply_impulsive_dv(x_pre, fr.dv_nominal), fr.time_s)))
        v_filt.append(float(brt.value_at_tau(plant.apply_impulsive_dv(x_pre, fr.dv_applied), fr.time_s)))
    idx = np.arange(len(v_nom))
    ax.bar(idx - 0.2, v_nom, width=0.4, label="Before filter", color="tab:red", alpha=0.75)
    ax.bar(idx + 0.2, v_filt, width=0.4, label="After filter", color="tab:green", alpha=0.75)
    ax.axhline(0.0, color="k", linewidth=1.2, label="Unsafe (V ≤ 0)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Burn time (s)")
    ax.set_ylabel("Value function V")
    ax.set_title("Safety margin after each burn")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    cat = _category_label(str(plan.tags.get("category", "")))
    fig.suptitle(f"{cat.title()} — {plan.plan_id.replace('_', ' ')}", fontsize=11, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_corpus_tables(
    scenario: LLMScenario,
    plans: list[LLMPlan],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Corpus composition table (CSV + Markdown)."""
    rows: list[dict[str, Any]] = []
    by_cat = Counter(p.tags.get("category", "?") for p in plans)
    by_ang = Counter(p.tags.get("approach_angle", "?") for p in plans)
    n_burns = [p.n_burns for p in plans]
    max_dv = []
    for p in plans:
        m = 0.0
        for _, dv in p.segments:
            if dv is not None:
                m = max(m, float(np.linalg.norm(dv)))
        max_dv.append(m)

    ys = scenario.start_y_values_m
    summary_rows = [
        {"metric": "scenario_id", "value": scenario.id},
        {"metric": "n_plans", "value": len(plans)},
        {"metric": "start_y_m", "value": float(scenario.start_state_lvlh_m[1])},
        {"metric": "brt_horizon_s", "value": float(scenario.brt_horizon_s)},
        {"metric": "dv_cap_m_s", "value": scenario.dv_cap_m_s or 0.5},
        {"metric": "koz_semi_axes_m", "value": str(tuple(int(a) for a in scenario.semi_axes_m))},
        {"metric": "n_burns_min", "value": min(n_burns)},
        {"metric": "n_burns_max", "value": max(n_burns)},
        {"metric": "n_burns_mean", "value": round(float(np.mean(n_burns)), 2)},
        {"metric": "max_dv_nom_min_m_s", "value": round(min(max_dv), 4)},
        {"metric": "max_dv_nom_max_m_s", "value": round(max(max_dv), 4)},
        {"metric": "max_dv_nom_mean_m_s", "value": round(float(np.mean(max_dv)), 4)},
    ]
    if ys:
        summary_rows.insert(3, {"metric": "start_y_values_m", "value": str(tuple(int(y) for y in ys))})
    cap = scenario.capture_radius_m
    if cap is not None:
        summary_rows.append({"metric": "capture_radius_m", "value": float(cap)})
    for cat, n in sorted(by_cat.items()):
        summary_rows.append({"metric": f"category_{cat}", "value": n})
    for ang, n in sorted(by_ang.items()):
        summary_rows.append({"metric": f"approach_{ang}", "value": n})

    detail_rows: list[dict[str, Any]] = []
    for p in plans:
        md = 0.0
        for _, dv in p.segments:
            if dv is not None:
                md = max(md, float(np.linalg.norm(dv)))
        detail_rows.append(
            {
                "plan_id": p.plan_id,
                "category": p.tags.get("category", ""),
                "start_y_m": p.tags.get("start_y_m", ""),
                "approach_angle": p.tags.get("approach_angle", ""),
                "n_burns": p.n_burns,
                "max_dv_nom_m_s": round(md, 4),
                "expected_intervention": p.expected_intervention if p.expected_intervention is not None else "",
            }
        )

    csv_path = out_dir / "corpus_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(summary_rows)

    detail_csv = out_dir / "corpus_plans.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    md_path = out_dir / "corpus_summary.md"
    start_desc = (
        f"deputy starts at y ∈ {{{', '.join(f'{int(y):.0f}' for y in ys)} m}}"
        if ys
        else f"deputy starts {scenario.start_state_lvlh_m[1]:.0f} m downrange"
    )
    header = (
        f"**{scenario.id}** — {start_desc}, "
        f"{scenario.brt_horizon_s:.0f} s horizon, {scenario.dv_cap_m_s or 0.5} m/s burn cap."
    )
    if cap is not None:
        header += f" Capture radius: {cap:.0f} m."
    lines = [
        "# LLM maneuver corpus",
        "",
        header,
        "",
        "## Plan types",
        "",
        "| Type | Count |",
        "|------|------:|",
    ]
    for cat, n in sorted(by_cat.items()):
        lines.append(f"| {_category_label(cat)} | {n} |")
    lines.extend(["", "## Approach direction", "", "| Direction | Count |", "|-----------|------:|"])
    for ang, n in sorted(by_ang.items()):
        lines.append(f"| {ang.replace('_', ' ')} | {n} |")
    lines.extend(
        [
            "",
            "## Summary stats",
            "",
            f"- Burns per plan: {min(n_burns)}–{max(n_burns)} (avg {np.mean(n_burns):.1f})",
            f"- Largest burn: {min(max_dv):.3f}–{max(max_dv):.3f} m/s (avg {np.mean(max_dv):.3f})",
            "",
            f"Full listing: `{detail_csv.name}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_results_tables(
    results: list[PlanEvalResult],
    summary: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Experiment metrics table for no_filter and brt_filter."""
    metric_rows: list[dict[str, str]] = []
    labels = {
        "n_plans": "Number of plans",
        "llm_unsafe_rate": "Flagged unsafe (nominal plan)",
        "brt_intervention_rate": "Plans where filter changed a burn",
        "mean_burns_intervened_per_plan": "Burns modified per plan (mean)",
        "mean_burns_suppressed_per_plan": "Burns dropped per plan (mean)",
        "mean_burns_scaled_per_plan": "Burns scaled per plan (mean)",
        "mean_dv_overhead_m_s": "Extra Δv from filtering (m/s, mean)",
        "post_filter_unsafe_rate": "Still fails safety checks",
        "filter_safety_success_rate": "Passes all safety checks",
        "passive_unsafe_nominal_rate": "Passive-unsafe before burn (nominal)",
        "brt_unsafe_nominal_rate": "V ≤ 0 after burn (nominal)",
        "brt_unsafe_rate": "Any post-burn V ≤ 0",
        "requires_intervention_rate": "Needs intervention (rollout)",
        "mission_success_safe_rate": "Safe by intervention criteria",
        "label_match_rate": "Matches corpus label",
        "interception_rate": "Entered keep-out zone",
        "mission_success_tier_b_rate": "Made approach progress (≥50 m)",
        "mean_range_closed_m": "Range closed (m, mean)",
    }
    cond_names = {"no_filter": "No filter", "brt_filter": "With filter"}
    for cond in ("no_filter", "brt_filter"):
        stats = summary.get("by_condition", {}).get(cond, {})
        for key, label in labels.items():
            if key not in stats:
                continue
            val = stats[key]
            if isinstance(val, float) and "rate" in key:
                disp = f"{100.0 * val:.1f}%"
            elif isinstance(val, float):
                disp = f"{val:.4f}"
            else:
                disp = str(val)
            metric_rows.append({"condition": cond_names.get(cond, cond), "metric": label, "value": disp})

    csv_path = out_dir / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "metric", "value"])
        w.writeheader()
        w.writerows(metric_rows)

    md_lines = [
        "# Experiment results",
        "",
        "| Metric | No filter | With filter |",
        "|--------|-----------|-------------|",
    ]
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    for row in metric_rows:
        pivot[row["metric"]][row["condition"]] = row["value"]
    for label in labels.values():
        if label not in pivot:
            continue
        nf = pivot[label].get("No filter", "—")
        bf = pivot[label].get("With filter", "—")
        md_lines.append(f"| {label} | {nf} | {bf} |")

    cat_lines = [
        "",
        "## Breakdown by plan type (unfiltered runs)",
        "",
        "| Plan type | Count | Flagged unsafe |",
        "|-----------|------:|---------------:|",
    ]
    for cat, stats in summary.get("by_category", {}).items():
        n = stats.get("n_plans", 0)
        u = stats.get("llm_unsafe_rate", 0)
        cat_lines.append(f"| {_category_label(cat)} | {n} | {100*u:.0f}% |")

    md_path = out_dir / "experiment_results.md"
    md_path.write_text("\n".join(md_lines + cat_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_reference_trajectories_table(
    summary: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    """Per-plan outcomes for hand-tuned reference schedules (not aggregated rates)."""
    refs = summary.get("reference_trajectories", [])
    if not refs:
        return None
    path = out_dir / "reference_trajectories.csv"
    fields = list(refs[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(refs)

    md = [
        "# Reference trajectories",
        "",
        "Hand-tuned radial braking schedules (not LLM plans). Listed individually — not pooled into rates.",
        "",
        "| Plan | Burns | Max |Δv| (m/s) | Entered KOZ | Approach progress | Range closed (m) | Final range (m) |",
        "|------|------:|-------------:|:-----------:|:-----------------:|-----------------:|----------------:|",
    ]
    for r in refs:
        md.append(
            f"| `{r['plan_id']}` | {r['n_burns']} | {r['max_dv_nom_m_s']:.3f} | "
            f"{'yes' if r['intercepted'] else 'no'} | "
            f"{'yes' if r['mission_success_tier_b'] else 'no'} | "
            f"{r['range_closed_m']:.1f} | {r['final_range_m']:.1f} |"
        )
    (out_dir / "reference_trajectories.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Generate report figures and tables.")
    p.add_argument("--llm-dir", type=str, default=str(default_llm_dir(root)))
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get(
            "DEEPREACH_CHECKPOINT_DIR",
            str(default_checkpoint_dir()),
        ),
    )
    p.add_argument(
        "--results-csv",
        type=str,
        default=str(root / "simulation_output" / "llm_benchmark_results.csv"),
    )
    p.add_argument("--output-dir", type=str, default=str(root / "simulation_output" / "report"))
    p.add_argument("--device", type=str, default=os.environ.get("DEEPREACH_DEVICE", "auto"))
    p.add_argument(
        "--representative-plan",
        type=str,
        default="aggr_intercept_along_track_009",
        help="Plan for scenario orbit figure.",
    )
    p.add_argument(
        "--case-studies",
        type=str,
        default="safe_multi_along_track_000,aggr_intercept_along_track_009",
    )
    args = p.parse_args()

    try:
        from simulation.brt.deepreach_mpc_brt import DEEPREACH_MPC_AVAILABLE, KozDeepReachBRT
    except ImportError:
        DEEPREACH_MPC_AVAILABLE = False

    if not DEEPREACH_MPC_AVAILABLE:
        print("DeepReach-MPC required for BRT figures.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    scenario, plans = load_llm_plans(args.llm_dir)
    plant = CWDynamics(scenario.mean_motion_rad_s)
    inner = EllipsoidKeepOut(np.array(scenario.semi_axes_m, dtype=np.float64))
    passive_h = float(os.environ.get("PASSIVE_CHECK_HORIZON_S", str(BRT_HORIZON_S)))

    rep = next((pl for pl in plans if pl.plan_id == args.representative_plan), plans[0])
    x0 = rep.x0(scenario)
    print("Rendering KOZ schematic…")
    render_koz_schematic_png(inner, x0, scenario, fig_dir / "koz_schematic_xy.png")
    print(f"Rendering scenario orbit ({rep.plan_id})…")
    render_scenario_orbit_png(plant, scenario, inner, rep, fig_dir / "scenario_orbit_lvlh.png")

    ck_dir = Path(args.checkpoint_dir).resolve()
    print(f"Loading BRT from {ck_dir}…")
    brt = KozDeepReachBRT.load(ck_dir, device=args.device)

    from simulation.brt.slice_viz import parse_time_slices_s, render_brt_koz_centered_png

    from simulation.brt.slice_viz import default_report_koz_xy_limits

    times = parse_time_slices_s(brt.horizon_s, (0.0, 450.0, 900.0, 1350.0, 1800.0))
    axes_sa = tuple(float(x) for x in scenario.semi_axes_m)
    rx, ry = default_report_koz_xy_limits(axes_sa)
    os.environ.setdefault("BRT_KOZ_VIZ_COLOR", "shared")
    print(f"Rendering BRT training slices ({len(times)} panels)…")
    render_brt_koz_centered_png(
        brt,
        str(fig_dir / "brt_training_slices_xy.png"),
        inner_koz=inner,
        semi_axes_m=axes_sa,
        times_s=times,
        xlim=rx,
        ylim=ry,
        grid_n=int(os.environ.get("BRT_KOZ_VIZ_GRID_N", "90")),
    )

    if os.environ.get("REPORT_BRT_3D", "1").lower() not in ("0", "false", "no"):
        from simulation.snapshot_viz import render_brt_lvlh_snapshot

        os.environ.setdefault("BRT_SNAPSHOT_HALF_XYZ", "220,2000,160")
        os.environ.setdefault("BRT_SNAPSHOT_DISPLAY_HALF_XYZ", "260,1300,180")
        os.environ.setdefault("BRT_SNAPSHOT_MAX_HALF_M", "2200")
        os.environ.setdefault("BRT_SNAPSHOT_VIEW_HALF_M", "900")
        os.environ.setdefault("BRT_SNAPSHOT_MESH_RADIUS_M", "2100")
        print("Rendering 3D BRT formation snapshot (may take ~1 min)…")
        render_brt_lvlh_snapshot(
            brt,
            inner,
            x0,
            output_path_png=str(fig_dir / "brt_formation_lvlh.png"),
            output_path_gif=None,
            chief_box_half_m=900.0,
            max_search_half_m=2200.0,
            iso_resolution=(28, 32, 20),
        )

    results_path = Path(args.results_csv)
    eval_results: list[PlanEvalResult] = []
    summary: dict[str, Any] = {"by_condition": {}, "by_category": {}}
    if results_path.is_file():
        raw = _load_results_csv(results_path)
        eval_results = _results_to_plan_eval(raw)
        summary = summarize_results(eval_results)
    else:
        print(f"Warning: no results at {results_path}; tables will be partial.", file=sys.stderr)

    print("Writing corpus tables…")
    write_corpus_tables(scenario, plans, tab_dir)
    print("Writing experiment results tables…")
    write_results_tables(eval_results, summary, tab_dir)
    ref_path = write_reference_trajectories_table(summary, tab_dir)
    if ref_path:
        print(f"Writing reference trajectories table… ({ref_path.name})")
    if summary.get("by_condition"):
        print("Rendering benchmark bar chart…")
        render_benchmark_barchart_png(summary, fig_dir / "benchmark_results_barchart.png")

    for pid in [s.strip() for s in args.case_studies.split(",") if s.strip()]:
        plan = next((pl for pl in plans if pl.plan_id == pid), None)
        if plan is None:
            print(f"  Skip case study (not found): {pid}")
            continue
        print(f"Rendering case study {pid}…")
        render_case_study_png(
            plant,
            inner,
            brt,
            plan,
            x0,
            fig_dir / f"case_study_{pid}.png",
            passive_horizon_s=passive_h,
            dv_cap_m_s=scenario.dv_cap_m_s,
        )

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Report figures and tables",
                "",
                "## Figures",
                "- `koz_schematic_xy.png` — keep-out zone and deputy starting position",
                "- `scenario_orbit_lvlh.png` — relative motion + inertial orbit view",
                "- `brt_training_slices_xy.png` — learned value function on x–y slices",
                "- `brt_formation_lvlh.png` — 3D unsafe set near the chief",
                "- `benchmark_results_barchart.png` — summary of filter and mission metrics",
                "- `case_study_*.png` — side-by-side examples (safe vs aggressive)",
                "",
                "## Tables",
                "- `corpus_summary.md` — what's in the plan bundle",
                "- `experiment_results.md` — benchmark numbers (no filter vs filtered)",
                "- `reference_trajectories.md` — hand-tuned radial schedules (per-plan, not rates)",
                "",
                "Regenerate:",
                "```",
                "DEEPREACH_AUTO_TRAIN=0 python -m simulation.benchmark",
                "DEEPREACH_AUTO_TRAIN=0 python -m simulation.benchmark.generate_report",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
