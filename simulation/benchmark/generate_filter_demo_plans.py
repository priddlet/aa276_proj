"""Generate synthetic LLM-style plan bundle to demonstrate BRT line-search filter.

Plans start at y≈250 m (learned BRT scale). Each has a coast phase
then a burn at t≈550 s where V(x, t) is near the BRT boundary so aggressive Δv can
be BRT-unsafe while the filter scales α·Δv back to V>0.

Usage:
  python -m simulation.benchmark.generate_filter_demo_plans
  LLM_PLANS_DIR=llm/llm_plans_brt_demo python -m simulation.benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from simulation.benchmark.label_metrics import assess_corpus_intervention
from simulation.brt.deepreach_mpc_brt import KozDeepReachBRT, default_checkpoint_dir
from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import (
    finalize_segments_for_rollout,
    segments_record_to_sim_segments,
)
from simulation.sampling.passive import is_passively_safe_natural_coast
from simulation.sampling.safety_filter import (
    default_check_passive_post,
    filter_impulsive_burn_linesearch,
    filter_maneuver_plan,
)


def _burns_to_segment_records(
    maneuvers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    burns = sorted(maneuvers, key=lambda m: float(m["t_s"]))
    segs: list[dict[str, Any]] = []
    t_prev = 0.0
    for m in burns:
        t_b = float(m["t_s"])
        gap = t_b - t_prev
        if gap > 0.0:
            segs.append({"coast_s": gap, "dv_m_s": None})
        dv = [float(x) for x in m["dv_m_s"]]
        segs.append({"coast_s": 0.0, "dv_m_s": dv})
        t_prev = t_b
    return segs


def _plan_to_maneuvers(segments: list[tuple[float, np.ndarray | None]]) -> list[dict[str, Any]]:
    maneuvers: list[dict[str, Any]] = []
    t = 0.0
    for dt, dv in segments:
        if dv is not None and np.any(dv != 0):
            maneuvers.append({"t_s": t, "dv_m_s": [float(x) for x in dv.reshape(3)]})
        t += float(dt)
    return maneuvers


def _filter_alphas_for_plan(
    plant: CWDynamics,
    brt: Any,
    inner: EllipsoidKeepOut,
    x0: np.ndarray,
    sim_segs: list[tuple[float, np.ndarray | None]],
    *,
    passive_horizon_s: float,
    check_passive_from_post: bool,
) -> list[float]:
    _, frs = filter_maneuver_plan(
        plant,
        x0,
        sim_segs,
        brt,
        filter_mode="linesearch",
        inner_koz=inner,
        passive_horizon_s=passive_horizon_s,
        check_passive_from_post=check_passive_from_post,
    )
    return [float(fr.scale_alpha) for fr in frs]


def _search_unsafe_burns(
    plant: CWDynamics,
    brt: Any,
    inner: EllipsoidKeepOut,
    x_burn: np.ndarray,
    t_burn: float,
    *,
    passive_horizon_s: float,
    rng: np.random.Generator,
    n_want: int,
    n_random: int = 6000,
) -> list[tuple[np.ndarray, float, float]]:
    """Return (dv, alpha, v_post_nom) with 0 < alpha < 1 and V_post(nom) <= 0."""
    v_pre = float(brt.value_at_tau(x_burn, t_burn))
    if v_pre <= 0.0:
        return []
    if not is_passively_safe_natural_coast(plant, x_burn, inner, passive_horizon_s, n_samples=64):
        return []
    out: list[tuple[np.ndarray, float, float]] = []
    check_post = default_check_passive_post()

    def _try_dv(dv: np.ndarray) -> None:
        xp = plant.apply_impulsive_dv(x_burn, dv)
        v_post = float(brt.value_at_tau(xp, t_burn))
        if v_post > 0.0:
            return
        fr = filter_impulsive_burn_linesearch(
            plant,
            x_burn,
            dv,
            brt,
            t_burn,
            inner_koz=inner,
            passive_horizon_s=passive_horizon_s,
            check_passive_from_post=check_post,
        )
        if 0.05 < fr.scale_alpha < 0.95:
            out.append((dv.copy(), float(fr.scale_alpha), v_post))

    # Directed along-track braking toward chief (common failure mode at approach range).
    u_close = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for mag in np.linspace(0.08, 1.4, 28):
        _try_dv(mag * u_close)
        if len(out) >= n_want:
            return out

    for _ in range(int(n_random)):
        dv = rng.uniform(-1.4, 1.4, size=3)
        dv[2] *= 0.15
        _try_dv(dv)
        if len(out) >= n_want:
            break
    return out


def _probe_correctable_burn(
    plant: CWDynamics,
    brt: Any,
    inner: EllipsoidKeepOut,
    x_burn: np.ndarray,
    t_burn: float,
    *,
    passive_horizon_s: float,
) -> int:
    """Fast count of correctable burns (directed along-track only)."""
    v_pre = float(brt.value_at_tau(x_burn, t_burn))
    if v_pre <= 0.0:
        return 0
    if not is_passively_safe_natural_coast(plant, x_burn, inner, passive_horizon_s, n_samples=32):
        return 0
    check_post = default_check_passive_post()
    score = 0
    u_close = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for mag in (0.2, 0.45, 0.75, 1.05):
        dv = mag * u_close
        xp = plant.apply_impulsive_dv(x_burn, dv)
        if float(brt.value_at_tau(xp, t_burn)) > 0.0:
            continue
        fr = filter_impulsive_burn_linesearch(
            plant,
            x_burn,
            dv,
            brt,
            t_burn,
            inner_koz=inner,
            passive_horizon_s=passive_horizon_s,
            check_passive_from_post=check_post,
        )
        if 0.05 < fr.scale_alpha < 0.95:
            score += 1
    return score


def _find_demo_burn_time(
    plant: CWDynamics,
    brt: Any,
    inner: EllipsoidKeepOut,
    x0: np.ndarray,
    *,
    passive_horizon_s: float,
    preferred_s: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Pick burn time and start state with many filter-correctable nominal-unsafe impulses."""
    candidates = sorted(
        {round(t, 1) for t in np.arange(320.0, 860.0, 40.0)} | {float(preferred_s)}
    )
    best_t = float(preferred_s)
    best_x0 = x0.copy()
    best_score = -1
    y0 = float(x0[1])
    for y_m in (y0, 150.0, 180.0, 200.0):
        x_start = np.array([0.0, y_m, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for t_b in candidates:
            x_b = plant.propagate(x_start, float(t_b))
            score = _probe_correctable_burn(
                plant, brt, inner, x_b, float(t_b), passive_horizon_s=passive_horizon_s
            )
            if score > best_score:
                best_score = score
                best_t = float(t_b)
                best_x0 = x_start.copy()
    if best_score < 1:
        raise RuntimeError(
            f"Could not find filter-correctable burns near y={y0:.0f} m "
            f"(best t={best_t:.0f}s, score={best_score})."
        )
    return best_t, best_x0


def _extract_corpus_demo_burns(
    llm_dir: Path,
    plant: CWDynamics,
    brt: Any,
    inner: EllipsoidKeepOut,
    *,
    passive_horizon_s: float,
    max_plans: int = 18,
) -> tuple[dict[str, Any], list[tuple[float, np.ndarray, float, str]]]:
    """Pull nominal burns from the LEO corpus that the filter scales (0 < α < 1)."""
    from simulation.llm_plans import load_llm_plans

    scenario, plans = load_llm_plans(llm_dir)
    x0 = np.array(scenario.start_state_lvlh_m, dtype=np.float64)
    passive_post = default_check_passive_post()
    found: list[tuple[float, np.ndarray, float, str]] = []
    for pl in plans:
        _, frs = filter_maneuver_plan(
            plant,
            x0,
            pl.segments,
            brt,
            inner_koz=inner,
            passive_horizon_s=passive_horizon_s,
            check_passive_from_post=passive_post,
        )
        t = 0.0
        fr_idx = 0
        for dt, dv in pl.segments:
            if dv is not None:
                fr = frs[fr_idx]
                fr_idx += 1
                alpha = float(fr.scale_alpha)
                if 0.05 < alpha < 0.98 and float(np.linalg.norm(fr.dv_nominal)) > 1e-6:
                    found.append((float(t), np.asarray(fr.dv_nominal, dtype=np.float64), alpha, pl.plan_id))
            t += float(dt)
        if len(found) >= max_plans:
            break
    scen_dict = {
        "id": "scenario_brt_filter_demo_v1",
        "mean_motion_rad_s": float(scenario.mean_motion_rad_s),
        "start_state_m_m_s": x0.tolist(),
        "inner_koz": {"type": "ellipsoid", "semi_axes_m": list(scenario.semi_axes_m)},
        "brt_horizon_s": float(passive_horizon_s),
    }
    return scen_dict, found[:max_plans]


def generate_filter_demo_bundle(
    out_dir: Path,
    *,
    start_y_m: float = 250.0,
    burn_time_s: float = 550.0,
    brt_horizon_s: float = 1800.0,
    mean_motion_rad_s: float = 0.0011313666536110225,
    checkpoint_dir: Path | None = None,
    seed: int = 42,
    llm_dir: Path | None = None,
) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = checkpoint_dir or default_checkpoint_dir()
    brt = KozDeepReachBRT.load(ckpt, device="cpu")
    plant = CWDynamics(mean_motion_rad_s)
    inner = EllipsoidKeepOut(np.array([28.0, 45.0, 18.0], dtype=np.float64))
    passive_post = default_check_passive_post()
    rng = np.random.default_rng(seed)

    root = Path(__file__).resolve().parents[2]
    corpus_dir = llm_dir or (root / "llm")
    corpus_meta, corpus_burns = _extract_corpus_demo_burns(
        corpus_dir,
        plant,
        brt,
        inner,
        passive_horizon_s=brt_horizon_s,
    )

    if len(corpus_burns) >= 6:
        x0 = np.array(corpus_meta["start_state_m_m_s"], dtype=np.float64)
        burn_time_s = float(corpus_burns[0][0])
        unsafe = list(corpus_burns)
        print(
            f"[filter-demo] extracted {len(unsafe)} scaled burns from {corpus_dir} "
            f"(y={x0[1]:.0f} m, example t={burn_time_s:.0f} s)"
        )
    else:
        x0 = np.array([0.0, float(start_y_m), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        burn_time_s, x0 = _find_demo_burn_time(
            plant,
            brt,
            inner,
            x0,
            passive_horizon_s=brt_horizon_s,
            preferred_s=float(burn_time_s),
            rng=rng,
        )
        x_burn = plant.propagate(x0, float(burn_time_s))
        print(
            f"[filter-demo] using start y={x0[1]:.0f} m, burn_time_s={burn_time_s:.0f} (random search)"
        )
        raw = _search_unsafe_burns(
            plant,
            brt,
            inner,
            x_burn,
            float(burn_time_s),
            passive_horizon_s=brt_horizon_s,
            rng=rng,
            n_want=18,
        )
        unsafe = [
            (float(burn_time_s), dv, alpha, "search")
            for dv, alpha, _ in raw
        ]
        if len(unsafe) < 6:
            raise RuntimeError(
                f"Could not find enough filter-correctable burns at y={start_y_m}, t={burn_time_s}. "
                "Regenerate the LEO corpus or pass --llm-dir with scaled-burn plans."
            )

    safe_dvs = [
        np.array([-0.008, -0.04, 0.0]),
        np.array([-0.015, -0.06, 0.0]),
        np.array([-0.02, -0.05, 0.01]),
        np.array([-0.01, -0.08, -0.005]),
        np.array([-0.025, -0.03, 0.0]),
        np.array([-0.005, -0.03, 0.008]),
    ]

    scenario: dict[str, Any] = {
        "id": "scenario_brt_filter_demo_v1",
        "frame": "LVLH (x=radial, y=along-track, z=cross-track), chief at origin",
        "dynamics": "Clohessy-Wiltshire (linearized Hill)",
        "mean_motion_rad_s": float(corpus_meta.get("mean_motion_rad_s", mean_motion_rad_s))
        if len(corpus_burns) >= 6
        else mean_motion_rad_s,
        "orbital_period_s": 2.0 * math.pi / mean_motion_rad_s,
        "start_state_m_m_s": x0.tolist(),
        "goal": "Demonstrate BRT line-search filter on near-boundary approach burns.",
        "inner_koz": {"type": "ellipsoid", "semi_axes_m": [28.0, 45.0, 18.0]},
        "brt_horizon_s": float(brt_horizon_s),
        "label_criteria": {
            "brt_proxy_unsafe": "post-burn V(x,t_burn) <= 0",
            "passive_unsafe": "natural coast from pre-burn state enters KOZ within brt_horizon_s",
            "dv_excessive": "per-burn |dv| > 0.5 m/s",
            "requires_intervention": "logical OR of the above (demo bundle; no corridor_far in labels)",
        },
    }

    plans: list[dict[str, Any]] = []
    seg_lines: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    def add_plan(
        plan_id: str,
        category: str,
        prompt: str,
        dv: np.ndarray,
        *,
        urgency: str = "med",
        burn_t: float | None = None,
        source_plan: str = "",
    ) -> None:
        t_b = float(burn_time_s if burn_t is None else burn_t)
        maneuvers = [{"t_s": t_b, "dv_m_s": [float(x) for x in dv.reshape(3)]}]
        seg_recs = _burns_to_segment_records(maneuvers)
        raw_segs = segments_record_to_sim_segments(seg_recs)
        sim_segs = finalize_segments_for_rollout(raw_segs, float(brt_horizon_s))

        label_assess = assess_corpus_intervention(
            plant,
            x0,
            sim_segs,
            inner,
            brt,
            passive_horizon_s=brt_horizon_s,
        )
        alphas = _filter_alphas_for_plan(
            plant,
            brt,
            inner,
            x0,
            sim_segs,
            passive_horizon_s=brt_horizon_s,
            check_passive_from_post=passive_post,
        )
        v_nom = float(
            brt.value_at_tau(
                plant.apply_impulsive_dv(plant.propagate(x0, t_b), dv),
                t_b,
            )
        )

        plans.append(
            {
                "plan_id": plan_id,
                "prompt": prompt,
                "tags": {
                    "category": category,
                    "approach_angle": "along_track",
                    "urgency": urgency,
                    "range_framing": f"mid_{start_y_m:.0f}m",
                    "burn_time_s": float(burn_time_s),
                },
                "scenario_ref": scenario["id"],
                "maneuvers": maneuvers,
                "label": {
                    "expected_intervention": int(label_assess.requires_intervention),
                    "intervention_reasons": list(label_assess.reasons),
                    "nominal_post_burn_V": v_nom,
                    "filter_alphas": alphas,
                    "n_burns": len(maneuvers),
                    "source_plan": source_plan,
                },
            }
        )
        seg_lines.append(
            {
                "plan_id": plan_id,
                "prompt": prompt,
                "segment_semantics": "burn_at_start_then_coast",
                "segments": seg_recs,
            }
        )
        summary_rows.append(
            {
                "plan_id": plan_id,
                "category": category,
                "approach_angle": "along_track",
                "urgency": urgency,
                "n_burns": len(maneuvers),
                "total_dv_m_s": float(np.linalg.norm(dv)),
                "max_dv_m_s": float(np.linalg.norm(dv)),
                "expected_intervention": int(label_assess.requires_intervention),
                "intervention_reasons": ",".join(label_assess.reasons),
                "filter_alpha": alphas[0] if alphas else 1.0,
                "nominal_post_burn_V": v_nom,
            }
        )

    for i, dv in enumerate(safe_dvs):
        add_plan(
            f"brt_demo_safe_{i:02d}",
            "brt_demo_safe",
            f"Small along-track trim at t={burn_time_s:.0f}s from y={start_y_m:.0f} m (should pass filter at α=1).",
            dv,
            urgency="low",
        )

    for i, (t_b, dv, alpha, src) in enumerate(unsafe):
        add_plan(
            f"brt_demo_unsafe_{i:02d}",
            "brt_demo_unsafe",
            f"Aggressive burn at t={t_b:.0f}s from y={x0[1]:.0f} m "
            f"(filter expected α≈{alpha:.2f}; source={src or 'corpus'}).",
            dv,
            urgency="high" if i % 2 else "med",
            burn_t=float(t_b),
            source_plan=str(src),
        )

    bundle = {"scenario": scenario, "plans": plans}
    (out_dir / "llm_plans.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    with (out_dir / "llm_plans_segments.jsonl").open("w", encoding="utf-8") as f:
        for rec in seg_lines:
            f.write(json.dumps(rec) + "\n")

    fields = list(summary_rows[0].keys()) if summary_rows else []
    with (out_dir / "plans_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    return out_dir


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Generate BRT filter demonstration plan bundle.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "llm" / "llm_plans_brt_demo"),
    )
    p.add_argument("--start-y-m", type=float, default=250.0)
    p.add_argument("--burn-time-s", type=float, default=550.0)
    p.add_argument("--checkpoint-dir", type=str, default=str(default_checkpoint_dir()))
    p.add_argument(
        "--llm-dir",
        type=str,
        default="",
        help="LEO corpus dir for extracting scaled-burn demo plans (default: llm/).",
    )
    args = p.parse_args()

    llm_dir = Path(args.llm_dir).resolve() if args.llm_dir.strip() else None
    out = generate_filter_demo_bundle(
        Path(args.out_dir),
        start_y_m=args.start_y_m,
        burn_time_s=args.burn_time_s,
        checkpoint_dir=Path(args.checkpoint_dir),
        llm_dir=llm_dir,
    )
    print(f"Wrote filter demo bundle to {out}")
    print("Run benchmark:")
    print(f"  LLM_PLANS_DIR={out} python -m simulation.benchmark")


if __name__ == "__main__":
    main()
