"""Evaluate maneuver plans: no filter, BRT line-search filter, rule-based baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from simulation.benchmark.metrics import RolloutMetrics, compute_rollout_metrics
from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import LLMPlan, LLMScenario
from simulation.sampling.safety_filter import FilterMode, filter_maneuver_plan


class EvalCondition(str, Enum):
    NO_FILTER = "no_filter"
    BRT_FILTER = "brt_filter"
    RULE_BASED = "rule_based"


@dataclass
class PlanEvalResult:
    plan_id: str
    condition: str
    category: str
    approach_angle: str
    urgency: str
    n_burns: int
    max_dv_nom_m_s: float
    intercepted: bool
    mission_success: bool
    mean_dv_overhead_m_s: float
    n_burns_corrected: int
    final_range_m: float
    min_koz_shape_value: float
    brt_unsafe_any_post_burn: bool
    filter_n_accepted: int
    filter_n_burns: int
    expected_intervention: int | None

    def to_csv_row(self) -> dict[str, str | int | float]:
        row: dict[str, str | int | float] = {}
        for k, v in asdict(self).items():
            if v is None:
                row[k] = ""
            elif isinstance(v, bool):
                row[k] = int(v)
            else:
                row[k] = v
        return row


def _max_burn_dv(segments: list[tuple[float, np.ndarray | None]]) -> float:
    m = 0.0
    for _, dv in segments:
        if dv is not None:
            m = max(m, float(np.linalg.norm(dv)))
    return m


def _brt_unsafe_any_post_burn(
    brt: Any,
    post_burns: list[tuple[float, np.ndarray]],
    brt_margin: float,
) -> bool:
    for t_s, s in post_burns:
        v = float(brt.value_at_tau(s, t_s)) if hasattr(brt, "value_at_tau") else float(brt.value(s))
        if v <= float(brt_margin) or not np.isfinite(v):
            return True
    return False


def _post_burn_states(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
) -> list[tuple[float, np.ndarray]]:
    x = np.asarray(x0, dtype=np.float64).reshape(6).copy()
    t = 0.0
    out: list[tuple[float, np.ndarray]] = []
    for dt, dv in segments:
        if dv is not None:
            x = plant.apply_impulsive_dv(x, dv)
            out.append((t, x.copy()))
        x = plant.propagate(x, float(dt))
        t += float(dt)
    return out


def evaluate_plan(
    plant: CWDynamics,
    plan: LLMPlan,
    x0: np.ndarray,
    brt: Any | None,
    condition: EvalCondition,
    inner_koz: EllipsoidKeepOut,
    *,
    passive_horizon_s: float | None = None,
    filter_mode: FilterMode | None = None,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    capture_radius_m: float = 100.0,
    target_pos_m: np.ndarray | None = None,
) -> PlanEvalResult:
    """Roll out one plan under ``condition``; compute paper-style metrics."""
    nominal = plan.segments
    segs = nominal
    filt_results: list = []
    n_accepted = 0
    n_filt = 0

    if condition == EvalCondition.BRT_FILTER:
        if brt is None:
            raise ValueError("brt required for brt_filter condition")
        segs, filt_results = filter_maneuver_plan(
            plant,
            x0,
            nominal,
            brt,
            filter_mode=filter_mode,
            max_perturb_m_s=max_perturb_m_s,
            n_sphere_samples=n_sphere_samples,
            brt_margin=brt_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
        )
        n_filt = len(filt_results)
        n_accepted = sum(1 for fr in filt_results if fr.accepted)

    rollout = compute_rollout_metrics(
        plant,
        x0,
        segs,
        inner_koz,
        nominal_segments=nominal if condition == EvalCondition.BRT_FILTER else None,
        target_pos_m=target_pos_m,
        capture_radius_m=capture_radius_m,
    )

    unsafe_post = False
    if brt is not None:
        post = _post_burn_states(plant, x0, segs)
        unsafe_post = _brt_unsafe_any_post_burn(brt, post, brt_margin)

    tags = plan.tags
    return PlanEvalResult(
        plan_id=plan.plan_id,
        condition=condition.value,
        category=str(tags.get("category", "")),
        approach_angle=str(tags.get("approach_angle", "")),
        urgency=str(tags.get("urgency", "")),
        n_burns=rollout.n_burns,
        max_dv_nom_m_s=_max_burn_dv(nominal),
        intercepted=rollout.intercepted,
        mission_success=rollout.mission_success,
        mean_dv_overhead_m_s=rollout.mean_dv_overhead_m_s,
        n_burns_corrected=rollout.n_burns_corrected,
        final_range_m=rollout.final_range_m,
        min_koz_shape_value=rollout.min_koz_shape_value,
        brt_unsafe_any_post_burn=unsafe_post,
        filter_n_accepted=n_accepted,
        filter_n_burns=n_filt,
        expected_intervention=plan.expected_intervention,
    )


def run_llm_benchmark(
    scenario: LLMScenario,
    plans: list[LLMPlan],
    brt: Any,
    plant: CWDynamics,
    inner_koz: EllipsoidKeepOut,
    *,
    conditions: tuple[EvalCondition, ...] = (
        EvalCondition.NO_FILTER,
        EvalCondition.BRT_FILTER,
    ),
    rule_based_plan: LLMPlan | None = None,
    passive_horizon_s: float | None = None,
    filter_mode: FilterMode | None = None,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    capture_radius_m: float = 100.0,
) -> list[PlanEvalResult]:
    x0 = scenario.start_state_lvlh_m
    tgt = np.zeros(3, dtype=np.float64)
    results: list[PlanEvalResult] = []
    for plan in plans:
        if EvalCondition.NO_FILTER in conditions:
            results.append(
                evaluate_plan(
                    plant, plan, x0, brt, EvalCondition.NO_FILTER, inner_koz,
                    passive_horizon_s=passive_horizon_s, filter_mode=filter_mode,
                    max_perturb_m_s=max_perturb_m_s, n_sphere_samples=n_sphere_samples,
                    brt_margin=brt_margin, capture_radius_m=capture_radius_m, target_pos_m=tgt,
                )
            )
        if EvalCondition.BRT_FILTER in conditions:
            results.append(
                evaluate_plan(
                    plant, plan, x0, brt, EvalCondition.BRT_FILTER, inner_koz,
                    passive_horizon_s=passive_horizon_s, filter_mode=filter_mode,
                    max_perturb_m_s=max_perturb_m_s, n_sphere_samples=n_sphere_samples,
                    brt_margin=brt_margin, capture_radius_m=capture_radius_m, target_pos_m=tgt,
                )
            )

    if EvalCondition.RULE_BASED in conditions and rule_based_plan is not None:
        results.append(
            evaluate_plan(
                plant,
                rule_based_plan,
                x0,
                brt,
                EvalCondition.RULE_BASED,
                inner_koz,
                passive_horizon_s=passive_horizon_s,
                capture_radius_m=capture_radius_m,
                target_pos_m=tgt,
            )
        )
    return results


def write_results_csv(path: str | Path, results: list[PlanEvalResult]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(results[0].to_csv_row().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.to_csv_row())
    return path


def summarize_results(results: list[PlanEvalResult]) -> dict[str, Any]:
    """Aggregate interception rate, mission success, mean Δv overhead by condition."""
    out: dict[str, Any] = {"by_condition": {}, "n_rows": len(results)}
    for cond in sorted({r.condition for r in results}):
        rows = [r for r in results if r.condition == cond]
        if not rows:
            continue
        n = len(rows)
        out["by_condition"][cond] = {
            "n_plans": n,
            "interception_rate": sum(1 for r in rows if r.intercepted) / n,
            "mission_success_rate": sum(1 for r in rows if r.mission_success) / n,
            "mean_dv_overhead_m_s": float(np.mean([r.mean_dv_overhead_m_s for r in rows])),
            "brt_unsafe_rate": sum(1 for r in rows if r.brt_unsafe_any_post_burn) / n,
        }
    return out


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
