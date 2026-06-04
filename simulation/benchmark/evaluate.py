"""Evaluate maneuver plans: no filter, BRT line-search filter, rule-based baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from simulation.benchmark.label_metrics import (
    InterventionAssessment,
    assess_requires_intervention,
    default_outer_corridor,
    with_label_match,
)
from simulation.benchmark.metrics import RolloutMetrics, compute_rollout_metrics
from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut, EllipsoidMaxSeparation
from simulation.llm_plans import LLMPlan, LLMScenario
from simulation.sampling.safety_filter import FilterMode, FilterResult, default_brt_margin, filter_maneuver_plan


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
    mission_success_tier_b: bool
    success_kind: str
    range_closed_m: float
    eval_time_s: float
    range_at_eval_m: float
    mean_dv_overhead_m_s: float
    n_burns_corrected: int
    final_range_m: float
    min_koz_shape_value: float
    brt_unsafe_any_post_burn: bool
    filter_n_accepted: int
    filter_n_burns: int
    llm_unsafe_nominal: bool
    nominal_intervention_reasons: str
    brt_intervened: bool
    n_burns_intervened: int
    n_burns_suppressed: int
    n_burns_scaled: int
    expected_intervention: int | None
    requires_intervention_rollout: bool
    label_match: bool
    intervention_reasons: str
    mission_success_safe: bool

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


def _summarize_filter_interventions(
    filt_results: list[FilterResult],
) -> tuple[int, int, int, bool]:
    """Count burns where BRT changed or suppressed the LLM-commanded Δv."""
    n_intervened = 0
    n_suppressed = 0
    n_scaled = 0
    for fr in filt_results:
        nom = float(np.linalg.norm(fr.dv_nominal))
        app = float(np.linalg.norm(fr.dv_applied))
        if fr.residual_norm <= 1e-9 and abs(fr.scale_alpha - 1.0) <= 1e-6:
            continue
        n_intervened += 1
        if app <= 1e-9 and nom > 1e-9:
            n_suppressed += 1
        elif app > 1e-9:
            n_scaled += 1
    return n_intervened, n_suppressed, n_scaled, n_intervened > 0


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
    brt_margin: float | None = None,
    dv_cap_m_s: float | None = None,
    omit_zero_burns: bool | None = None,
    capture_radius_m: float = 100.0,
    progress_min_m: float = 50.0,
    target_pos_m: np.ndarray | None = None,
    outer_corridor: EllipsoidMaxSeparation | None = None,
) -> PlanEvalResult:
    """Roll out one plan under ``condition``; compute paper-style metrics."""
    nominal = plan.segments
    segs = nominal
    filt_results: list[FilterResult] = []
    n_accepted = 0
    n_filt = 0
    n_intervened = 0
    n_suppressed = 0
    n_scaled = 0
    brt_intervened = False

    tgt = np.zeros(3, dtype=np.float64) if target_pos_m is None else np.asarray(target_pos_m, dtype=np.float64).reshape(3)
    tags = plan.tags
    outer = outer_corridor if outer_corridor is not None else default_outer_corridor()
    eval_margin = 0.0

    nominal_assessment = assess_requires_intervention(
        plant,
        x0,
        nominal,
        inner_koz,
        brt,
        passive_horizon_s=float(passive_horizon_s or 1800.0),
        brt_margin=eval_margin,
        outer=outer,
    )

    if condition == EvalCondition.BRT_FILTER:
        if brt is None:
            raise ValueError("brt required for brt_filter condition")
        filter_margin = default_brt_margin() if brt_margin is None else float(brt_margin)
        segs, filt_results = filter_maneuver_plan(
            plant,
            x0,
            nominal,
            brt,
            filter_mode=filter_mode,
            max_perturb_m_s=max_perturb_m_s,
            n_sphere_samples=n_sphere_samples,
            brt_margin=filter_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
            dv_cap_m_s=dv_cap_m_s,
            omit_zero_burns=omit_zero_burns,
        )
        n_filt = len(filt_results)
        n_accepted = sum(1 for fr in filt_results if fr.accepted)
        n_intervened, n_suppressed, n_scaled, brt_intervened = _summarize_filter_interventions(filt_results)

    rollout = compute_rollout_metrics(
        plant,
        x0,
        segs,
        inner_koz,
        nominal_segments=nominal if condition == EvalCondition.BRT_FILTER else None,
        target_pos_m=target_pos_m,
        capture_radius_m=capture_radius_m,
    )

    eval_margin = 0.0
    unsafe_post = False
    if brt is not None:
        post = _post_burn_states(plant, x0, segs)
        unsafe_post = _brt_unsafe_any_post_burn(brt, post, eval_margin)

    intervention = assess_requires_intervention(
        plant,
        x0,
        segs,
        inner_koz,
        brt,
        passive_horizon_s=float(passive_horizon_s or 1800.0),
        brt_margin=eval_margin,
        outer=outer,
    )
    intervention = with_label_match(intervention, plan.expected_intervention)

    from simulation.benchmark.success_tier_b import TierBSuccessResult, evaluate_tier_b_success

    tier_b: TierBSuccessResult = evaluate_tier_b_success(
        plant,
        x0,
        segs,
        inner_koz,
        category=str(tags.get("category", "")),
        prompt=plan.prompt,
        intercepted_full=rollout.intercepted,
        brt_unsafe_any_post_burn=unsafe_post,
        final_range_m=rollout.final_range_m,
        target_pos_m=tgt,
        capture_radius_m=capture_radius_m,
        progress_min_m=progress_min_m,
    )

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
        mission_success_tier_b=tier_b.success,
        success_kind=tier_b.kind.value,
        range_closed_m=tier_b.range_closed_m,
        eval_time_s=tier_b.eval_time_s,
        range_at_eval_m=tier_b.range_at_eval_m,
        mean_dv_overhead_m_s=rollout.mean_dv_overhead_m_s,
        n_burns_corrected=rollout.n_burns_corrected,
        final_range_m=rollout.final_range_m,
        min_koz_shape_value=rollout.min_koz_shape_value,
        brt_unsafe_any_post_burn=unsafe_post,
        filter_n_accepted=n_accepted,
        filter_n_burns=n_filt,
        llm_unsafe_nominal=nominal_assessment.requires_intervention,
        nominal_intervention_reasons=",".join(nominal_assessment.reasons),
        brt_intervened=brt_intervened,
        n_burns_intervened=n_intervened,
        n_burns_suppressed=n_suppressed,
        n_burns_scaled=n_scaled,
        expected_intervention=plan.expected_intervention,
        requires_intervention_rollout=intervention.requires_intervention,
        label_match=intervention.label_match,
        intervention_reasons=",".join(intervention.reasons),
        mission_success_safe=not intervention.requires_intervention,
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
    brt_margin: float | None = None,
    dv_cap_m_s: float | None = None,
    omit_zero_burns: bool | None = None,
    capture_radius_m: float = 100.0,
    progress_min_m: float = 50.0,
    use_outer_corridor: bool = True,
) -> list[PlanEvalResult]:
    x0 = scenario.start_state_lvlh_m
    tgt = np.zeros(3, dtype=np.float64)
    outer = default_outer_corridor() if use_outer_corridor else None
    cap = scenario.dv_cap_m_s if dv_cap_m_s is None else dv_cap_m_s
    results: list[PlanEvalResult] = []
    for plan in plans:
        if EvalCondition.NO_FILTER in conditions:
            results.append(
                evaluate_plan(
                    plant, plan, x0, brt, EvalCondition.NO_FILTER, inner_koz,
                    passive_horizon_s=passive_horizon_s, filter_mode=filter_mode,
                    max_perturb_m_s=max_perturb_m_s, n_sphere_samples=n_sphere_samples,
                    brt_margin=brt_margin, capture_radius_m=capture_radius_m, target_pos_m=tgt,
                    progress_min_m=progress_min_m,
                    outer_corridor=outer, dv_cap_m_s=cap, omit_zero_burns=omit_zero_burns,
                )
            )
        if EvalCondition.BRT_FILTER in conditions:
            results.append(
                evaluate_plan(
                    plant, plan, x0, brt, EvalCondition.BRT_FILTER, inner_koz,
                    passive_horizon_s=passive_horizon_s, filter_mode=filter_mode,
                    max_perturb_m_s=max_perturb_m_s, n_sphere_samples=n_sphere_samples,
                    brt_margin=brt_margin, capture_radius_m=capture_radius_m, target_pos_m=tgt,
                    progress_min_m=progress_min_m,
                    outer_corridor=outer, dv_cap_m_s=cap, omit_zero_burns=omit_zero_burns,
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
                progress_min_m=progress_min_m,
                outer_corridor=outer,
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
    """Aggregate rates by condition and by category (Tier-B success)."""
    out: dict[str, Any] = {"by_condition": {}, "by_category": {}, "n_rows": len(results)}

    def _cond_stats(rows: list[PlanEvalResult]) -> dict[str, Any]:
        n = len(rows)
        labeled = [r for r in rows if r.expected_intervention is not None]
        nl = len(labeled) or 1
        stats = {
            "n_plans": n,
            "llm_unsafe_rate": sum(1 for r in rows if r.llm_unsafe_nominal) / n,
            "interception_rate": sum(1 for r in rows if r.intercepted) / n,
            "mission_success_rate": sum(1 for r in rows if r.mission_success) / n,
            "mission_success_tier_b_rate": sum(1 for r in rows if r.mission_success_tier_b) / n,
            "post_filter_unsafe_rate": sum(1 for r in rows if r.requires_intervention_rollout) / n,
            "mission_success_safe_rate": sum(1 for r in rows if r.mission_success_safe) / n,
            "label_match_rate": sum(1 for r in labeled if r.label_match) / nl,
            "requires_intervention_rate": sum(1 for r in rows if r.requires_intervention_rollout) / n,
            "mean_dv_overhead_m_s": float(np.mean([r.mean_dv_overhead_m_s for r in rows])),
            "brt_unsafe_rate": sum(1 for r in rows if r.brt_unsafe_any_post_burn) / n,
            "mean_range_closed_m": float(np.mean([r.range_closed_m for r in rows])),
        }
        if any(r.condition == EvalCondition.BRT_FILTER.value for r in rows):
            brt_rows = [r for r in rows if r.condition == EvalCondition.BRT_FILTER.value]
            nb = len(brt_rows) or 1
            stats["brt_intervention_rate"] = sum(1 for r in brt_rows if r.brt_intervened) / nb
            stats["mean_burns_intervened_per_plan"] = float(np.mean([r.n_burns_intervened for r in brt_rows]))
            stats["mean_burns_suppressed_per_plan"] = float(np.mean([r.n_burns_suppressed for r in brt_rows]))
            stats["mean_burns_scaled_per_plan"] = float(np.mean([r.n_burns_scaled for r in brt_rows]))
        return stats

    for cond in sorted({r.condition for r in results}):
        rows = [r for r in results if r.condition == cond]
        if rows:
            out["by_condition"][cond] = _cond_stats(rows)

    for cat in sorted({r.category for r in results if r.category}):
        base = [r for r in results if r.category == cat and r.condition == EvalCondition.NO_FILTER.value]
        rows = base if base else [r for r in results if r.category == cat]
        if not rows:
            continue
        kinds = sorted({r.success_kind for r in rows})
        out["by_category"][cat] = {
            "success_kind": kinds[0] if len(kinds) == 1 else kinds,
            **_cond_stats(rows),
        }
    return out


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
