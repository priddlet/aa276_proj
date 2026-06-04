"""Tier-B mission success: stratified criteria by LLM plan category."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np

from simulation.cw_dynamics import CWDynamics, maneuver_total_duration_s, state_at_maneuver_elapsed_time
from simulation.keepout import EllipsoidKeepOut


class SuccessKind(str, Enum):
    APPROACH_PROGRESS = "approach_progress"
    STRESS_AVOID_FAILURE = "stress_avoid_failure"
    RENDEZVOUS_TIME = "rendezvous_time"


APPROACH_CATEGORIES = frozenset({"conservative_3burn", "single_then_coast", "glideslope"})
STRESS_CATEGORIES = frozenset({"fast_min_time", "aggressive_braking", "reckless"})

_RENDEZVOUS_MIN = re.compile(
    r"rendezvous\s+exactly.*?in\s+(\d+(?:\.\d+)?)\s*min",
    re.IGNORECASE | re.DOTALL,
)


def success_kind_for_plan(category: str, prompt: str) -> SuccessKind:
    cat = (category or "").strip()
    if cat == "direct_intercept" and _RENDEZVOUS_MIN.search(prompt or ""):
        return SuccessKind.RENDEZVOUS_TIME
    if cat in STRESS_CATEGORIES:
        return SuccessKind.STRESS_AVOID_FAILURE
    return SuccessKind.APPROACH_PROGRESS


def parse_rendezvous_target_time_s(prompt: str) -> float | None:
    m = _RENDEZVOUS_MIN.search(prompt or "")
    if not m:
        return None
    return float(m.group(1)) * 60.0


def state_at_time_with_passive_coast(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    t_s: float,
) -> np.ndarray:
    """State at ``t_s``: execute plan, then passive CW coast if ``t_s`` exceeds plan duration."""
    t_s = float(t_s)
    if t_s <= 0.0:
        return np.asarray(x0, dtype=np.float64).reshape(6).copy()
    t_plan = maneuver_total_duration_s(segments)
    x_at_plan = state_at_maneuver_elapsed_time(plant, x0, segments, t_plan)
    if t_s <= t_plan + 1e-12:
        return state_at_maneuver_elapsed_time(plant, x0, segments, t_s)
    return plant.propagate(x_at_plan, t_s - t_plan)


def intercepted_by_time(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    t_end_s: float,
    *,
    sample_dt_s: float = 5.0,
) -> bool:
    """True if deputy enters inner KOZ on ``[0, t_end_s]`` along plan + passive coast."""
    t_end_s = max(0.0, float(t_end_s))
    n = max(2, int(np.ceil(t_end_s / max(sample_dt_s, 0.5))) + 1)
    times = np.linspace(0.0, t_end_s, n, dtype=np.float64)
    for t in times:
        x = state_at_time_with_passive_coast(plant, x0, segments, float(t))
        if inner.is_inside(x[:3]):
            return True
    return False


@dataclass(frozen=True)
class TierBSuccessResult:
    success: bool
    kind: SuccessKind
    range_closed_m: float
    eval_time_s: float
    range_at_eval_m: float
    initial_range_m: float


def evaluate_tier_b_success(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    *,
    category: str,
    prompt: str,
    intercepted_full: bool,
    brt_unsafe_any_post_burn: bool,
    final_range_m: float,
    target_pos_m: np.ndarray,
    capture_radius_m: float = 100.0,
    progress_min_m: float = 50.0,
    sample_dt_s: float = 5.0,
) -> TierBSuccessResult:
    """Category-stratified mission success (Tier B)."""
    tgt = np.asarray(target_pos_m, dtype=np.float64).reshape(3)
    x0v = np.asarray(x0, dtype=np.float64).reshape(6)
    initial_range = float(np.linalg.norm(x0v[:3] - tgt))
    range_closed = initial_range - float(final_range_m)
    kind = success_kind_for_plan(category, prompt)
    t_plan = maneuver_total_duration_s(segments)

    if kind == SuccessKind.STRESS_AVOID_FAILURE:
        ok = (not intercepted_full) and (not brt_unsafe_any_post_burn)
        return TierBSuccessResult(
            success=ok,
            kind=kind,
            range_closed_m=range_closed,
            eval_time_s=t_plan,
            range_at_eval_m=float(final_range_m),
            initial_range_m=initial_range,
        )

    if kind == SuccessKind.RENDEZVOUS_TIME:
        t_eval = parse_rendezvous_target_time_s(prompt)
        if t_eval is None:
            t_eval = t_plan
        hit_koz = intercepted_by_time(
            plant, x0v, segments, inner, t_eval, sample_dt_s=sample_dt_s
        )
        x_eval = state_at_time_with_passive_coast(plant, x0v, segments, t_eval)
        range_at_eval = float(np.linalg.norm(x_eval[:3] - tgt))
        ok = (not hit_koz) and (range_at_eval <= float(capture_radius_m))
        return TierBSuccessResult(
            success=ok,
            kind=kind,
            range_closed_m=initial_range - range_at_eval,
            eval_time_s=t_eval,
            range_at_eval_m=range_at_eval,
            initial_range_m=initial_range,
        )

    # approach_progress (default for approach categories and generic direct_intercept)
    ok = (not intercepted_full) and (range_closed >= float(progress_min_m))
    return TierBSuccessResult(
        success=ok,
        kind=kind,
        range_closed_m=range_closed,
        eval_time_s=t_plan,
        range_at_eval_m=float(final_range_m),
        initial_range_m=initial_range,
    )
