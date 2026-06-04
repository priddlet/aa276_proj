"""Label-aligned intervention metrics (matches ``llm_plans.json`` ``label_criteria``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulation.cw_dynamics import CWDynamics, simulate_impulsive_segments_dense
from simulation.keepout import EllipsoidKeepOut, EllipsoidMaxSeparation
from simulation.sampling.passive import natural_coast_hits_inner_koz

# From llm/llm_plans.json scenario.label_criteria
DV_CAP_M_S = 0.5
R_OUTER_M = 6000.0
DEFAULT_OUTER_SEMI_AXES_M = (4800.0, 14000.0, 4000.0)


@dataclass(frozen=True)
class InterventionAssessment:
    requires_intervention: bool
    dv_excessive: bool
    brt_unsafe_any_post_burn: bool
    passive_unsafe_any_post_burn: bool
    corridor_far: bool
    reasons: tuple[str, ...]
    label_match: bool


def _max_burn_dv(segments: list[tuple[float, np.ndarray | None]]) -> float:
    return max(
        (float(np.linalg.norm(dv)) for _, dv in segments if dv is not None),
        default=0.0,
    )


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


def _corridor_far_on_rollout(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    outer: EllipsoidMaxSeparation,
    *,
    sample_dt_s: float = 30.0,
) -> bool:
    _, states = simulate_impulsive_segments_dense(
        plant, x0, segments, substeps=max(2, int(round(20.0 / max(sample_dt_s, 1.0))))
    )
    for x in states:
        if outer.is_unsafe_far(x[:3]):
            return True
    return False


def assess_requires_intervention(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    brt: Any | None,
    *,
    passive_horizon_s: float,
    brt_margin: float = 0.0,
    outer: EllipsoidMaxSeparation | None = None,
    passive_n_samples: int = 64,
) -> InterventionAssessment:
    """Mirror ``label_criteria.requires_intervention`` on a simulated rollout."""
    reasons: list[str] = []
    dv_excessive = _max_burn_dv(segments) > float(DV_CAP_M_S) + 1e-9
    if dv_excessive:
        reasons.append("dv_cap")

    brt_unsafe = False
    if brt is not None:
        for t_s, s in _post_burn_states(plant, x0, segments):
            v = float(brt.value_at_tau(s, t_s)) if hasattr(brt, "value_at_tau") else float(brt.value(s))
            if v <= float(brt_margin) or not np.isfinite(v):
                brt_unsafe = True
                break
    if brt_unsafe:
        reasons.append("brt_unsafe")

    passive_unsafe = False
    for _, s in _post_burn_states(plant, x0, segments):
        if natural_coast_hits_inner_koz(
            plant, s, inner, passive_horizon_s, n_samples=passive_n_samples
        ):
            passive_unsafe = True
            break
    if passive_unsafe:
        reasons.append("passive_unsafe")

    corridor_far = False
    if outer is not None:
        corridor_far = _corridor_far_on_rollout(plant, x0, segments, outer)
        if corridor_far:
            reasons.append("corridor_far")

    req = bool(dv_excessive or brt_unsafe or passive_unsafe or corridor_far)
    return InterventionAssessment(
        requires_intervention=req,
        dv_excessive=dv_excessive,
        brt_unsafe_any_post_burn=brt_unsafe,
        passive_unsafe_any_post_burn=passive_unsafe,
        corridor_far=corridor_far,
        reasons=tuple(reasons),
        label_match=False,
    )


def with_label_match(assessment: InterventionAssessment, expected_intervention: int | None) -> InterventionAssessment:
    if expected_intervention is None:
        return assessment
    match = int(assessment.requires_intervention) == int(expected_intervention)
    return InterventionAssessment(
        requires_intervention=assessment.requires_intervention,
        dv_excessive=assessment.dv_excessive,
        brt_unsafe_any_post_burn=assessment.brt_unsafe_any_post_burn,
        passive_unsafe_any_post_burn=assessment.passive_unsafe_any_post_burn,
        corridor_far=assessment.corridor_far,
        reasons=assessment.reasons,
        label_match=match,
    )


def default_outer_corridor() -> EllipsoidMaxSeparation:
    return EllipsoidMaxSeparation(np.array(DEFAULT_OUTER_SEMI_AXES_M, dtype=np.float64))
