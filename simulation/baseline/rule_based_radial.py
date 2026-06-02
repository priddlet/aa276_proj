"""Fixed radial + along-track impulsive baseline (non-LLM)."""

from __future__ import annotations

import numpy as np

from simulation.llm_plans import LLMPlan, LLMScenario, absolute_burns_to_segments

RULE_BASED_PLAN_ID = "rule_based_radial_v1"

# Absolute burn times (s) and fixed LVLH Δv (m/s): radial inward + along-track braking.
_DEFAULT_BURNS: list[dict] = [
    {"t_s": 0.0, "dv_m_s": [-0.05, -0.12, 0.0]},
    {"t_s": 60.0, "dv_m_s": [-0.04, -0.10, 0.0]},
    {"t_s": 120.0, "dv_m_s": [-0.03, -0.08, 0.0]},
]


def build_rule_based_radial_plan(
    scenario: LLMScenario,
    *,
    burns: list[dict] | None = None,
) -> LLMPlan:
    """Deterministic three-burn approach from the bundle start state."""
    maneuvers = burns if burns is not None else _DEFAULT_BURNS
    segments = absolute_burns_to_segments(maneuvers)
    return LLMPlan(
        plan_id=RULE_BASED_PLAN_ID,
        prompt="Rule-based fixed radial + along-track burns toward chief.",
        segments=segments,
        tags={
            "category": "rule_based_radial",
            "approach_angle": "radial",
            "urgency": "med",
        },
        expected_intervention=None,
        label=None,
        maneuvers_absolute=list(maneuvers),
    )
