"""Fixed radial + along-track impulsive reference trajectories (non-LLM)."""

from __future__ import annotations

import numpy as np

from simulation.llm_plans import LLMPlan, LLMScenario, absolute_burns_to_segments

RULE_BASED_PLAN_ID = "rule_based_radial_v1"


def _burn(t_s: float, dv: tuple[float, float, float]) -> dict:
    return {"t_s": float(t_s), "dv_m_s": [float(dv[0]), float(dv[1]), float(dv[2])]}


def _min_coast(scenario: LLMScenario) -> float:
    raw = scenario.raw.get("min_coast_before_first_burn_s", 300.0)
    return float(raw if raw is not None else 300.0)


def build_rule_based_radial_plan(
    scenario: LLMScenario,
    *,
    burns: list[dict] | None = None,
    plan_id: str = RULE_BASED_PLAN_ID,
    prompt: str = "Rule-based fixed radial + along-track burns toward chief.",
) -> LLMPlan:
    """Single deterministic burn schedule from the bundle start state."""
    maneuvers = burns if burns is not None else _default_burns_legacy()
    segments = absolute_burns_to_segments(maneuvers)
    return LLMPlan(
        plan_id=plan_id,
        prompt=prompt,
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


def _default_burns_legacy() -> list[dict]:
    """Original schedule (t=0 burns); kept for backwards compatibility."""
    return [
        _burn(0.0, (-0.05, -0.12, 0.0)),
        _burn(60.0, (-0.04, -0.10, 0.0)),
        _burn(120.0, (-0.03, -0.08, 0.0)),
    ]


def build_rule_based_radial_variants(scenario: LLMScenario) -> list[LLMPlan]:
    """Several hand-tuned radial braking schedules (reference trajectories, not LLM plans).

    First burn is always after the scenario min-coast (300 s for the LEO bundle).
    All Δv point radially inward (-x) and along-track toward the chief (-y).
    """
    t0 = _min_coast(scenario) + 20.0
    specs: list[tuple[str, str, list[dict]]] = [
        (
            "rule_ref_gentle_3burn",
            "Three small radial trims after coast — cautious reference glide.",
            [
                _burn(t0, (-0.04, -0.06, 0.0)),
                _burn(t0 + 180, (-0.03, -0.05, 0.0)),
                _burn(t0 + 360, (-0.02, -0.04, 0.0)),
            ],
        ),
        (
            "rule_ref_medium_3burn",
            "Three moderate braking burns spaced ~3 min apart.",
            [
                _burn(t0, (-0.06, -0.10, 0.0)),
                _burn(t0 + 160, (-0.05, -0.09, 0.0)),
                _burn(t0 + 320, (-0.04, -0.07, 0.0)),
            ],
        ),
        (
            "rule_ref_strong_2burn",
            "Two stronger radial + along-track pulses.",
            [
                _burn(t0 + 30, (-0.08, -0.14, 0.0)),
                _burn(t0 + 240, (-0.06, -0.12, 0.0)),
            ],
        ),
        (
            "rule_ref_single_trim",
            "Single post-coast trim toward the chief.",
            [_burn(t0 + 80, (-0.05, -0.11, 0.0))],
        ),
        (
            "rule_ref_spaced_4burn",
            "Four light burns on a glideslope-style schedule.",
            [
                _burn(t0, (-0.03, -0.05, 0.0)),
                _burn(t0 + 120, (-0.03, -0.06, 0.0)),
                _burn(t0 + 240, (-0.04, -0.07, 0.0)),
                _burn(t0 + 360, (-0.04, -0.08, 0.0)),
            ],
        ),
        (
            "rule_ref_radial_heavy",
            "Radial-heavy braking with smaller along-track component.",
            [
                _burn(t0 + 40, (-0.10, -0.06, 0.0)),
                _burn(t0 + 220, (-0.08, -0.05, 0.0)),
            ],
        ),
        (
            "rule_ref_along_heavy",
            "Along-track-heavy braking with smaller radial component.",
            [
                _burn(t0 + 40, (-0.04, -0.15, 0.0)),
                _burn(t0 + 220, (-0.03, -0.12, 0.0)),
            ],
        ),
        (
            "rule_ref_late_commit",
            "Long coast then two closing burns later in the horizon.",
            [
                _burn(t0 + 200, (-0.07, -0.13, 0.0)),
                _burn(t0 + 420, (-0.05, -0.10, 0.0)),
            ],
        ),
    ]
    out: list[LLMPlan] = []
    ref_y = scenario.reference_start_y_m
    ref_x0 = np.array([0.0, ref_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for plan_id, prompt, maneuvers in specs:
        out.append(
            build_rule_based_radial_plan(
                scenario,
                burns=maneuvers,
                plan_id=plan_id,
                prompt=prompt,
            )
        )
        out[-1].start_state_lvlh_m = ref_x0.copy()
    return out
