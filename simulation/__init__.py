"""Relative-orbit simulation: CW dynamics and keep-out geometry."""

from simulation.cw_dynamics import (
    CWDynamics,
    cw_mean_motion_leo,
    maneuver_total_duration_s,
    propagate_coast_samples,
    simulate_impulsive_segments,
    simulate_impulsive_segments_dense,
    state_at_maneuver_elapsed_time,
)
from simulation.brt_safety import BRTStepResult, KOZBRTPlaceholder, simulate_plan_with_brt
from simulation.keepout import EllipsoidKeepOut, EllipsoidMaxSeparation, RendezvousSafetyZones
from simulation.maneuver_plan import ParsedBurn, burns_to_segments, parse_llm_maneuver_json
from simulation.passive_safety import is_passively_safe_natural_coast, natural_coast_hits_inner_koz

__all__ = [
    "BRTStepResult",
    "KOZBRTPlaceholder",
    "ParsedBurn",
    "burns_to_segments",
    "parse_llm_maneuver_json",
    "CWDynamics",
    "cw_mean_motion_leo",
    "maneuver_total_duration_s",
    "state_at_maneuver_elapsed_time",
    "EllipsoidKeepOut",
    "EllipsoidMaxSeparation",
    "RendezvousSafetyZones",
    "simulate_impulsive_segments",
    "simulate_impulsive_segments_dense",
    "simulate_plan_with_brt",
    "propagate_coast_samples",
    "is_passively_safe_natural_coast",
    "natural_coast_hits_inner_koz",
]
