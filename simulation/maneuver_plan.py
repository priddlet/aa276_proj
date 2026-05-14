"""LLM-style maneuver plans: JSON parsing and burn execution (CW impulsive).

Expected JSON shapes (flexible keys, see :func:`parse_llm_maneuver_json`):

.. code-block:: json

    {
      "maneuvers": [
        {"t_s": 45, "dv_m_s": [-0.02, -0.12, 0.0], "kind": "tangent"},
        {"timestep": 60, "delta_v": [0, 0, 0], "kind": "coast"}
      ]
    }

or a bare list ``[ {...}, ... ]``. All burns are modeled as **impulsive** Δv in **LVLH**
(m/s) at the start of each coast of duration ``t_s`` seconds, matching the pitch's
``{timestep, ΔV}`` JSON and :func:`simulation.cw_dynamics.simulate_impulsive_segments`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParsedBurn:
    """One segment: coast ``duration_s`` then impulsive ``dv_m_s`` (LVLH, m/s)."""

    duration_s: float
    dv_m_s: np.ndarray
    kind: str | None = None


def _vec3(obj: Any) -> np.ndarray:
    a = np.asarray(obj, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError("dv must be length-3")
    return a.copy()


def _one_entry(d: dict[str, Any], idx: int) -> ParsedBurn:
    dt = d.get("t_s", d.get("timestep", d.get("time_s", d.get("dt", d.get("duration_s")))))
    if dt is None:
        raise ValueError(f"segment {idx}: missing duration (t_s / timestep / dt)")
    dv_raw = d.get("dv_m_s", d.get("dv", d.get("delta_v", d.get("DeltaV", d.get("dV")))))
    if dv_raw is None:
        dv_raw = [0.0, 0.0, 0.0]
    kind = d.get("kind", d.get("type", d.get("burn_type")))
    if isinstance(kind, str):
        k = kind.lower()
    else:
        k = None
    return ParsedBurn(float(dt), _vec3(dv_raw), k)


def parse_llm_maneuver_json(text: str) -> list[ParsedBurn]:
    """Parse JSON string from an LLM into a list of :class:`ParsedBurn`."""
    data = json.loads(text)
    if isinstance(data, dict):
        if "maneuvers" in data:
            arr = data["maneuvers"]
        elif "plan" in data:
            arr = data["plan"]
        elif "segments" in data:
            arr = data["segments"]
        else:
            raise ValueError("JSON object must contain 'maneuvers', 'plan', or 'segments'")
    elif isinstance(data, list):
        arr = data
    else:
        raise TypeError("JSON root must be list or object")

    return [_one_entry(dict(x), i) for i, x in enumerate(arr)]


def burns_to_segments(parsed: list[ParsedBurn]) -> tuple[list[tuple[float, np.ndarray | None]], list[str | None]]:
    """Convert to ``simulate_impulsive_segments`` format ``(dt, dv)`` and kind labels."""
    segs: list[tuple[float, np.ndarray | None]] = []
    kinds: list[str | None] = []
    for b in parsed:
        dv = b.dv_m_s if np.any(b.dv_m_s != 0) else None
        segs.append((float(b.duration_s), dv))
        kinds.append(b.kind)
    return segs, kinds
