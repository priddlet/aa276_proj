"""BRT-style safety interface for CW deputy states (LVLH, SI).

The AA276 pitch evaluates a value function ``V(x)`` on the deputy CW state. This
package provides:

- :class:`KOZBRTPlaceholder` — fast geometry surrogate (inner KOZ + outer corridor).
- :func:`simulate_plan_with_brt` — impulsive rollout with optional **passive** drift
  check (Option 2) via ``passive_inner_koz`` / ``passive_horizon_s``.

HJ collision BRT (Option 1) lives in :mod:`simulation.hj_koz_brt` (:class:`KozHJTable6D`);
there ``value(x) <= 0`` means inside the backward reachable tube. :class:`KOZBRTPlaceholder`
uses the opposite sign convention (positive ``value`` means KOZ/corridor violation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from simulation.cw_dynamics import CWDynamics
    from simulation.keepout import EllipsoidKeepOut, RendezvousSafetyZones


class BRTValueFunction(Protocol):
    """Contract for a BRT or surrogate safety field on CW state ``x`` (6,) SI."""

    def value(self, x_lvlh_m: np.ndarray) -> float:
        """Scalar signed distance / value; use :meth:`is_unsafe` for the verdict."""

    def is_unsafe(self, x_lvlh_m: np.ndarray) -> bool:
        ...


@dataclass
class BRTStepResult:
    step_index: int
    time_s: float
    state_lvlh_m: np.ndarray
    brt_value: float
    unsafe: bool
    burn_kind: str | None
    passive_safe: bool | None = None


class KOZBRTPlaceholder:
    """KOZ / corridor surrogate for ``V(x)`` until hj-reach BRT is available."""

    def __init__(self, zones: RendezvousSafetyZones) -> None:
        self._zones = zones

    def value(self, x_lvlh_m: np.ndarray) -> float:
        x = np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)
        if self._zones.is_unsafe_state(x):
            r = x[:3]
            if self._zones.inner.is_inside(r):
                return float(self._zones.inner.shape_value(r))
            return float(max(0.0, self._zones.outer.shape_value(r) - 1.0) + 1.0)
        return -1.0

    def is_unsafe(self, x_lvlh_m: np.ndarray) -> bool:
        return self._zones.is_unsafe_state(np.asarray(x_lvlh_m, dtype=np.float64).reshape(6))


def simulate_plan_with_brt(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    brt: Any,
    *,
    burn_kinds: list[str | None] | None = None,
    passive_inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
    passive_n_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[BRTStepResult]]:
    """Impulsive rollout + BRT classification at each segment **end** (LVLH SI).

    If ``passive_inner_koz`` and ``passive_horizon_s`` are set (Option 2), each node
    also records whether **natural coast** (no further thrust) stays outside the
    inner KOZ over that horizon.
    """
    from simulation.cw_dynamics import simulate_impulsive_segments
    from simulation.passive_safety import is_passively_safe_natural_coast

    times, states = simulate_impulsive_segments(plant, x0_lvlh_m, segments)
    kinds = burn_kinds or [None] * len(segments)
    if len(kinds) != len(segments):
        raise ValueError("burn_kinds length must match segments")

    do_passive = passive_inner_koz is not None and passive_horizon_s is not None
    if passive_inner_koz is not None and passive_horizon_s is None:
        raise ValueError("passive_horizon_s is required when passive_inner_koz is set")

    results: list[BRTStepResult] = []
    for k in range(len(times)):
        xk = states[k]
        v = float(brt.value(xk))
        bk = None if k == 0 else kinds[k - 1]
        ps: bool | None = None
        if do_passive:
            ps = is_passively_safe_natural_coast(
                plant,
                xk,
                passive_inner_koz,  # type: ignore[arg-type]
                float(passive_horizon_s),
                n_samples=int(passive_n_samples),
            )
        results.append(
            BRTStepResult(
                step_index=k,
                time_s=float(times[k]),
                state_lvlh_m=xk.copy(),
                brt_value=v,
                unsafe=bool(brt.is_unsafe(xk)),
                burn_kind=bk,
                passive_safe=ps,
            )
        )
    return times, states, results
