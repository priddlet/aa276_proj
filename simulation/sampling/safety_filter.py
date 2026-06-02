"""Sampling-based safety filter for impulsive CW maneuvers (non-QP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut
from simulation.sampling.passive import is_passively_safe_natural_coast


@dataclass
class FilterResult:
    dv_applied: np.ndarray
    accepted: bool
    brt_value: float
    brt_unsafe: bool
    passive_safe: bool | None
    n_candidates: int
    dv_nominal: np.ndarray
    residual_norm: float


def _candidate_delta_vs(
    dv_nom: np.ndarray,
    *,
    max_perturb_m_s: float,
    n_sphere: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    dv_nom = np.asarray(dv_nom, dtype=np.float64).reshape(3)
    cands: list[np.ndarray] = [dv_nom.copy()]
    axes = np.eye(3, dtype=np.float64)
    for j in range(3):
        for sign in (-1.0, 1.0):
            cands.append(dv_nom + sign * max_perturb_m_s * axes[j])
    for _ in range(int(n_sphere)):
        d = rng.normal(size=3)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            continue
        scale = rng.uniform(0.0, max_perturb_m_s) / nrm
        cands.append(dv_nom + scale * d)
    return cands


def filter_impulsive_burn(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    dv_nominal: np.ndarray,
    brt: Any,
    *,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    passive_inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
    passive_n_samples: int = 128,
    seed: int | None = None,
) -> FilterResult:
    """Search perturbed Δv around nominal; pick closest safe candidate under the BRT field.

    Safe: ``V(x⁺) > brt_margin`` and optional passive coast stays outside inner KOZ.
    """
    x = np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)
    dv_nom = np.asarray(dv_nominal, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(seed)

    cands = _candidate_delta_vs(
        dv_nom, max_perturb_m_s=float(max_perturb_m_s), n_sphere=int(n_sphere_samples), rng=rng
    )

    best: FilterResult | None = None
    for dv in cands:
        x_post = plant.apply_impulsive_dv(x, dv)
        v_brt = float(brt.value(x_post))
        unsafe = v_brt <= float(brt_margin) or not np.isfinite(v_brt)
        ps: bool | None = None
        if passive_inner_koz is not None and passive_horizon_s is not None:
            ps = is_passively_safe_natural_coast(
                plant,
                x_post,
                passive_inner_koz,
                float(passive_horizon_s),
                n_samples=passive_n_samples,
            )
            if not ps:
                continue
        if unsafe:
            continue
        res = float(np.linalg.norm(dv - dv_nom))
        fr = FilterResult(
            dv_applied=dv.copy(),
            accepted=True,
            brt_value=v_brt,
            brt_unsafe=False,
            passive_safe=ps,
            n_candidates=len(cands),
            dv_nominal=dv_nom.copy(),
            residual_norm=res,
        )
        if best is None or res < best.residual_norm:
            best = fr

    if best is not None:
        return best

    x_post_nom = plant.apply_impulsive_dv(x, dv_nom)
    v0 = float(brt.value(x_post_nom))
    ps0: bool | None = None
    if passive_inner_koz is not None and passive_horizon_s is not None:
        ps0 = is_passively_safe_natural_coast(
            plant, x_post_nom, passive_inner_koz, float(passive_horizon_s), n_samples=passive_n_samples
        )
    return FilterResult(
        dv_applied=dv_nom.copy(),
        accepted=False,
        brt_value=v0,
        brt_unsafe=v0 <= float(brt_margin),
        passive_safe=ps0,
        n_candidates=len(cands),
        dv_nominal=dv_nom.copy(),
        residual_norm=0.0,
    )


def filter_maneuver_plan(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    brt: Any,
    *,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    passive_inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
) -> tuple[list[tuple[float, np.ndarray | None]], list[FilterResult]]:
    """Apply :func:`filter_impulsive_burn` at each segment with non-zero Δv."""
    x = np.asarray(x0_lvlh_m, dtype=np.float64).reshape(6).copy()
    out_segs: list[tuple[float, np.ndarray | None]] = []
    results: list[FilterResult] = []
    for dt, dv in segments:
        dt = float(dt)
        if dv is None:
            out_segs.append((dt, None))
            x = plant.propagate(x, dt)
            continue
        dv_arr = np.asarray(dv, dtype=np.float64).reshape(3)
        fr = filter_impulsive_burn(
            plant,
            x,
            dv_arr,
            brt,
            max_perturb_m_s=max_perturb_m_s,
            n_sphere_samples=n_sphere_samples,
            brt_margin=brt_margin,
            passive_inner_koz=passive_inner_koz,
            passive_horizon_s=passive_horizon_s,
        )
        results.append(fr)
        out_segs.append((dt, fr.dv_applied.copy()))
        x = plant.step_with_dv(x, dt, fr.dv_applied)
    return out_segs, results
