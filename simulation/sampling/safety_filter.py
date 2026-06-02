"""Safety filter for impulsive CW maneuvers: BRT line-search (paper) or sampling fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut
from simulation.sampling.passive import is_passively_safe_natural_coast

FilterMode = Literal["linesearch", "sample"]


@dataclass
class FilterResult:
    dv_applied: np.ndarray
    accepted: bool
    brt_value: float
    brt_unsafe: bool
    passive_safe: bool | None
    passive_pre_safe: bool | None
    n_candidates: int
    dv_nominal: np.ndarray
    residual_norm: float
    scale_alpha: float
    time_s: float


def _brt_value_at(brt: Any, x_lvlh_m: np.ndarray, time_s: float) -> float:
    """Query ``V(x, t)``; uses ``value_at_tau`` when available."""
    if hasattr(brt, "value_at_tau"):
        horizon = float(getattr(brt, "horizon_s", time_s))
        tau = float(np.clip(time_s, 0.0, horizon))
        return float(brt.value_at_tau(x_lvlh_m, tau))
    return float(brt.value(x_lvlh_m))


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


def _is_burn_safe(
    plant: CWDynamics,
    x_pre: np.ndarray,
    dv: np.ndarray,
    brt: Any,
    time_s: float,
    *,
    brt_margin: float,
    inner_koz: EllipsoidKeepOut | None,
    passive_horizon_s: float | None,
    passive_n_samples: int,
    check_passive_from_pre: bool,
) -> tuple[bool, float, bool | None, bool | None]:
    """BRT on post-burn state; passive coast from pre-burn ``x_pre`` if requested."""
    x_post = plant.apply_impulsive_dv(x_pre, dv)
    v_brt = _brt_value_at(brt, x_post, time_s)
    brt_ok = v_brt > float(brt_margin) and np.isfinite(v_brt)

    ps_pre: bool | None = None
    if check_passive_from_pre and inner_koz is not None and passive_horizon_s is not None:
        ps_pre = is_passively_safe_natural_coast(
            plant,
            x_pre,
            inner_koz,
            float(passive_horizon_s),
            n_samples=passive_n_samples,
        )
        if not ps_pre:
            return False, v_brt, ps_pre, None

    ps_post: bool | None = None
    if inner_koz is not None and passive_horizon_s is not None and not check_passive_from_pre:
        ps_post = is_passively_safe_natural_coast(
            plant,
            x_post,
            inner_koz,
            float(passive_horizon_s),
            n_samples=passive_n_samples,
        )
        if not ps_post:
            return False, v_brt, ps_pre, ps_post

    return brt_ok, v_brt, ps_pre, ps_post


def filter_impulsive_burn_linesearch(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    dv_nominal: np.ndarray,
    brt: Any,
    time_s: float,
    *,
    brt_margin: float = 0.0,
    inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
    passive_n_samples: int = 128,
    line_search_iters: int = 24,
    check_passive_from_pre: bool = True,
) -> FilterResult:
    """Line-search ``α ∈ [0,1]``: largest ``α·Δv_nom`` with ``V(x⁺, t) > 0`` (and passive from ``x``)."""
    x = np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)
    dv_nom = np.asarray(dv_nominal, dtype=np.float64).reshape(3)
    t_s = float(time_s)

    safe_full, v_full, ps_pre, ps_post = _is_burn_safe(
        plant,
        x,
        dv_nom,
        brt,
        t_s,
        brt_margin=brt_margin,
        inner_koz=inner_koz,
        passive_horizon_s=passive_horizon_s,
        passive_n_samples=passive_n_samples,
        check_passive_from_pre=check_passive_from_pre,
    )
    if safe_full:
        return FilterResult(
            dv_applied=dv_nom.copy(),
            accepted=True,
            brt_value=v_full,
            brt_unsafe=False,
            passive_safe=ps_post if ps_post is not None else ps_pre,
            passive_pre_safe=ps_pre,
            n_candidates=line_search_iters,
            dv_nominal=dv_nom.copy(),
            residual_norm=0.0,
            scale_alpha=1.0,
            time_s=t_s,
        )

    lo, hi = 0.0, 1.0
    best_alpha = 0.0
    best_dv = np.zeros(3, dtype=np.float64)
    best_v = _brt_value_at(brt, x, t_s)
    best_ps_pre = ps_pre
    best_ps_post = ps_post

    for _ in range(int(line_search_iters)):
        mid = 0.5 * (lo + hi)
        dv_mid = mid * dv_nom
        ok, v_mid, ps_pre_m, ps_post_m = _is_burn_safe(
            plant,
            x,
            dv_mid,
            brt,
            t_s,
            brt_margin=brt_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
            passive_n_samples=passive_n_samples,
            check_passive_from_pre=check_passive_from_pre,
        )
        if ok:
            best_alpha = mid
            best_dv = dv_mid.copy()
            best_v = v_mid
            best_ps_pre = ps_pre_m
            best_ps_post = ps_post_m
            lo = mid
        else:
            hi = mid

    zero = np.zeros(3, dtype=np.float64)
    if best_alpha <= 1e-12:
        v0 = _brt_value_at(brt, plant.apply_impulsive_dv(x, zero), t_s)
        return FilterResult(
            dv_applied=zero.copy(),
            accepted=False,
            brt_value=v0,
            brt_unsafe=v0 <= float(brt_margin),
            passive_safe=best_ps_pre,
            passive_pre_safe=best_ps_pre,
            n_candidates=line_search_iters,
            dv_nominal=dv_nom.copy(),
            residual_norm=float(np.linalg.norm(dv_nom)),
            scale_alpha=0.0,
            time_s=t_s,
        )

    return FilterResult(
        dv_applied=best_dv,
        accepted=True,
        brt_value=best_v,
        brt_unsafe=False,
        passive_safe=best_ps_post if best_ps_post is not None else best_ps_pre,
        passive_pre_safe=best_ps_pre,
        n_candidates=line_search_iters,
        dv_nominal=dv_nom.copy(),
        residual_norm=float(np.linalg.norm(best_dv - dv_nom)),
        scale_alpha=float(best_alpha),
        time_s=t_s,
    )


def filter_impulsive_burn_sample(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    dv_nominal: np.ndarray,
    brt: Any,
    time_s: float,
    *,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
    passive_n_samples: int = 128,
    seed: int | None = None,
) -> FilterResult:
    """Legacy sampling search around nominal Δv."""
    x = np.asarray(x_lvlh_m, dtype=np.float64).reshape(6)
    dv_nom = np.asarray(dv_nominal, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(seed)
    t_s = float(time_s)

    cands = _candidate_delta_vs(
        dv_nom, max_perturb_m_s=float(max_perturb_m_s), n_sphere=int(n_sphere_samples), rng=rng
    )

    best: FilterResult | None = None
    for dv in cands:
        ok, v_brt, ps_pre, ps_post = _is_burn_safe(
            plant,
            x,
            dv,
            brt,
            t_s,
            brt_margin=brt_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
            passive_n_samples=passive_n_samples,
            check_passive_from_pre=True,
        )
        if not ok:
            continue
        res = float(np.linalg.norm(dv - dv_nom))
        fr = FilterResult(
            dv_applied=dv.copy(),
            accepted=True,
            brt_value=v_brt,
            brt_unsafe=False,
            passive_safe=ps_post if ps_post is not None else ps_pre,
            passive_pre_safe=ps_pre,
            n_candidates=len(cands),
            dv_nominal=dv_nom.copy(),
            residual_norm=res,
            scale_alpha=float(np.linalg.norm(dv) / max(np.linalg.norm(dv_nom), 1e-12)),
            time_s=t_s,
        )
        if best is None or res < best.residual_norm:
            best = fr

    if best is not None:
        return best

    v0 = _brt_value_at(brt, plant.apply_impulsive_dv(x, dv_nom), t_s)
    ps0: bool | None = None
    if inner_koz is not None and passive_horizon_s is not None:
        ps0 = is_passively_safe_natural_coast(
            plant, x, inner_koz, float(passive_horizon_s), n_samples=passive_n_samples
        )
    return FilterResult(
        dv_applied=dv_nom.copy(),
        accepted=False,
        brt_value=v0,
        brt_unsafe=v0 <= float(brt_margin),
        passive_safe=ps0,
        passive_pre_safe=ps0,
        n_candidates=len(cands),
        dv_nominal=dv_nom.copy(),
        residual_norm=0.0,
        scale_alpha=1.0,
        time_s=t_s,
    )


def default_filter_mode() -> FilterMode:
    mode = os.environ.get("FILTER_MODE", "linesearch").strip().lower()
    if mode in ("sample", "sampling", "sphere"):
        return "sample"
    return "linesearch"


def filter_impulsive_burn(
    plant: CWDynamics,
    x_lvlh_m: np.ndarray,
    dv_nominal: np.ndarray,
    brt: Any,
    time_s: float = 0.0,
    *,
    filter_mode: FilterMode | None = None,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
    passive_n_samples: int = 128,
    seed: int | None = None,
) -> FilterResult:
    mode = filter_mode or default_filter_mode()
    if mode == "sample":
        return filter_impulsive_burn_sample(
            plant,
            x_lvlh_m,
            dv_nominal,
            brt,
            time_s,
            max_perturb_m_s=max_perturb_m_s,
            n_sphere_samples=n_sphere_samples,
            brt_margin=brt_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
            passive_n_samples=passive_n_samples,
            seed=seed,
        )
    return filter_impulsive_burn_linesearch(
        plant,
        x_lvlh_m,
        dv_nominal,
        brt,
        time_s,
        brt_margin=brt_margin,
        inner_koz=inner_koz,
        passive_horizon_s=passive_horizon_s,
        passive_n_samples=passive_n_samples,
    )


def filter_maneuver_plan(
    plant: CWDynamics,
    x0_lvlh_m: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    brt: Any,
    *,
    filter_mode: FilterMode | None = None,
    max_perturb_m_s: float = 0.08,
    n_sphere_samples: int = 48,
    brt_margin: float = 0.0,
    inner_koz: EllipsoidKeepOut | None = None,
    passive_horizon_s: float | None = None,
) -> tuple[list[tuple[float, np.ndarray | None]], list[FilterResult]]:
    """Apply the safety filter at each impulsive burn; ``V(x⁺, t_k)`` uses elapsed plan time."""
    x = np.asarray(x0_lvlh_m, dtype=np.float64).reshape(6).copy()
    t = 0.0
    out_segs: list[tuple[float, np.ndarray | None]] = []
    results: list[FilterResult] = []
    for dt, dv in segments:
        dt = float(dt)
        if dv is None:
            out_segs.append((dt, None))
            x = plant.propagate(x, dt)
            t += dt
            continue
        dv_arr = np.asarray(dv, dtype=np.float64).reshape(3)
        fr = filter_impulsive_burn(
            plant,
            x,
            dv_arr,
            brt,
            t,
            filter_mode=filter_mode,
            max_perturb_m_s=max_perturb_m_s,
            n_sphere_samples=n_sphere_samples,
            brt_margin=brt_margin,
            inner_koz=inner_koz,
            passive_horizon_s=passive_horizon_s,
        )
        results.append(fr)
        out_segs.append((dt, fr.dv_applied.copy()))
        x = plant.step_with_dv(x, dt, fr.dv_applied)
        t += dt
    return out_segs, results
