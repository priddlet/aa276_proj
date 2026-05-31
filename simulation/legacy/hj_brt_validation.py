"""Sanity checks for 6D KOZ HJ BRT solves and x–y slice evolution semantics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _grid_axes(domain_lo: np.ndarray, domain_hi: np.ndarray, grid_shape: tuple[int, ...]) -> list[np.ndarray]:
    lo = np.asarray(domain_lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(domain_hi, dtype=np.float64).reshape(-1)
    return [
        np.linspace(float(lo[d]), float(hi[d]), int(grid_shape[d]), endpoint=True, dtype=np.float64)
        for d in range(len(grid_shape))
    ]


def validate_brt_backward_evolution(
    result: Any,
    *,
    slice_z_m: float = 0.0,
    slice_vx_m_s: float = 0.0,
    slice_vy_m_s: float = 0.0,
    slice_vz_m_s: float = 0.0,
    deputy_pos_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check that stored HJ slices behave like a backward-growing BRT on the grid.

    Returns a report dict; prints a short summary when ``verbose=True`` via :func:`print_brt_validation_report`.
    """
    times = np.asarray(result.times_s, dtype=np.float64).reshape(-1)
    vals = np.asarray(result.values, dtype=np.float64)
    gshape = tuple(int(x) for x in result.grid_shape)
    lo = np.asarray(result.domain_lo, dtype=np.float64).reshape(6)
    hi = np.asarray(result.domain_hi, dtype=np.float64).reshape(6)
    axes = _grid_axes(lo, hi, gshape)

    def _idx(axis: np.ndarray, target: float) -> int:
        return int(np.argmin(np.abs(axis - float(target))))

    iz = _idx(axes[2], slice_z_m)
    ivx = _idx(axes[3], slice_vx_m_s)
    ivy = _idx(axes[4], slice_vy_m_s)
    ivz = _idx(axes[5], slice_vz_m_s)

    frac_unsafe: list[float] = []
    min_v: list[float] = []
    for k in range(len(times)):
        sl = vals[k, :, :, iz, ivx, ivy, ivz]
        finite = sl[np.isfinite(sl)]
        frac_unsafe.append(float(np.mean(sl <= 0.0)))
        min_v.append(float(np.min(finite)) if finite.size else float("nan"))

    # Backward time should run 0 -> -horizon (non-increasing).
    time_monotone = bool(np.all(np.diff(times) <= 1e-9))
    # On a resolved grid, the discrete {V<=0} set should not shrink as |τ| increases.
    frac_monotone = all(
        frac_unsafe[i] <= frac_unsafe[i + 1] + 1e-12 for i in range(len(frac_unsafe) - 1)
    )
    # More negative min(V) backward in time is consistent with expanding unsafe set.
    min_v_monotone = all(
        min_v[i] >= min_v[i + 1] - 1e-6 or not np.isfinite(min_v[i + 1])
        for i in range(len(min_v) - 1)
    )

    dep_v: list[tuple[float, float]] = []
    if deputy_pos_m is not None:
        r = np.asarray(deputy_pos_m, dtype=np.float64).reshape(3)
        ix, iy = _idx(axes[0], r[0]), _idx(axes[1], r[1])
        for k, tau in enumerate(times):
            v = float(vals[k, ix, iy, iz, ivx, ivy, ivz])
            dep_v.append((float(tau), v))

    report = {
        "times_s": times,
        "time_monotone_backward": time_monotone,
        "frac_V_le_0_monotone": frac_monotone,
        "min_V_monotone": min_v_monotone,
        "frac_V_le_0": frac_unsafe,
        "min_V": min_v,
        "tau_horizon_s": float(times[-1]),
        "frac_at_horizon": float(frac_unsafe[-1]),
        "deputy_V_vs_tau": dep_v,
        "slice_indices": (iz, ivx, ivy, ivz),
    }
    return report


def print_brt_validation_report(report: dict[str, Any]) -> None:
    """Human-readable summary for console after ``solve_koz_collision_brt_6d``."""
    times = np.asarray(report["times_s"], dtype=np.float64)
    fr = report["frac_V_le_0"]
    print("BRT validation (backward HJ semantics):")
    print(f"  τ nodes: {times[0]:.0f} s (terminal / KOZ) → {times[-1]:.0f} s (horizon on grid)")
    print(f"  Time ordering OK: {report['time_monotone_backward']}")
    print(f"  {{V≤0}} fraction monotone in τ: {report['frac_V_le_0_monotone']}  "
          f"({fr[0]*100:.3f}% → {fr[-1]*100:.2f}%)")
    print(f"  min(V) monotone on slice (more negative backward): {report['min_V_monotone']}  "
          f"({report['min_V'][0]:.3g} → {report['min_V'][-1]:.3g})")
    if report.get("deputy_V_vs_tau"):
        t0, v0 = report["deputy_V_vs_tau"][0]
        t1, v1 = report["deputy_V_vs_tau"][-1]
        print(f"  V at deputy grid node: τ={t0:.0f} → {v0:.1f},  τ={t1:.0f} → {v1:.1f}  "
              f"(should decrease backward if approaching BRT)")
    if not report["frac_V_le_0_monotone"]:
        print("  Warning: discrete unsafe fraction decreased between stored τ slices — check horizon/grid.")
    if report["frac_at_horizon"] < 0.01:
        print("  Warning: <1% grid nodes unsafe at horizon — BRT may be under-resolved or horizon too short.")
