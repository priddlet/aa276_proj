#!/usr/bin/env python3
"""Relabel boundary corpus plans with the trained BRT (hybrid assess_corpus rules).

Replaces the free-drift KOZ proxy in ``llm_plans_boundary.json`` labels with
learned ``V(x, t_burn) <= 0``, passive pre-burn, and dv_cap — matching
``simulation.benchmark.label_metrics.assess_corpus_intervention``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.brt.config import BRT_HORIZON_S
from simulation.brt.deepreach_mpc_brt import DEEPREACH_MPC_AVAILABLE, KozDeepReachBRT
from simulation.benchmark.label_metrics import (
    _post_burn_states,
    _pre_burn_states,
    assess_corpus_intervention,
)
from simulation.cw_dynamics import CWDynamics
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import load_llm_plans
from simulation.sampling.passive import natural_coast_hits_inner_koz


def _reasons_to_corpus(reasons: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for r in reasons:
        if r == "brt_unsafe":
            out.append("post_V")
        elif r not in out:
            out.append(r)
    return sorted(out)


def _binding_constraint(reasons: list[str]) -> str:
    v_or_passive = any(r in ("post_V", "passive") for r in reasons)
    dv_only = ("dv_cap" in reasons) and not v_or_passive
    if not reasons:
        return "none"
    if v_or_passive and "dv_cap" not in reasons:
        return "v_bound"
    if dv_only:
        return "dv_bound"
    return "v_and_dv"


def build_step_labels(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    brt: KozDeepReachBRT,
    *,
    passive_horizon_s: float,
    brt_margin: float = 0.0,
    passive_n_samples: int = 64,
) -> list[dict]:
    pres = _pre_burn_states(plant, x0, segments)
    posts = _post_burn_states(plant, x0, segments)
    steps: list[dict] = []
    for (t_s, pre), (_, post) in zip(pres, posts):
        dv = post[3:6] - pre[3:6]
        mag = float(np.linalg.norm(dv))
        v_post = float(brt.value_at_tau(post, t_s))
        post_v_le_0 = int(v_post <= float(brt_margin) or not np.isfinite(v_post))
        passive_pre = int(
            natural_coast_hits_inner_koz(
                plant, pre, inner, passive_horizon_s, n_samples=passive_n_samples
            )
        )
        post_passive = natural_coast_hits_inner_koz(
            plant, post, inner, passive_horizon_s, n_samples=passive_n_samples
        )
        dv_exc = int(mag > 0.5 + 1e-9)
        steps.append(
            {
                "t_s": round(float(t_s), 1),
                "dv_mag_m_s": round(mag, 5),
                "pre_burn_state": [round(float(s), 4) for s in pre],
                "post_burn_state": [round(float(s), 4) for s in post],
                "dv_excessive": dv_exc,
                "post_burn_V_le_0": post_v_le_0,
                "passive_preburn_unsafe": passive_pre,
                "requires_intervention": int(dv_exc or post_v_le_0 or passive_pre),
                "V_post": round(v_post, 4),
                "passive_post_unsafe": int(post_passive),
            }
        )
    return steps


def relabel_plan(
    plant: CWDynamics,
    x0: np.ndarray,
    segments: list[tuple[float, np.ndarray | None]],
    inner: EllipsoidKeepOut,
    brt: KozDeepReachBRT,
    *,
    passive_horizon_s: float,
    brt_margin: float = 0.0,
) -> dict:
    steps = build_step_labels(
        plant, x0, segments, inner, brt, passive_horizon_s=passive_horizon_s, brt_margin=brt_margin
    )
    assessment = assess_corpus_intervention(
        plant,
        x0,
        segments,
        inner,
        brt,
        passive_horizon_s=passive_horizon_s,
        brt_margin=brt_margin,
    )
    reasons = _reasons_to_corpus(assessment.reasons)
    total_dv = sum(s["dv_mag_m_s"] for s in steps)
    max_dv = max((s["dv_mag_m_s"] for s in steps), default=0.0)
    return {
        "steps": steps,
        "n_burns": len(steps),
        "first_burn_t_s": steps[0]["t_s"] if steps else None,
        "total_dv_m_s": round(total_dv, 5),
        "max_dv_m_s": round(max_dv, 5),
        "expected_intervention": int(assessment.requires_intervention),
        "intervention_reasons": reasons,
        "binding_constraint": _binding_constraint(reasons),
        "label_mode": "hybrid_brt",
    }


def main() -> None:
    if not DEEPREACH_MPC_AVAILABLE:
        print("DeepReach-MPC required for BRT relabel.", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Relabel boundary corpus with trained BRT.")
    ap.add_argument("--llm-dir", type=str, default=str(ROOT / "llm"))
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(ROOT / "simulation_output" / "deepreach_mpc_koz_v3"),
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--passive-horizon-s", type=float, default=float(BRT_HORIZON_S))
    ap.add_argument("--brt-margin", type=float, default=0.0)
    ap.add_argument(
        "--bundle",
        type=str,
        default="llm_plans_boundary.json",
        help="Bundle filename inside --llm-dir.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    llm_dir = Path(args.llm_dir).resolve()
    bundle_path = llm_dir / args.bundle
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    scenario, plans = load_llm_plans(llm_dir)

    plant = CWDynamics(scenario.mean_motion_rad_s)
    inner = EllipsoidKeepOut(np.array(scenario.semi_axes_m, dtype=np.float64))
    ck_dir = Path(args.checkpoint_dir).resolve()
    print(f"Loading BRT from {ck_dir} …")
    brt = KozDeepReachBRT.load(ck_dir, device=args.device)

    plan_by_id = {p.plan_id: p for p in plans}
    updated = 0
    for rec in data["plans"]:
        pid = str(rec["plan_id"])
        plan = plan_by_id.get(pid)
        if plan is None:
            raise KeyError(f"plan {pid!r} missing from loaded plans")
        x0 = plan.x0(scenario)
        new_label = relabel_plan(
            plant,
            x0,
            plan.segments,
            inner,
            brt,
            passive_horizon_s=float(args.passive_horizon_s),
            brt_margin=float(args.brt_margin),
        )
        rec["label"] = new_label
        updated += 1

    scen = data["scenario"]
    scen.pop("relabel_warning", None)
    scen["label_mode"] = "hybrid_brt"
    scen["label_criteria"]["post_burn_V_le_0"] = "V(x+, t_burn) <= 0 (learned DeepReach-MPC)"
    scen["label_checkpoint"] = str(ck_dir)

    bind = Counter(r["label"]["binding_constraint"] for r in data["plans"])
    reasons = Counter(
        reason
        for r in data["plans"]
        for reason in r["label"]["intervention_reasons"]
    )
    unsafe = sum(int(r["label"]["expected_intervention"]) for r in data["plans"])
    brt_only = sum(
        1
        for r in data["plans"]
        if "post_V" in r["label"]["intervention_reasons"]
        and "dv_cap" not in r["label"]["intervention_reasons"]
        and "passive" not in r["label"]["intervention_reasons"]
    )
    n = len(data["plans"])
    print(f"Relabeled {updated}/{n} plans")
    print(f"  expected_intervention: {unsafe}/{n} ({100*unsafe/n:.0f}%)")
    print(f"  binding: {dict(bind)}")
    print(f"  reasons: {dict(reasons)}")
    print(f"  V-only (post_V, no dv_cap/passive): {brt_only} ({100*brt_only/n:.0f}%)")

    if args.dry_run:
        print("Dry run — bundle not written.")
        return

    bundle_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {bundle_path}")

    summary_csv = llm_dir / "plans_boundary_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "plan_id",
                "category",
                "start_y_m",
                "intent",
                "n_burns",
                "first_burn_t_s",
                "max_dv_m_s",
                "requires_intervention",
                "reasons",
                "binding_constraint",
            ]
        )
        for r in data["plans"]:
            lab = r["label"]
            w.writerow(
                [
                    r["plan_id"],
                    r["tags"]["category"],
                    r["tags"]["start_y_m"],
                    r["tags"]["intent"],
                    lab["n_burns"],
                    lab["first_burn_t_s"],
                    lab["max_dv_m_s"],
                    lab["expected_intervention"],
                    "|".join(lab["intervention_reasons"]),
                    lab["binding_constraint"],
                ]
            )
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
