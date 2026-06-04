"""Run paper-style benchmark: no filter, BRT line-search filter, rule-based radial.

Example (after BRT training)::

    # Original 3.2 km LLM bundle (labels; filter rarely changes plans):
    python -m simulation.benchmark \\
        --checkpoint-dir simulation_output/deepreach_mpc_koz_v3 \\
        --conditions no_filter,brt_filter,rule_based

    # Filter demonstration bundle (y=1200 m, timed near-boundary burns):
    python -m simulation.benchmark.generate_filter_demo_plans
    LLM_PLANS_DIR=llm/llm_plans_brt_demo python -m simulation.benchmark \\
        --conditions no_filter,brt_filter
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from simulation.baseline.rule_based_radial import build_rule_based_radial_plan
from simulation.brt.config import BRT_HORIZON_S
from simulation.brt.deepreach_mpc_brt import (
    DEEPREACH_MPC_AVAILABLE,
    DEEPREACH_MPC_IMPORT_ERROR,
    KozDeepReachBRT,
    default_checkpoint_dir,
)
from simulation.benchmark.evaluate import (
    EvalCondition,
    run_llm_benchmark,
    summarize_results,
    write_results_csv,
    write_summary_json,
)
from simulation.cw_dynamics import CWDynamics, leo_circular_orbit
from simulation.keepout import EllipsoidKeepOut
from simulation.llm_plans import default_llm_dir, load_llm_plans
from simulation.sampling.safety_filter import default_brt_margin, default_filter_mode


def _parse_conditions(s: str) -> tuple[EvalCondition, ...]:
    s = s.strip().lower()
    if s in ("all", "paper", "default"):
        return (
            EvalCondition.NO_FILTER,
            EvalCondition.BRT_FILTER,
            EvalCondition.RULE_BASED,
        )
    out: list[EvalCondition] = []
    for part in s.split(","):
        p = part.strip()
        if p in ("no_filter", "raw", "unfiltered", "1"):
            out.append(EvalCondition.NO_FILTER)
        elif p in ("brt_filter", "filter", "brt", "2"):
            out.append(EvalCondition.BRT_FILTER)
        elif p in ("rule_based", "rule", "baseline", "3"):
            out.append(EvalCondition.RULE_BASED)
        else:
            raise ValueError(f"unknown condition {p!r}")
    return tuple(dict.fromkeys(out))


def main() -> None:
    if not DEEPREACH_MPC_AVAILABLE:
        print(
            "DeepReach-MPC required:\n  pip install -r requirements-deepreach.txt\n"
            f"  ({DEEPREACH_MPC_IMPORT_ERROR})",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Paper-style LLM + BRT + rule-based benchmark.")
    p.add_argument("--llm-dir", type=str, default=str(default_llm_dir(root)))
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("DEEPREACH_CHECKPOINT_DIR", str(default_checkpoint_dir(root))),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(root / "simulation_output" / "llm_benchmark_results.csv"),
    )
    p.add_argument(
        "--summary-json",
        type=str,
        default=str(root / "simulation_output" / "llm_benchmark_summary.json"),
    )
    p.add_argument("--device", type=str, default=os.environ.get("DEEPREACH_DEVICE", "auto"))
    p.add_argument("--altitude-km", type=float, default=float(os.environ.get("LEO_ALTITUDE_KM", "400")))
    p.add_argument(
        "--conditions",
        type=str,
        default=os.environ.get("LLM_BENCHMARK_CONDITIONS", "no_filter,brt_filter,rule_based"),
    )
    p.add_argument("--passive-horizon-s", type=float, default=None)
    p.add_argument("--capture-radius-m", type=float, default=float(os.environ.get("CAPTURE_RADIUS_M", "100")))
    p.add_argument(
        "--progress-min-m",
        type=float,
        default=float(os.environ.get("BENCHMARK_PROGRESS_MIN_M", "50")),
        help="Min range closed (m) for approach-category Tier-B success.",
    )
    p.add_argument(
        "--filter-mode",
        type=str,
        default=default_filter_mode(),
        choices=("linesearch", "sample", "hybrid"),
    )
    p.add_argument("--filter-max-perturb", type=float, default=float(os.environ.get("FILTER_MAX_PERTURB_M_S", "0.2")))
    p.add_argument("--filter-n-sphere", type=int, default=int(os.environ.get("FILTER_N_SPHERE", "160")))
    p.add_argument("--brt-margin", type=float, default=default_brt_margin())
    p.add_argument("--plan-id", type=str, default="")
    args = p.parse_args()

    ck_dir = Path(args.checkpoint_dir).resolve()
    from simulation.brt.training_metrics import resolve_inference_checkpoint

    try:
        ckpt, _ep, reason = resolve_inference_checkpoint(ck_dir)
    except FileNotFoundError as exc:
        print(f"{exc}. Train: python -m simulation.brt.train --force", file=sys.stderr)
        sys.exit(1)
    print(f"Using checkpoint {ckpt.name} ({reason})")

    scenario, plans = load_llm_plans(args.llm_dir)
    if args.plan_id.strip():
        plans = [pl for pl in plans if pl.plan_id == args.plan_id.strip()]
        if not plans:
            print(f"plan_id not found: {args.plan_id}", file=sys.stderr)
            sys.exit(1)

    leo = leo_circular_orbit(args.altitude_km)
    plant = CWDynamics(scenario.mean_motion_rad_s if scenario.mean_motion_rad_s > 0 else leo.n_rad_s)
    inner = EllipsoidKeepOut(np.array(scenario.semi_axes_m, dtype=np.float64))

    passive_h = args.passive_horizon_s
    if passive_h is None:
        passive_h = float(os.environ.get("PASSIVE_CHECK_HORIZON_S", str(BRT_HORIZON_S)))

    conditions = _parse_conditions(args.conditions)
    rule_plan = build_rule_based_radial_plan(scenario) if EvalCondition.RULE_BASED in conditions else None

    print(f"Loading BRT from {ck_dir} …")
    brt = KozDeepReachBRT.load(ck_dir, device=args.device)
    print(
        f"Evaluating {len(plans)} LLM plan(s), conditions={[c.value for c in conditions]}, "
        f"filter_mode={args.filter_mode}, capture_radius={args.capture_radius_m:.0f} m"
    )

    results = run_llm_benchmark(
        scenario,
        plans,
        brt,
        plant,
        inner_koz=inner,
        conditions=conditions,
        rule_based_plan=rule_plan,
        passive_horizon_s=passive_h,
        filter_mode=args.filter_mode,  # type: ignore[arg-type]
        max_perturb_m_s=args.filter_max_perturb,
        n_sphere_samples=args.filter_n_sphere,
        brt_margin=args.brt_margin,
        dv_cap_m_s=scenario.dv_cap_m_s,
        capture_radius_m=args.capture_radius_m,
        progress_min_m=args.progress_min_m,
    )

    out = write_results_csv(args.output, results)
    summary = summarize_results(results)
    summary_path = write_summary_json(args.summary_json, summary)
    print(f"Wrote {out} ({len(results)} rows)")
    print(f"Wrote {summary_path}")
    for cond, stats in summary.get("by_condition", {}).items():
        if cond == EvalCondition.BRT_FILTER.value:
            print(
                f"  [{cond}] n={stats['n_plans']:.0f}  "
                f"LLM_unsafe={stats['llm_unsafe_rate']:.1%}  "
                f"BRT_intervened={stats.get('brt_intervention_rate', 0):.1%}  "
                f"burns_intervened/plan={stats.get('mean_burns_intervened_per_plan', 0):.2f}  "
                f"suppressed/plan={stats.get('mean_burns_suppressed_per_plan', 0):.2f}  "
                f"post_filter_unsafe={stats['post_filter_unsafe_rate']:.1%}  "
                f"mean_Δv_overhead={stats['mean_dv_overhead_m_s']:.4f} m/s"
            )
        else:
            print(
                f"  [{cond}] n={stats['n_plans']:.0f}  "
                f"LLM_unsafe={stats['llm_unsafe_rate']:.1%}  "
                f"reasons=(see CSV nominal_intervention_reasons)  "
                f"brt_unsafe={stats['brt_unsafe_rate']:.1%}  "
                f"intercept={stats['interception_rate']:.1%}  "
                f"tier_b={stats['mission_success_tier_b_rate']:.1%}"
            )
    print("\n=== Tier-B by category ===")
    for cat, stats in summary.get("by_category", {}).items():
        kind = stats.get("success_kind", "?")
        print(
            f"  {cat}: kind={kind}  n={stats['n_plans']:.0f}  "
            f"tier_b={stats['mission_success_tier_b_rate']:.1%}  "
            f"intercept={stats['interception_rate']:.1%}  "
            f"mean_Δclosed={stats['mean_range_closed_m']:.1f} m"
        )


if __name__ == "__main__":
    main()
