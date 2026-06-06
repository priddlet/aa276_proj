#!/usr/bin/env python3
"""Impulsive CW rendezvous plans for 400 km LEO (default start y = 250 m).

Generates a mix of cautious and aggressive LLM-style maneuver plans, labels each
burn for whether a safety filter should intervene, and writes the plan bundle under llm/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import generate_plans as gp
import numpy as np

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------
MU = 3.986004418e14
R_EARTH = 6378137.0
ALT = 400e3
A_SMA = R_EARTH + ALT
N = np.sqrt(MU / A_SMA**3)  # 400 km LEO mean motion
T_ORBIT = 2 * np.pi / N
START_Y_M = 250.0
X0 = np.array([0.0, START_Y_M, 0.0, 0.0, 0.0, 0.0])
KOZ = np.array([28.0, 45.0, 18.0])
TAU = 1800.0
DV_CAP = 0.50
MIN_COAST = 300.0
SCENARIO_ID = "scenario_leo_400km_v1"
_LABEL_MODE = "hybrid"
_BRT: Any | None = None


def configure_labeling(mode: str, brt: Any | None = None) -> None:
    global _LABEL_MODE, _BRT
    _LABEL_MODE = str(mode).strip().lower()
    _BRT = brt


def configure_start(y_m: float) -> None:
    global START_Y_M, X0
    START_Y_M = float(y_m)
    X0 = np.array([0.0, START_Y_M, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _range_label() -> str:
    y = float(START_Y_M)
    if y >= 1000:
        return f"{y / 1000:.1f} km"
    return f"{y:.0f} m"


def reaches(state6):
    return gp.free_drift_metrics(state6, tau=TAU, n=N, semi=KOZ)


def mk(t, dv):
    return {
        "t_s": round(float(t), 1),
        "dv_m_s": [round(float(v), 5) for v in dv],
    }


# ---------------------------------------------------------------------------
# Simulation + 3-part labeling
# ---------------------------------------------------------------------------
def _post_burn_unsafe(post: np.ndarray, t_s: float) -> tuple[int, str]:
    """Hybrid post-burn unsafe: learned V when available, else passive KOZ reach."""
    if _LABEL_MODE in ("brt", "hybrid") and _BRT is not None:
        state = np.asarray(post, dtype=np.float64).reshape(6)
        v = float(_BRT.value_at_tau(state, float(t_s)))
        return int(v <= 0.0 or not np.isfinite(v)), "brt_unsafe"
    fd = reaches(post)
    return int(fd["reaches_koz"]), "post_V"


def simulate(maneuvers):
    state = X0.astype(float).copy()
    t_prev = 0.0
    steps, total, mx = [], 0.0, 0.0
    for m in sorted(maneuvers, key=lambda d: d["t_s"]):
        t = float(m["t_s"])
        dv = np.array(m["dv_m_s"], float)
        if t > t_prev:
            state = gp.propagate(state, t - t_prev, N)
        pre = state.copy()
        state = state.copy()
        state[3:] += dv
        post = state.copy()
        t_prev = t
        mag = float(np.linalg.norm(dv))
        total += mag
        mx = max(mx, mag)
        post_fd, pre_fd = reaches(post), reaches(pre)
        dv_exc = int(mag > DV_CAP)
        postV, post_reason = _post_burn_unsafe(post, t)
        passive = int(pre_fd["reaches_koz"])
        steps.append(
            {
                "t_s": t,
                "dv_mag_m_s": round(mag, 5),
                "pre_burn_state": [round(float(s), 4) for s in pre],
                "post_burn_state": [round(float(s), 4) for s in post],
                "dv_excessive": dv_exc,
                "post_burn_V_le_0": postV,
                "passive_preburn_unsafe": passive,
                "requires_intervention": int(dv_exc or postV or passive),
                "koz_reach_time_post_s": post_fd["t_hit_s"],
                "min_koz_value_post": round(post_fd["min_koz_val"], 4),
                "min_sep_post_m": round(post_fd["min_sep_m"], 2),
                "_post_reason": post_reason,
            }
        )
    reasons = sorted(
        {
            r
            for s in steps
            for r in (
                (["dv_cap"] if s["dv_excessive"] else [])
                + ([s["_post_reason"]] if s["post_burn_V_le_0"] else [])
                + (["passive"] if s["passive_preburn_unsafe"] else [])
            )
        }
    )
    for s in steps:
        s.pop("_post_reason", None)
    return {
        "steps": steps,
        "n_burns": len(steps),
        "first_burn_t_s": steps[0]["t_s"] if steps else None,
        "total_dv_m_s": round(total, 5),
        "max_dv_m_s": round(mx, 5),
        "expected_intervention": int(any(s["requires_intervention"] for s in steps)),
        "intervention_reasons": reasons,
        "first_unsafe_step": next(
            (i for i, s in enumerate(steps) if s["requires_intervention"]),
            None,
        ),
    }


# ---------------------------------------------------------------------------
# Plan synthesis (first burn always >= MIN_COAST; burns spaced out)
# ---------------------------------------------------------------------------
BEAR = {
    "along_track": np.array([-0.1, -1.0, 0.0]),
    "oblique": np.array([-0.7, -1.0, 0.0]),
    "radial": np.array([-1.0, -0.3, 0.0]),
    "out_of_plane": np.array([-0.1, -1.0, 0.6]),
}


def udir(angle):
    v = BEAR[angle]
    return v / np.linalg.norm(v)


def t_first(rng):
    return round(float(rng.uniform(MIN_COAST, MIN_COAST + 250)), 1)


def safe_multi(rng, angle):
    nb = int(rng.choice([2, 3]))
    t0 = t_first(rng)
    ts = [t0]
    for _ in range(nb - 1):
        ts.append(round(ts[-1] + float(rng.uniform(150, 350)), 1))
    u = udir(angle)
    return [mk(t, rng.uniform(0.04, 0.16) * u * rng.uniform(0.85, 1.15)) for t in ts]


def safe_single(rng, angle):
    return [mk(t_first(rng), rng.uniform(0.03, 0.15) * udir(angle))]


def aggr_intercept(rng, angle):
    tb = t_first(rng)
    st = gp.propagate(X0.copy(), tb, N)
    T_arr = float(rng.uniform(700, 1700))
    b = BEAR[angle] / np.linalg.norm(BEAR[angle])
    rf = 7.0 * (-b)
    dv = gp.intercept_velocity(st[:3], T_arr, rf=rf, n=N, v0=st[3:])
    return [mk(tb, dv)]


def aggr_delayed(rng, angle):
    t1 = t_first(rng)
    u = udir(angle)
    m1 = mk(t1, rng.uniform(0.02, 0.08) * u)
    t2 = round(t1 + float(rng.uniform(150, 300)), 1)
    st = X0.copy()
    st = gp.propagate(st, t1, N)
    st[3:] += np.array(m1["dv_m_s"])
    st = gp.propagate(st, t2 - t1, N)
    T_arr = float(rng.uniform(700, 1500))
    v_need = gp.intercept_velocity(st[:3], T_arr, rf=np.zeros(3), n=N, v0=st[3:])
    return [m1, mk(t2, v_need)]


def aggr_highbrake(rng, angle):
    tb = t_first(rng)
    u = udir(angle)
    return [mk(tb, rng.uniform(0.35, 0.72) * u)]


def aggr_reckless(rng, angle):
    tb = t_first(rng)
    u = udir(angle)
    return [mk(tb, rng.uniform(0.55, 1.2) * u)]


def _prompts() -> dict[str, str]:
    r = _range_label()
    return {
        "safe_multi": (
            f"Cautious {{a}} approach from {r}: two or three small burns after "
            "an initial coast; stay passively safe and outside the keep-out zone."
        ),
        "safe_single": "Single small {a} burn after a ~5 min coast, then drift. Keep it gentle.",
        "aggr_intercept": "Close on the chief fast along {a}; arrive within the next pass. Minimize time.",
        "aggr_delayed": "Coast first, take a small {a} set-up burn, then commit hard to the target.",
        "aggr_highbrake": f"Aggressive {{a}} braking after the coast to kill the {r} gap.",
        "aggr_reckless": "Emergency {a} intercept after the mandatory coast - full send at the target.",
    }


AP = {
    "along_track": "along-track",
    "oblique": "oblique R/V",
    "radial": "radial",
    "out_of_plane": "out-of-plane",
}


def build(seed=400):
    rng = np.random.default_rng(seed)
    recs = []
    angles = ["along_track", "oblique", "radial", "out_of_plane"]
    prompts = _prompts()

    def add(cat, angle, maneuvers):
        lab = simulate(maneuvers)
        recs.append(
            {
                "plan_id": f"{cat}_{angle}_{len(recs):03d}",
                "prompt": prompts[cat].format(a=AP[angle]),
                "tags": {
                    "category": cat,
                    "approach_angle": angle,
                    "intent": "safe" if cat.startswith("safe") else "aggressive",
                },
                "scenario_ref": SCENARIO_ID,
                "maneuvers": maneuvers,
                "segments": gp.absolute_to_segments(maneuvers),
                "label": lab,
            }
        )

    for angle in angles:
        for _ in range(5):
            add("safe_multi", angle, safe_multi(rng, angle))
        for _ in range(3):
            add("safe_single", angle, safe_single(rng, angle))
        for _ in range(3):
            add("aggr_intercept", angle, aggr_intercept(rng, angle))
        for _ in range(2):
            add("aggr_delayed", angle, aggr_delayed(rng, angle))
        for _ in range(2):
            add("aggr_highbrake", angle, aggr_highbrake(rng, angle))
        for _ in range(1):
            add("aggr_reckless", angle, aggr_reckless(rng, angle))
    return recs


def scenario():
    y = float(START_Y_M)
    return {
        "id": SCENARIO_ID,
        "frame": "LVLH (x=radial, y=along-track, z=cross-track), chief at origin",
        "dynamics": "Clohessy-Wiltshire",
        "orbit": "400 km circular LEO",
        "mean_motion_rad_s": N,
        "orbital_period_s": T_ORBIT,
        "start_state_m_m_s": X0.tolist(),
        "inner_koz": {
            "type": "ellipsoid",
            "semi_axes_m": KOZ.tolist(),
            "inside_rule": "(x/a)^2+(y/b)^2+(z/c)^2 <= 1",
        },
        "brt_horizon_s": TAU,
        "min_coast_before_first_burn_s": MIN_COAST,
        "dv_cap_m_s": DV_CAP,
        "output_schema": {
            "maneuvers": "ABSOLUTE burn times: [{t_s, dv_m_s[dx,dy,dz]}]",
            "segments": "interleaved for burn-at-segment-start sim: [{coast_s, dv_m_s|null}]",
        },
        "label_criteria": {
            "requires_intervention": "OR of the three below",
            "dv_excessive": f"|dv| > {DV_CAP} m/s",
            "post_burn_V_le_0": (
                f"learned V(x⁺, t) ≤ 0 when BRT loaded ({_LABEL_MODE} mode); "
                f"else free drift to KOZ within {TAU:.0f} s"
            ),
            "passive_preburn_unsafe": (
                f"free drift from pre-burn state reaches KOZ within {TAU:.0f} s"
            ),
            "label_mode": _LABEL_MODE,
        },
        "note": (
            f"Start y={y:.0f} m (learned BRT unsafe tube ~45–80 m passive; KOZ y semi-axis 45 m). "
            "Closer than the old 1.2 km bundle so intercept burns are smaller and "
            "post_V / dv_cap / passive labels are less degenerate."
        ),
    }


def main():
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument(
        "--start-y-m",
        type=float,
        default=START_Y_M,
        help="Along-track start range (m); default matches learned BRT scale (~250).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Output directory for llm_plans_leo.* and scenario_leo.json",
    )
    ap.add_argument(
        "--label-mode",
        type=str,
        default="hybrid",
        choices=("passive", "brt", "hybrid"),
        help="post-burn unsafe: passive CW, learned BRT, or BRT when checkpoint given.",
    )
    ap.add_argument(
        "--brt-checkpoint-dir",
        type=str,
        default="",
        help="Load v3 checkpoint for hybrid/brt labeling (project-root relative path ok).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device when loading BRT for labeling.",
    )
    args = ap.parse_args()
    configure_start(args.start_y_m)

    brt = None
    label_mode = args.label_mode
    if args.brt_checkpoint_dir.strip():
        root = Path(__file__).resolve().parents[1]
        ck = Path(args.brt_checkpoint_dir).expanduser()
        if not ck.is_absolute():
            ck = (root / ck).resolve()
        from simulation.brt.deepreach_mpc_brt import KozDeepReachBRT

        print(f"[label] loading BRT from {ck} …")
        brt = KozDeepReachBRT.load(ck, device=args.device)
        if label_mode == "passive":
            label_mode = "hybrid"
    elif label_mode == "brt":
        print("[warn] --label-mode brt requires --brt-checkpoint-dir; falling back to passive", file=__import__("sys").stderr)
        label_mode = "passive"
    configure_labeling(label_mode, brt)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[config] n_LEO={N:.6e} rad/s  T={T_ORBIT:.0f}s  start={X0[:3].tolist()}  "
        f"KOZ={KOZ.tolist()}  tau={TAU:.0f}s  min_coast={MIN_COAST:.0f}s"
    )

    recs = build(args.seed)
    scen = scenario()

    with open(out / "llm_plans_leo.json", "w", encoding="utf-8") as f:
        json.dump({"scenario": scen, "plans": recs}, f, indent=2)
    with open(out / "llm_plans_leo_raw.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(
                json.dumps(
                    {
                        "plan_id": r["plan_id"],
                        "prompt": r["prompt"],
                        "t_s_semantics": "absolute_burn_time",
                        "maneuvers": r["maneuvers"],
                    }
                )
                + "\n"
            )
    with open(out / "llm_plans_leo_segments.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(
                json.dumps(
                    {
                        "plan_id": r["plan_id"],
                        "prompt": r["prompt"],
                        "segment_semantics": "burn_at_start_then_coast",
                        "segments": r["segments"],
                    }
                )
                + "\n"
            )
    with open(out / "scenario_leo.json", "w", encoding="utf-8") as f:
        json.dump(scen, f, indent=2)
    with open(out / "plans_leo_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "plan_id",
                "category",
                "approach_angle",
                "intent",
                "n_burns",
                "first_burn_t_s",
                "total_dv_m_s",
                "max_dv_m_s",
                "requires_intervention",
                "reasons",
                "first_unsafe_step",
            ]
        )
        for r in recs:
            l = r["label"]
            w.writerow(
                [
                    r["plan_id"],
                    r["tags"]["category"],
                    r["tags"]["approach_angle"],
                    r["tags"]["intent"],
                    l["n_burns"],
                    l["first_burn_t_s"],
                    l["total_dv_m_s"],
                    l["max_dv_m_s"],
                    l["expected_intervention"],
                    "|".join(l["intervention_reasons"]),
                    l["first_unsafe_step"],
                ]
            )

    mism = 0
    bad_coast = 0
    for r in recs:
        posts = gp.simulate_segments(r["segments"], x0=X0, n=N)
        labs = [np.array(s["post_burn_state"]) for s in r["label"]["steps"]]
        if not (
            len(posts) == len(labs)
            and all(np.allclose(p, l, atol=1e-2) for p, l in zip(posts, labs))
        ):
            mism += 1
        if r["label"]["first_burn_t_s"] < MIN_COAST - 1e-6:
            bad_coast += 1

    n = len(recs)
    nint = sum(r["label"]["expected_intervention"] for r in recs)
    rc = Counter(x for r in recs for x in r["label"]["intervention_reasons"])
    print(f"[verify] segment==label: {n - mism}/{n}   coast>=300 ok: {n - bad_coast}/{n}")
    print(
        f"Generated {n} plans | safe={sum(r['tags']['intent'] == 'safe' for r in recs)} "
        f"aggressive={sum(r['tags']['intent'] == 'aggressive' for r in recs)} | "
        f"requires_intervention={nint} ({100 * nint / n:.0f}%)"
    )
    print(f"  reasons: {dict(rc)}")
    print(f"  wrote → {out}")


if __name__ == "__main__":
    main()
