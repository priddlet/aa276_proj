#!/usr/bin/env python3
"""
BRT-V corpus: plans where the *learned* value function V(x+, t) <= 0 at burn time.

The proxy-based boundary bundle (y=70-120 m) rarely triggers learned V rejection
(V ~ 0.4-0.9 at post-burn). This generator searches burns with the trained network
at closer starts (y ~ 45-65 m) where V <= 0 is reachable under the 0.5 m/s cap.

Output: llm_plans_brt_v.json + segments jsonl (same layout as boundary bundle).
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
from simulation.cw_dynamics import CWDynamics, leo_circular_orbit
from simulation.keepout import EllipsoidKeepOut
from simulation.sampling.passive import natural_coast_hits_inner_koz

MU = 3.986004418e14
R_EARTH = 6378137.0
ALT = 400e3
N = np.sqrt(MU / (R_EARTH + ALT) ** 3)
T_ORBIT = 2 * np.pi / N
KOZ = np.array([28.0, 45.0, 18.0])
TAU = float(BRT_HORIZON_S)
DV_CAP = 0.50
MIN_COAST = 300.0
STARTS_Y = [48.0, 58.0, 65.0]


def x0_of(y: float) -> np.ndarray:
    return np.array([0.0, float(y), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def mk(t: float, dv: np.ndarray) -> dict:
    return {
        "t_s": round(float(t), 1),
        "dv_m_s": [round(float(v), 5) for v in np.asarray(dv, dtype=np.float64).reshape(3)],
    }


def absolute_to_segments(maneuvers: list[dict]) -> list[dict]:
    burns = sorted(maneuvers, key=lambda m: float(m["t_s"]))
    segs: list[dict] = []
    t_prev = 0.0
    for m in burns:
        t_b = float(m["t_s"])
        gap = t_b - t_prev
        if gap > 0.0:
            segs.append({"coast_s": round(gap, 1), "dv_m_s": None})
        segs.append({"coast_s": 0.0, "dv_m_s": list(m["dv_m_s"])})
        t_prev = t_b
    return segs


def simulate_segments(
    plant: CWDynamics,
    segments: list[dict],
    x0: np.ndarray,
) -> list[np.ndarray]:
    x = np.asarray(x0, dtype=np.float64).reshape(6).copy()
    posts: list[np.ndarray] = []
    for seg in segments:
        dt = float(seg["coast_s"])
        dv_raw = seg.get("dv_m_s")
        if dt > 0.0:
            x = plant.propagate(x, dt)
        if dv_raw is not None:
            dv = np.asarray(dv_raw, dtype=np.float64).reshape(3)
            x = plant.apply_impulsive_dv(x, dv)
            posts.append(x.copy())
    return posts


class BrtLabeler:
    def __init__(
        self,
        plant: CWDynamics,
        brt: KozDeepReachBRT,
        inner: EllipsoidKeepOut,
        *,
        brt_margin: float = 0.0,
    ) -> None:
        self.plant = plant
        self.brt = brt
        self.inner = inner
        self.brt_margin = float(brt_margin)

    def v_at(self, state6: np.ndarray, t_s: float) -> float:
        return float(self.brt.value_at_tau(state6, t_s))

    def label_maneuvers(self, maneuvers: list[dict], x0: np.ndarray) -> dict:
        state = np.asarray(x0, dtype=np.float64).reshape(6).copy()
        t_prev = 0.0
        steps: list[dict] = []
        total = mx = 0.0
        for m in sorted(maneuvers, key=lambda d: float(d["t_s"])):
            t = float(m["t_s"])
            dv = np.asarray(m["dv_m_s"], dtype=np.float64).reshape(3)
            if t > t_prev:
                state = self.plant.propagate(state, t - t_prev)
            pre = state.copy()
            state = self.plant.apply_impulsive_dv(state, dv)
            post = state.copy()
            t_prev = t
            mag = float(np.linalg.norm(dv))
            total += mag
            mx = max(mx, mag)
            v_post = self.v_at(post, t)
            v_pre = self.v_at(pre, t)
            dv_exc = int(mag > DV_CAP + 1e-9)
            post_v = int(v_post <= self.brt_margin or not np.isfinite(v_post))
            passive_pre = int(
                natural_coast_hits_inner_koz(self.plant, pre, self.inner, TAU, n_samples=64)
            )
            steps.append(
                {
                    "t_s": round(t, 1),
                    "dv_mag_m_s": round(mag, 5),
                    "pre_burn_state": [round(float(s), 4) for s in pre],
                    "post_burn_state": [round(float(s), 4) for s in post],
                    "dv_excessive": dv_exc,
                    "post_burn_V_le_0": post_v,
                    "passive_preburn_unsafe": passive_pre,
                    "requires_intervention": int(dv_exc or post_v or passive_pre),
                    "V_pre": round(v_pre, 4),
                    "V_post": round(v_post, 4),
                }
            )
        reasons = sorted(
            {
                r
                for s in steps
                for r in (
                    (["dv_cap"] if s["dv_excessive"] else [])
                    + (["post_V"] if s["post_burn_V_le_0"] else [])
                    + (["passive"] if s["passive_preburn_unsafe"] else [])
                )
            }
        )
        v_or_passive = any(s["post_burn_V_le_0"] or s["passive_preburn_unsafe"] for s in steps)
        dv_only = ("dv_cap" in reasons) and not v_or_passive
        binding = (
            "none"
            if not reasons
            else "v_bound"
            if (v_or_passive and "dv_cap" not in reasons)
            else "dv_bound"
            if dv_only
            else "v_and_dv"
        )
        return {
            "steps": steps,
            "n_burns": len(steps),
            "first_burn_t_s": steps[0]["t_s"] if steps else None,
            "total_dv_m_s": round(total, 5),
            "max_dv_m_s": round(mx, 5),
            "expected_intervention": int(bool(reasons)),
            "intervention_reasons": reasons,
            "binding_constraint": binding,
            "label_mode": "hybrid_brt",
        }


def _burn_time(rng: np.random.Generator) -> float:
    return round(float(rng.uniform(MIN_COAST, MIN_COAST + 900)), 1)


def _pre_state(plant: CWDynamics, x0: np.ndarray, t_b: float) -> np.ndarray:
    return plant.propagate(x0, float(t_b))


def search_brt_unsafe(
    labeler: BrtLabeler,
    plant: CWDynamics,
    x0: np.ndarray,
    rng: np.random.Generator,
    *,
    dv_max: float = 0.48,
    require_passive_pre_safe: bool = True,
    max_trials: int = 12000,
) -> tuple[dict, float, float] | None:
    """Find burn with learned V(x+, t) <= 0 and |dv| <= dv_max."""
    u_close = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for _ in range(int(max_trials)):
        t = _burn_time(rng)
        pre = _pre_state(plant, x0, t)
        if require_passive_pre_safe and natural_coast_hits_inner_koz(
            plant, pre, labeler.inner, TAU, n_samples=48
        ):
            continue
        # Mix directed closing burns and random samples.
        if rng.random() < 0.35:
            mag = float(rng.uniform(0.02, dv_max))
            dv = mag * u_close + rng.normal(scale=0.015, size=3)
            dv[2] *= 0.15
        else:
            dv = rng.uniform(-dv_max, dv_max, size=3)
            dv[2] *= 0.2
        n = float(np.linalg.norm(dv))
        if n < 1e-6 or n > dv_max:
            continue
        post = plant.apply_impulsive_dv(pre, dv)
        v_post = labeler.v_at(post, t)
        if v_post <= labeler.brt_margin:
            return mk(t, dv), labeler.v_at(pre, t), v_post
    return None


def search_brt_safe(
    labeler: BrtLabeler,
    plant: CWDynamics,
    x0: np.ndarray,
    rng: np.random.Generator,
    *,
    v_min: float = 0.03,
    dv_max: float = 0.04,
    max_trials: int = 8000,
) -> tuple[dict, float, float] | None:
    """Small trim with V_pre > 0 and V(x+, t) > v_min."""
    for _ in range(int(max_trials)):
        t = _burn_time(rng)
        pre = _pre_state(plant, x0, t)
        v_pre = labeler.v_at(pre, t)
        if v_pre <= v_min:
            continue
        u = rng.normal(size=3)
        u[1] = abs(u[1]) + 0.35
        u[0] *= 0.4
        u[2] *= 0.25
        u /= max(np.linalg.norm(u), 1e-9)
        mag = float(rng.uniform(0.004, dv_max))
        dv = mag * u
        post = plant.apply_impulsive_dv(pre, dv)
        v_post = labeler.v_at(post, t)
        if v_post > v_min and float(np.linalg.norm(dv)) <= dv_max:
            return mk(t, dv), v_pre, v_post
    return None


def search_brt_borderline(
    labeler: BrtLabeler,
    plant: CWDynamics,
    x0: np.ndarray,
    rng: np.random.Generator,
    *,
    dv_max: float = 0.35,
    max_trials: int = 8000,
) -> tuple[dict, float, float] | None:
    """0 < V(x+, t) <= 0.15 — filter should scale but not always zero."""
    for _ in range(int(max_trials)):
        t = _burn_time(rng)
        pre = _pre_state(plant, x0, t)
        dv = rng.uniform(-dv_max, dv_max, size=3)
        dv[2] *= 0.15
        n = float(np.linalg.norm(dv))
        if n < 0.02 or n > dv_max:
            continue
        post = plant.apply_impulsive_dv(pre, dv)
        v_post = labeler.v_at(post, t)
        if 0.0 < v_post <= 0.15:
            return mk(t, dv), labeler.v_at(pre, t), v_post
    return None


def mk_brt_v_dv(
    labeler: BrtLabeler,
    plant: CWDynamics,
    x0: np.ndarray,
    rng: np.random.Generator,
) -> list[dict] | None:
    found = search_brt_unsafe(labeler, plant, x0, rng, dv_max=0.42, require_passive_pre_safe=True)
    if found is None:
        return None
    m, _, _ = found
    dv = np.asarray(m["dv_m_s"], dtype=np.float64)
    target = float(rng.uniform(0.55, 0.68))
    if float(np.linalg.norm(dv)) < 1e-9:
        return None
    m["dv_m_s"] = (dv * (target / float(np.linalg.norm(dv)))).tolist()
    return [m]


PROMPTS = {
    "safe_brt": "Close-range hold at ~{y:.0f} m: tiny trim after coast; learned V stays clearly positive.",
    "brt_v_unsafe": "From ~{y:.0f} m, small closing burn (under cap) that the BRT flags unsafe (V<=0).",
    "brt_v_borderline": "From ~{y:.0f} m, moderate burn near the V=0 surface (filter should scale it).",
    "brt_v_dv": "From ~{y:.0f} m, fast closing burn over the delta-v cap (dv-only failure mode).",
}


def build(
    labeler: BrtLabeler,
    plant: CWDynamics,
    *,
    seed: int = 42,
    starts_y: list[float] | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    ys_unsafe = STARTS_Y if starts_y is None else list(starts_y)
    ys_safe = [y for y in ys_unsafe if y >= 58.0] or [58.0, 65.0]
    recs: list[dict] = []
    seq = 0

    def add(cat: str, y: float, maneuvers: list[dict], intent: str) -> None:
        nonlocal seq
        x0 = x0_of(y)
        lab = labeler.label_maneuvers(maneuvers, x0)
        recs.append(
            {
                "plan_id": f"{cat}_y{int(y)}_{seq:03d}",
                "prompt": PROMPTS[cat].format(y=y),
                "tags": {"category": cat, "start_y_m": y, "intent": intent},
                "scenario_ref": "scenario_brt_v_48_65_v1",
                "start_state_m_m_s": x0.tolist(),
                "maneuvers": maneuvers,
                "segments": absolute_to_segments(maneuvers),
                "label": lab,
            }
        )
        seq += 1

    def _need(cat: str, y: float, finder, intent: str, n: int) -> None:
        for _ in range(n):
            got = finder(y)
            if got is None:
                raise RuntimeError(f"{cat} search failed at y={y}")
            add(cat, y, [got[0]] if cat != "brt_v_dv" else got, intent)

    find_unsafe = lambda y: search_brt_unsafe(labeler, plant, x0_of(y), rng)
    find_safe = lambda y: search_brt_safe(labeler, plant, x0_of(y), rng)
    find_border = lambda y: search_brt_borderline(labeler, plant, x0_of(y), rng)
    find_dv = lambda y: mk_brt_v_dv(labeler, plant, x0_of(y), rng)

    for y in ys_unsafe:
        for _ in range(10):
            _need("brt_v_unsafe", y, find_unsafe, "brt_v_unsafe", 1)
        for _ in range(4):
            mans = find_dv(y)
            if mans is None:
                raise RuntimeError(f"brt_v_dv search failed at y={y}")
            add("brt_v_dv", y, mans, "brt_v_dv")

    for y in ys_safe:
        for _ in range(8):
            _need("safe_brt", y, find_safe, "safe", 1)
        for _ in range(4):
            _need("brt_v_borderline", y, find_border, "brt_borderline", 1)

    return recs


def scenario(starts_y: list[float], checkpoint: str) -> dict:
    return {
        "id": "scenario_brt_v_48_65_v1",
        "frame": "LVLH (x=radial, y=along-track, z=cross-track), chief at origin",
        "dynamics": "Clohessy-Wiltshire, 400 km circular LEO",
        "mean_motion_rad_s": N,
        "orbital_period_s": T_ORBIT,
        "start_states_y_m": starts_y,
        "inner_koz": {
            "type": "ellipsoid",
            "semi_axes_m": KOZ.tolist(),
            "inside_rule": "(x/a)^2+(y/b)^2+(z/c)^2 <= 1",
        },
        "brt_horizon_s": TAU,
        "dv_cap_m_s": DV_CAP,
        "min_coast_before_first_burn_s": MIN_COAST,
        "label_mode": "hybrid_brt",
        "label_checkpoint": checkpoint,
        "purpose": "Corpus tuned to the LEARNED V surface: burns found by search with "
        "V(x+,t_burn)<=0 at y~48-65 m (under dv cap). Filter V-rejection should be active.",
        "label_criteria": {
            "requires_intervention": "OR of dv_cap, learned post_burn_V_le_0, passive_preburn",
            "post_burn_V_le_0": "V(x+, t_burn) <= 0 (learned DeepReach-MPC)",
            "passive_preburn_unsafe": f"pre-burn free drift reaches KOZ within {TAU:.0f}s",
            "binding_constraint": "v_bound | dv_bound | v_and_dv | none",
        },
    }


def write_bundle(recs: list[dict], scen: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "llm_plans_brt_v.json").open("w", encoding="utf-8") as f:
        json.dump({"scenario": scen, "plans": recs}, f, indent=2)
    with (out_dir / "llm_plans_brt_v_raw.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(
                json.dumps(
                    {
                        "plan_id": r["plan_id"],
                        "prompt": r["prompt"],
                        "start_state_m_m_s": r["start_state_m_m_s"],
                        "t_s_semantics": "absolute_burn_time",
                        "maneuvers": r["maneuvers"],
                    }
                )
                + "\n"
            )
    with (out_dir / "llm_plans_brt_v_segments.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(
                json.dumps(
                    {
                        "plan_id": r["plan_id"],
                        "prompt": r["prompt"],
                        "start_state_m_m_s": r["start_state_m_m_s"],
                        "segment_semantics": "burn_at_start_then_coast",
                        "segments": r["segments"],
                    }
                )
                + "\n"
            )
    with (out_dir / "scenario_brt_v.json").open("w", encoding="utf-8") as f:
        json.dump(scen, f, indent=2)
    with (out_dir / "plans_brt_v_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "plan_id",
                "category",
                "start_y_m",
                "n_burns",
                "max_dv_m_s",
                "V_post",
                "requires_intervention",
                "reasons",
                "binding_constraint",
            ]
        )
        for r in recs:
            lab = r["label"]
            vpost = lab["steps"][0]["V_post"] if lab["steps"] else ""
            w.writerow(
                [
                    r["plan_id"],
                    r["tags"]["category"],
                    r["tags"]["start_y_m"],
                    lab["n_burns"],
                    lab["max_dv_m_s"],
                    vpost,
                    lab["expected_intervention"],
                    "|".join(lab["intervention_reasons"]),
                    lab["binding_constraint"],
                ]
            )


def main() -> None:
    if not DEEPREACH_MPC_AVAILABLE:
        print("DeepReach-MPC required.", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Generate BRT-V-targeted LLM plan corpus.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument("--starts", type=float, nargs="+", default=None)
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(ROOT / "simulation_output" / "deepreach_mpc_koz_v3"),
    )
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    starts = STARTS_Y if args.starts is None else list(args.starts)
    out_dir = Path(args.out).resolve()
    ck = Path(args.checkpoint_dir).resolve()

    leo = leo_circular_orbit(400.0)
    plant = CWDynamics(leo.n_rad_s)
    inner = EllipsoidKeepOut(KOZ.copy())
    print(f"Loading BRT from {ck} …")
    brt = KozDeepReachBRT.load(ck, device=args.device)
    labeler = BrtLabeler(plant, brt, inner)

    print(f"[config] starts_y={starts}  KOZ={KOZ.tolist()}  tau={TAU:.0f}s  dv_cap={DV_CAP}")
    recs = build(labeler, plant, seed=args.seed, starts_y=starts)
    scen = scenario(starts, str(ck))
    write_bundle(recs, scen, out_dir)

    n = len(recs)
    bind = Counter(r["label"]["binding_constraint"] for r in recs)
    post_v = sum(1 for r in recs if "post_V" in r["label"]["intervention_reasons"])
    v_only = sum(
        1
        for r in recs
        if r["label"]["binding_constraint"] == "v_bound"
        and "post_V" in r["label"]["intervention_reasons"]
    )
    mism = 0
    for r in recs:
        posts = simulate_segments(plant, r["segments"], np.array(r["start_state_m_m_s"]))
        labs = [np.array(s["post_burn_state"]) for s in r["label"]["steps"]]
        if not (
            len(posts) == len(labs)
            and all(np.allclose(p, l, atol=1e-2) for p, l in zip(posts, labs))
        ):
            mism += 1

    print(f"[verify] segment==label: {n-mism}/{n}")
    print(f"Generated {n} plans | binding: {dict(bind)}")
    print(f"  -> post_V (learned V<=0): {post_v}/{n} ({100*post_v/n:.0f}%)")
    print(f"  -> v_bound (V-only binding): {v_only}/{n} ({100*v_only/n:.0f}%)")
    print(f"Wrote {out_dir / 'llm_plans_brt_v.json'}")


if __name__ == "__main__":
    main()
