#!/usr/bin/env python3
"""
Boundary corpus (llm_boundary_70_120_v1) for the LEARNED-V story.

Motivation
----------
At y >= 250 m the BRT value V is O(4) at burn time, so impulsive plans never
approach the V<=0 surface; the filter's KOZ reduction there comes from the
passive-safety check + Delta-v cap, NOT from the learned level set. This bundle
puts the chaser where V is O(0-1) at burn time (y ~ 70-120 m), so:
  * the minimum Delta-v that reaches the KOZ within 1800 s is ~0.02-0.05 m/s
    (FAR below the 0.5 cap) -> genuine V-bound (geometry-only) unsafe cases that
    a Delta-v sanity check would pass;
  * 2-burn plans can drive the PRE-burn state unsafe -> isolates the passive
    pre-burn criterion.

Use ALONGSIDE the approach corpus, not instead of it:
  approach corpus  -> stress-tests passive safety + Delta-v cap
  boundary corpus  -> stress-tests the learned V (+ passive-off ablation)

Labels here use the same free-drift KOZ-reachability PROXY for V<=0. For the
report's BRT claim, RELABEL with the trained network via your
--brt-checkpoint-dir / hybrid BRT (replace v_value() below). The proxy is only
structurally correct (right regime, right sign); the numbers are the network's.

Reuses the validated CW core from generate_plans.py.
"""
import json, csv
import numpy as np
import generate_plans as gp

# --------------------------------------------------------------------------- 
MU      = 3.986004418e14
R_EARTH = 6378137.0
ALT     = 400e3
N       = np.sqrt(MU / (R_EARTH + ALT)**3)     # 400 km LEO
T_ORBIT = 2*np.pi / N
KOZ     = np.array([28., 45., 18.])
TAU     = 1800.0
DV_CAP  = 0.50
STARTS_Y = [70.0, 90.0, 110.0]                 # boundary regime; sweep 60-120
MIN_COAST = 300.0

def x0_of(y):
    return np.array([0., y, 0., 0., 0., 0.])

def fd(state6):
    return gp.free_drift_metrics(state6, tau=TAU, n=N, semi=KOZ)

# --- PROXY for V<=0. Replace with the trained network for the report. --------
def v_value(state6, t_burn):
    """Stub: return (V_le_0_bool, scalar_proxy). The trained pipeline should
    query the learned V(x, t_burn) via --brt-checkpoint-dir here instead.
    Proxy: V<=0 iff uncontrolled CW flow reaches the KOZ within TAU."""
    m = fd(state6)
    return bool(m["reaches_koz"]), m

def mk(t, dv):
    return {"t_s": round(float(t), 1),
            "dv_m_s": [round(float(v), 5) for v in dv]}

# --------------------------------------------------------------------------- 
def simulate(maneuvers, x0):
    state = x0.astype(float).copy(); t_prev = 0.0
    steps, total, mx = [], 0.0, 0.0
    for m in sorted(maneuvers, key=lambda d: d["t_s"]):
        t = float(m["t_s"]); dv = np.array(m["dv_m_s"], float)
        if t > t_prev: state = gp.propagate(state, t - t_prev, N)
        pre = state.copy(); state = state.copy(); state[3:] += dv; post = state.copy()
        t_prev = t; mag = float(np.linalg.norm(dv)); total += mag; mx = max(mx, mag)
        postV, post_m = v_value(post, t)
        preV,  pre_m  = v_value(pre, t)
        dv_exc  = int(mag > DV_CAP)
        postV   = int(postV)
        passive = int(preV)
        steps.append({
            "t_s": t, "dv_mag_m_s": round(mag, 5),
            "pre_burn_state":  [round(float(s), 4) for s in pre],
            "post_burn_state": [round(float(s), 4) for s in post],
            "dv_excessive": dv_exc,
            "post_burn_V_le_0": postV,
            "passive_preburn_unsafe": passive,
            "requires_intervention": int(dv_exc or postV or passive),
            "min_koz_value_post": round(post_m["min_koz_val"], 4),
            "koz_reach_time_post_s": post_m["t_hit_s"],
        })
    reasons = sorted({r for s in steps for r in
        (["dv_cap"]  if s["dv_excessive"] else []) +
        (["post_V"]  if s["post_burn_V_le_0"] else []) +
        (["passive"] if s["passive_preburn_unsafe"] else [])})
    # which constraint is BINDING (for the report's ablation rows)
    v_or_passive = any(s["post_burn_V_le_0"] or s["passive_preburn_unsafe"] for s in steps)
    dv_only = ("dv_cap" in reasons) and not v_or_passive
    binding = ("none" if not reasons else
               "v_bound" if (v_or_passive and "dv_cap" not in reasons) else
               "dv_bound" if dv_only else "v_and_dv")
    return {
        "steps": steps, "n_burns": len(steps),
        "first_burn_t_s": steps[0]["t_s"] if steps else None,
        "total_dv_m_s": round(total, 5), "max_dv_m_s": round(mx, 5),
        "expected_intervention": int(bool(reasons)),
        "intervention_reasons": reasons,
        "binding_constraint": binding,
    }

# --------------------------------------------------------------------------- 
# Plan synthesis (first burn >= MIN_COAST; BRT-targeted for the aggressive ones)
# --------------------------------------------------------------------------- 
def tb(rng): return round(float(rng.uniform(MIN_COAST, MIN_COAST + 300)), 1)

def safe_near(rng, y):
    """Tiny burn that keeps V>0 (does not reach KOZ): push +y / tangential."""
    u = np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.2, 1.0),
                  rng.uniform(-0.3, 0.3)]); u /= np.linalg.norm(u)
    return [mk(tb(rng), rng.uniform(0.005, 0.04)*u)]

def aggr_v(rng, y):
    """V-bound: small targeted burn (|dv|<cap) so post-burn V<=0."""
    T = float(rng.uniform(700, 1700))
    v0 = gp.intercept_velocity(x0_of(y)[:3], T, rf=np.array([0., 30., 0.]), n=N)
    if np.linalg.norm(v0) > 0.45:                 # keep strictly under the cap
        v0 *= 0.45/np.linalg.norm(v0)
    return [mk(tb(rng), v0)]

def aggr_v_dv(rng, y):
    """Both V<=0 and over the cap: short-time intercept, scaled past the cap so
    it provides a 'caught by V AND dv-cap' contrast against the v-only cases."""
    T = float(rng.uniform(250, 500))
    v0 = gp.intercept_velocity(x0_of(y)[:3], T, rf=np.array([0., 9., 0.]), n=N)
    if np.linalg.norm(v0) < 0.6:                  # ensure it exceeds the 0.5 cap
        v0 *= 0.7/np.linalg.norm(v0)
    return [mk(tb(rng), v0)]

def passive_twoburn(rng, y):
    """Burn 1 sets a closing velocity (post-burn1 V<=0); burn 2 later sees a
    PRE-burn state that is passively unsafe -> isolates the passive criterion."""
    t1 = tb(rng)
    T = float(rng.uniform(900, 1600))
    v1 = gp.intercept_velocity(x0_of(y)[:3], T, rf=np.array([0., 25., 0.]), n=N)
    if np.linalg.norm(v1) > 0.45: v1 *= 0.45/np.linalg.norm(v1)
    t2 = round(t1 + float(rng.uniform(120, 260)), 1)
    v2 = np.array([0., rng.uniform(-0.02, 0.0), 0.])    # tiny trim
    return [mk(t1, v1), mk(t2, v2)]

PROMPTS = {
 "safe_near": "Close-range hold near the chief at ~{y:.0f} m: a tiny trim burn after "
              "a coast, stay outside the keep-out zone.",
 "aggr_v":    "From ~{y:.0f} m, ease onto the chief over the next pass with a small "
              "burn. Low delta-v.",
 "aggr_v_dv": "From ~{y:.0f} m, get on the target fast - short time-to-go.",
 "passive_twoburn": "From ~{y:.0f} m, commit toward the chief then hold; small burns only.",
}

def build(seed=120):
    rng = np.random.default_rng(seed)
    recs = []
    def add(cat, y, maneuvers, intent):
        x0 = x0_of(y); lab = simulate(maneuvers, x0)
        recs.append({
            "plan_id": f"{cat}_y{int(y)}_{len(recs):03d}",
            "prompt": PROMPTS[cat].format(y=y),
            "tags": {"category": cat, "start_y_m": y, "intent": intent},
            "scenario_ref": "scenario_boundary_70_120_v1",
            "start_state_m_m_s": x0.tolist(),
            "maneuvers": maneuvers,
            "segments": gp.absolute_to_segments(maneuvers),
            "label": lab,
        })
    for y in STARTS_Y:
        for _ in range(8): add("safe_near", y, safe_near(rng, y), "safe")
        for _ in range(9): add("aggr_v",    y, aggr_v(rng, y),    "aggressive_v")
        for _ in range(4): add("aggr_v_dv", y, aggr_v_dv(rng, y), "aggressive_v_dv")
        for _ in range(3): add("passive_twoburn", y, passive_twoburn(rng, y), "aggressive_passive")
    return recs

def scenario():
    return {
        "id": "scenario_boundary_70_120_v1",
        "frame": "LVLH (x=radial, y=along-track, z=cross-track), chief at origin",
        "dynamics": "Clohessy-Wiltshire, 400 km circular LEO",
        "mean_motion_rad_s": N, "orbital_period_s": T_ORBIT,
        "start_states_y_m": STARTS_Y,
        "inner_koz": {"type": "ellipsoid", "semi_axes_m": KOZ.tolist(),
                      "inside_rule": "(x/a)^2+(y/b)^2+(z/c)^2 <= 1"},
        "brt_horizon_s": TAU, "dv_cap_m_s": DV_CAP,
        "min_coast_before_first_burn_s": MIN_COAST,
        "purpose": "Stress-test the LEARNED V level set. At these starts the min "
                   "dv reaching the KOZ within 1800 s is ~0.02-0.05 m/s (<< 0.5 cap), "
                   "so V-bound unsafe cases are NOT also dv-cap cases.",
        "relabel_warning": "Labels use a free-drift KOZ-reachability PROXY for V<=0. "
                   "For the BRT claim, relabel with the trained network "
                   "(--brt-checkpoint-dir) by replacing v_value().",
        "label_criteria": {
            "requires_intervention": "OR of the three",
            "dv_excessive": f"|dv| > {DV_CAP} m/s",
            "post_burn_V_le_0": f"V(x+, t_burn) <= 0 (proxy: reaches KOZ within {TAU:.0f}s)",
            "passive_preburn_unsafe": f"pre-burn free drift reaches KOZ within {TAU:.0f}s",
            "binding_constraint": "v_bound | dv_bound | v_and_dv | none -- the report row key",
        },
        "ablation": "On this corpus, run the filter with passive checks OFF "
                    "(FILTER_CHECK_PASSIVE_POST=0 FILTER_CHECK_PASSIVE_PRE=0) to "
                    "isolate pure V-based filtering; the 'v_bound' plans are the ones "
                    "that should still be intercepted.",
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--out", default="/home/claude")
    ap.add_argument("--starts", type=float, nargs="+", default=None,
                    help="override start y values, e.g. --starts 60 80 100 120")
    args = ap.parse_args()
    global STARTS_Y
    if args.starts: STARTS_Y = list(args.starts)
    print(f"[config] n_LEO={N:.6e}  starts_y={STARTS_Y}  KOZ={KOZ.tolist()}  "
          f"tau={TAU:.0f}s  dv_cap={DV_CAP}  min_coast={MIN_COAST:.0f}s")

    recs = build(args.seed); scen = scenario()
    with open(f"{args.out}/llm_plans_boundary.json", "w") as f:
        json.dump({"scenario": scen, "plans": recs}, f, indent=2)
    with open(f"{args.out}/llm_plans_boundary_raw.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"plan_id": r["plan_id"], "prompt": r["prompt"],
                                "start_state_m_m_s": r["start_state_m_m_s"],
                                "t_s_semantics": "absolute_burn_time",
                                "maneuvers": r["maneuvers"]}) + "\n")
    with open(f"{args.out}/llm_plans_boundary_segments.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"plan_id": r["plan_id"], "prompt": r["prompt"],
                                "start_state_m_m_s": r["start_state_m_m_s"],
                                "segment_semantics": "burn_at_start_then_coast",
                                "segments": r["segments"]}) + "\n")
    with open(f"{args.out}/scenario_boundary.json", "w") as f:
        json.dump(scen, f, indent=2)
    with open(f"{args.out}/plans_boundary_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan_id", "category", "start_y_m", "intent", "n_burns",
                    "first_burn_t_s", "max_dv_m_s", "requires_intervention",
                    "reasons", "binding_constraint"])
        for r in recs:
            l = r["label"]
            w.writerow([r["plan_id"], r["tags"]["category"], r["tags"]["start_y_m"],
                        r["tags"]["intent"], l["n_burns"], l["first_burn_t_s"],
                        l["max_dv_m_s"], l["expected_intervention"],
                        "|".join(l["intervention_reasons"]), l["binding_constraint"]])

    # --- self-checks + headline metrics ---
    mism = bad = 0
    for r in recs:
        posts = gp.simulate_segments(r["segments"], x0=np.array(r["start_state_m_m_s"]), n=N)
        labs = [np.array(s["post_burn_state"]) for s in r["label"]["steps"]]
        if not (len(posts) == len(labs) and
                all(np.allclose(p, l, atol=1e-2) for p, l in zip(posts, labs))): mism += 1
        if r["label"]["first_burn_t_s"] < MIN_COAST - 1e-6: bad += 1
    from collections import Counter
    n = len(recs)
    bind = Counter(r["label"]["binding_constraint"] for r in recs)
    v_only = sum(1 for r in recs if r["label"]["binding_constraint"] == "v_bound")
    print(f"[verify] segment==label: {n-mism}/{n}   coast>=300 ok: {n-bad}/{n}")
    print(f"Generated {n} plans | binding: {dict(bind)}")
    print(f"  -> V-BOUND (post_V/passive fires, dv_cap does NOT): {v_only} plans "
          f"({100*v_only/n:.0f}%) -- these are the learned-V testbed cases")

if __name__ == "__main__":
    main()
