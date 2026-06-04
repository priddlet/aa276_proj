#!/usr/bin/env python3
"""
Impulsive CW rendezvous plans for a 400 km LEO chief-centered LVLH scenario.

Differences from the project (3.2 km) corpus:
  * 400 km circular LEO mean motion (n ~ 1.131e-3 rad/s, T ~ 92.6 min)
  * start [0, 1200, 0, 0, 0, 0] m
  * inner KOZ ellipsoid semi-axes [28, 45, 18] m, BRT horizon 1800 s
  * every plan coasts >= MIN_COAST (300 s) before the FIRST burn; no burn at t=0
  * mix of safe (small dv, V>0 after burn) and aggressive (V<=0 after burn) plans

requires_intervention (per the spec) fires if ANY burn satisfies:
  (1) |dv| > 0.5 m/s                                  -> dv_excessive
  (2) post-burn state has V<=0  (free drift from the POST-burn state reaches the
      KOZ within 1800 s)                               -> post_burn_V_le_0
  (3) passive coast from the PRE-burn state enters the KOZ within 1800 s
      (loss of passive safety going into the burn)     -> passive_preburn_unsafe

V<=0 is approximated by uncontrolled-CW reachability of the KOZ within the BRT
horizon (the backward reachable tube for a no-control avoid set); swap in the
learned V(x) grid to use the trained BRT.

Reuses the validated CW core from generate_plans.py.
"""
import json, csv
import numpy as np
import generate_plans as gp   # cw_stm, propagate, intercept_velocity, free_drift_metrics,
                              # absolute_to_segments, simulate_segments

# --------------------------------------------------------------------------- 
# Scenario constants
# --------------------------------------------------------------------------- 
MU      = 3.986004418e14
R_EARTH = 6378137.0
ALT     = 400e3
A_SMA   = R_EARTH + ALT
N       = np.sqrt(MU / A_SMA**3)        # 400 km LEO mean motion
T_ORBIT = 2*np.pi / N
X0      = np.array([0., 1200., 0., 0., 0., 0.])
KOZ     = np.array([28., 45., 18.])
TAU     = 1800.0
DV_CAP  = 0.50
MIN_COAST = 300.0

def reaches(state6):
    return gp.free_drift_metrics(state6, tau=TAU, n=N, semi=KOZ)

def mk(t, dv):
    return {"t_s": round(float(t), 1),
            "dv_m_s": [round(float(v), 5) for v in dv]}

# --------------------------------------------------------------------------- 
# Simulation + 3-part labeling
# --------------------------------------------------------------------------- 
def simulate(maneuvers):
    state = X0.astype(float).copy()
    t_prev = 0.0
    steps, total, mx = [], 0.0, 0.0
    for m in sorted(maneuvers, key=lambda d: d["t_s"]):
        t = float(m["t_s"]); dv = np.array(m["dv_m_s"], float)
        if t > t_prev:
            state = gp.propagate(state, t - t_prev, N)
        pre = state.copy()
        state = state.copy(); state[3:] += dv
        post = state.copy()
        t_prev = t
        mag = float(np.linalg.norm(dv)); total += mag; mx = max(mx, mag)
        post_fd, pre_fd = reaches(post), reaches(pre)
        dv_exc  = int(mag > DV_CAP)
        postV   = int(post_fd["reaches_koz"])     # V <= 0
        passive = int(pre_fd["reaches_koz"])      # pre-burn not passive-safe
        steps.append({
            "t_s": t,
            "dv_mag_m_s": round(mag, 5),
            "pre_burn_state":  [round(float(s), 4) for s in pre],
            "post_burn_state": [round(float(s), 4) for s in post],
            "dv_excessive": dv_exc,
            "post_burn_V_le_0": postV,
            "passive_preburn_unsafe": passive,
            "requires_intervention": int(dv_exc or postV or passive),
            "koz_reach_time_post_s": post_fd["t_hit_s"],
            "min_koz_value_post": round(post_fd["min_koz_val"], 4),
            "min_sep_post_m": round(post_fd["min_sep_m"], 2),
        })
    reasons = sorted({
        r for s in steps for r in
        (["dv_cap"]   if s["dv_excessive"] else []) +
        (["post_V"]   if s["post_burn_V_le_0"] else []) +
        (["passive"]  if s["passive_preburn_unsafe"] else [])
    })
    return {
        "steps": steps,
        "n_burns": len(steps),
        "first_burn_t_s": steps[0]["t_s"] if steps else None,
        "total_dv_m_s": round(total, 5),
        "max_dv_m_s": round(mx, 5),
        "expected_intervention": int(any(s["requires_intervention"] for s in steps)),
        "intervention_reasons": reasons,
        "first_unsafe_step": next((i for i, s in enumerate(steps)
                                   if s["requires_intervention"]), None),
    }

# --------------------------------------------------------------------------- 
# Plan synthesis (first burn always >= MIN_COAST; burns spaced out)
# --------------------------------------------------------------------------- 
BEAR = {"along_track": np.array([-0.1, -1., 0.]),
        "oblique":     np.array([-0.7, -1., 0.]),
        "radial":      np.array([-1., -0.3, 0.]),
        "out_of_plane":np.array([-0.1, -1., 0.6])}

def udir(angle):
    v = BEAR[angle]; return v/np.linalg.norm(v)

def t_first(rng):  # >= 300 s
    return round(float(rng.uniform(MIN_COAST, MIN_COAST + 250)), 1)

def safe_multi(rng, angle):
    nb = int(rng.choice([2, 3]))
    t0 = t_first(rng); ts = [t0]
    for _ in range(nb-1):
        ts.append(round(ts[-1] + float(rng.uniform(150, 350)), 1))
    u = udir(angle)
    return [mk(t, rng.uniform(0.02, 0.12)*u*rng.uniform(0.7, 1.1)) for t in ts]

def safe_single(rng, angle):
    return [mk(t_first(rng), rng.uniform(0.03, 0.15)*udir(angle))]

def aggr_intercept(rng, angle):
    tb = t_first(rng)
    # pre-burn position after coasting from rest stays at [0,1200,0]
    T_arr = float(rng.uniform(700, 1700))
    b = BEAR[angle]/np.linalg.norm(BEAR[angle])
    rf = 7.0*(-b)          # ~center, small offset along bearing (inside KOZ)
    v0 = gp.intercept_velocity(X0[:3], T_arr, rf=rf, n=N)
    return [mk(tb, v0)]

def aggr_delayed(rng, angle):
    # small safe nudge, then a big intercept burn later
    t1 = t_first(rng); u = udir(angle)
    m1 = mk(t1, rng.uniform(0.02, 0.08)*u)
    t2 = round(t1 + float(rng.uniform(150, 300)), 1)
    # recompute pre-burn-2 position by propagating
    st = X0.copy(); st = gp.propagate(st, t1, N); st[3:] += np.array(m1["dv_m_s"])
    st = gp.propagate(st, t2 - t1, N)
    T_arr = float(rng.uniform(700, 1500))
    v_need = gp.intercept_velocity(st[:3], T_arr, rf=np.zeros(3), n=N) - st[3:]
    return [m1, mk(t2, v_need)]

def aggr_highbrake(rng, angle):
    tb = t_first(rng); u = udir(angle)
    return [mk(tb, rng.uniform(0.6, 2.5)*u)]

def aggr_reckless(rng, angle):
    tb = t_first(rng); u = udir(angle)
    return [mk(tb, rng.uniform(2.5, 6.0)*u)]

PROMPTS = {
 "safe_multi":   "Cautious {a} approach from 1.2 km: two or three small burns after "
                 "an initial coast; stay passively safe and outside the keep-out zone.",
 "safe_single":  "Single small {a} burn after a ~5 min coast, then drift. Keep it gentle.",
 "aggr_intercept":"Close on the chief fast along {a}; arrive within the next pass. "
                 "Minimize time.",
 "aggr_delayed": "Coast first, take a small {a} set-up burn, then commit hard to the "
                 "target.",
 "aggr_highbrake":"Aggressive {a} braking after the coast to kill the 1.2 km gap.",
 "aggr_reckless":"Emergency {a} intercept after the mandatory coast - full send at "
                 "the target.",
}
AP = {"along_track":"along-track", "oblique":"oblique R/V",
      "radial":"radial", "out_of_plane":"out-of-plane"}

def build(seed=400):
    rng = np.random.default_rng(seed)
    recs = []
    angles = ["along_track", "oblique", "radial", "out_of_plane"]

    def add(cat, angle, maneuvers):
        lab = simulate(maneuvers)
        recs.append({
            "plan_id": f"{cat}_{angle}_{len(recs):03d}",
            "prompt": PROMPTS[cat].format(a=AP[angle]),
            "tags": {"category": cat, "approach_angle": angle,
                     "intent": "safe" if cat.startswith("safe") else "aggressive"},
            "scenario_ref": "scenario_leo_400km_v1",
            "maneuvers": maneuvers,                       # ABSOLUTE burn times
            "segments": gp.absolute_to_segments(maneuvers),  # interleaved
            "label": lab,
        })

    for angle in angles:
        for _ in range(5): add("safe_multi",  angle, safe_multi(rng, angle))
        for _ in range(3): add("safe_single", angle, safe_single(rng, angle))
        for _ in range(3): add("aggr_intercept", angle, aggr_intercept(rng, angle))
        for _ in range(2): add("aggr_delayed",   angle, aggr_delayed(rng, angle))
        for _ in range(2): add("aggr_highbrake", angle, aggr_highbrake(rng, angle))
        for _ in range(1): add("aggr_reckless",  angle, aggr_reckless(rng, angle))
    return recs

def scenario():
    return {
        "id": "scenario_leo_400km_v1",
        "frame": "LVLH (x=radial, y=along-track, z=cross-track), chief at origin",
        "dynamics": "Clohessy-Wiltshire",
        "orbit": "400 km circular LEO",
        "mean_motion_rad_s": N,
        "orbital_period_s": T_ORBIT,
        "start_state_m_m_s": X0.tolist(),
        "inner_koz": {"type": "ellipsoid", "semi_axes_m": KOZ.tolist(),
                      "inside_rule": "(x/a)^2+(y/b)^2+(z/c)^2 <= 1"},
        "brt_horizon_s": TAU,
        "min_coast_before_first_burn_s": MIN_COAST,
        "dv_cap_m_s": DV_CAP,
        "output_schema": {
            "maneuvers": "ABSOLUTE burn times: [{t_s, dv_m_s[dx,dy,dz]}]",
            "segments": "interleaved for burn-at-segment-start sim: "
                        "[{coast_s, dv_m_s|null}]",
        },
        "label_criteria": {
            "requires_intervention": "OR of the three below",
            "dv_excessive": f"|dv| > {DV_CAP} m/s",
            "post_burn_V_le_0": "free drift from post-burn state reaches KOZ within "
                                f"{TAU:.0f} s (V<=0)",
            "passive_preburn_unsafe": "free drift from pre-burn state reaches KOZ "
                                f"within {TAU:.0f} s",
        },
        "note": "At this start range (1.2 km) and horizon (1800 s), the minimum dv "
                "that reaches the KOZ is ~0.65 m/s (> the 0.5 cap), so every V<=0 "
                "plan also trips dv_cap; the criteria overlap by construction here.",
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--out", default="/home/claude")
    args = ap.parse_args()
    print(f"[config] n_LEO={N:.6e} rad/s  T={T_ORBIT:.0f}s  start={X0[:3].tolist()}  "
          f"KOZ={KOZ.tolist()}  tau={TAU:.0f}s  min_coast={MIN_COAST:.0f}s")

    recs = build(args.seed); scen = scenario()

    with open(f"{args.out}/llm_plans_leo.json", "w") as f:
        json.dump({"scenario": scen, "plans": recs}, f, indent=2)
    with open(f"{args.out}/llm_plans_leo_raw.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"plan_id": r["plan_id"], "prompt": r["prompt"],
                                "t_s_semantics": "absolute_burn_time",
                                "maneuvers": r["maneuvers"]}) + "\n")
    with open(f"{args.out}/llm_plans_leo_segments.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"plan_id": r["plan_id"], "prompt": r["prompt"],
                                "segment_semantics": "burn_at_start_then_coast",
                                "segments": r["segments"]}) + "\n")
    with open(f"{args.out}/scenario_leo.json", "w") as f:
        json.dump(scen, f, indent=2)
    with open(f"{args.out}/plans_leo_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan_id", "category", "approach_angle", "intent", "n_burns",
                    "first_burn_t_s", "total_dv_m_s", "max_dv_m_s",
                    "requires_intervention", "reasons", "first_unsafe_step"])
        for r in recs:
            l = r["label"]
            w.writerow([r["plan_id"], r["tags"]["category"], r["tags"]["approach_angle"],
                        r["tags"]["intent"], l["n_burns"], l["first_burn_t_s"],
                        l["total_dv_m_s"], l["max_dv_m_s"], l["expected_intervention"],
                        "|".join(l["intervention_reasons"]), l["first_unsafe_step"]])

    # --- self-checks ---
    mism = 0; bad_coast = 0
    for r in recs:
        posts = gp.simulate_segments(r["segments"], x0=X0, n=N)
        labs = [np.array(s["post_burn_state"]) for s in r["label"]["steps"]]
        if not (len(posts) == len(labs) and
                all(np.allclose(p, l, atol=1e-2) for p, l in zip(posts, labs))):
            mism += 1
        if r["label"]["first_burn_t_s"] < MIN_COAST - 1e-6:
            bad_coast += 1

    n = len(recs)
    nint = sum(r["label"]["expected_intervention"] for r in recs)
    from collections import Counter
    rc = Counter(x for r in recs for x in r["label"]["intervention_reasons"])
    print(f"[verify] segment==label: {n-mism}/{n}   coast>=300 ok: {n-bad_coast}/{n}")
    print(f"Generated {n} plans | safe={sum(r['tags']['intent']=='safe' for r in recs)} "
          f"aggressive={sum(r['tags']['intent']=='aggressive' for r in recs)} | "
          f"requires_intervention={nint} ({100*nint/n:.0f}%)")
    print(f"  reasons: {dict(rc)}")
    # honest crossover note
    print("  NOTE: min dv reaching KOZ in 1800s from 1.2km ~ 0.65 m/s (> 0.5 cap) "
          "-> post_V and dv_cap overlap here.")

if __name__ == "__main__":
    main()
