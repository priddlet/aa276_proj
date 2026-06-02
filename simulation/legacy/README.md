# Legacy HJ reachability (Option 1 BRT)

Grid-based 6D backward reachable tube for the CW deputy and ellipsoidal inner KOZ:

- `hj_koz_brt.py` — `solve_koz_collision_brt_6d`, `load_or_solve_koz_brt_6d`, `KozHJTable6D`
- `hj_cw_6d_dynamics.py` — dynamics for `hj_reachability`
- `hj_brt_validation.py` — slice sanity checks on stored grids

Requires vendored `hj_reachability/` at the project root and `pip install -r requirements-brt.txt`.

The demo (`python -m simulation`) loads or solves the grid and queries `V(x)` via `KozHJTable6D`.
