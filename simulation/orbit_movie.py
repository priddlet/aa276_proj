"""Animated ECI scene: global + formation panels, Earth proximity KOZ, spacecraft."""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable

import numpy as np

from simulation.cw_dynamics import R_EARTH_KM
from simulation.isosurface import extract_brt_v0_near_center
from simulation.keepout import EllipsoidKeepOut
from simulation.spacecraft_wire import bus_and_panel_edges, edges_to_nan_polyline, scale_edges


def lvlh_points_m_to_eci_km(
    r_chief_km: np.ndarray,
    R_lvlh_to_eci: np.ndarray,
    pts_lvlh_m: np.ndarray,
) -> np.ndarray:
    """Map relative positions (m) in chief LVLH to ECI (km). 'pts_lvlh_m' is '(..., 3)'."""
    r_c = np.asarray(r_chief_km, dtype=np.float64).reshape(3)
    R = np.asarray(R_lvlh_to_eci, dtype=np.float64).reshape(3, 3)
    p = np.asarray(pts_lvlh_m, dtype=np.float64)
    flat = p.reshape(-1, 3)
    rho_km = (R @ flat.T).T / 1000.0
    out = r_c.reshape(1, 3) + rho_km
    return out.reshape(p.shape)


def inner_koz_wireframe_eci_km(
    inner: EllipsoidKeepOut,
    r_chief_km: np.ndarray,
    R_lvlh_to_eci: np.ndarray,
    *,
    nu: int = 32,
    nv: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KOZ surface mesh in ECI km, attached to chief (LVLH origin at chief)."""
    Xm, Ym, Zm = inner.surface_mesh(nu=nu, nv=nv)
    pts = np.stack([Xm.ravel(), Ym.ravel(), Zm.ravel()], axis=-1)
    eci = lvlh_points_m_to_eci_km(r_chief_km, R_lvlh_to_eci, pts)
    n0, n1 = Xm.shape
    Xk = eci[:, 0].reshape(n0, n1)
    Yk = eci[:, 1].reshape(n0, n1)
    Zk = eci[:, 2].reshape(n0, n1)
    return Xk, Yk, Zk


def brt_position_isosurface_lvlh_m(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    vx: float,
    vy: float,
    vz: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    z_lo: float,
    z_hi: float,
    nx: int,
    ny: int,
    nz: int,
    *,
    level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return '(verts_lvlh_m, faces)' for 'V=level' in position at fixed velocity, or 'None'."""
    from simulation.isosurface import marching_cubes_v0_lvlh

    return marching_cubes_v0_lvlh(
        value_on_grid,
        vx,
        vy,
        vz,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        z_lo,
        z_hi,
        nx,
        ny,
        nz,
        level=level,
    )


def _earth_wireframe_km(
    radius_km: float = 6_378.137,
    nu: int = 72,
    nv: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    v = np.linspace(0.0, np.pi, nv)
    U, V = np.meshgrid(u, v, indexing="xy")
    x = radius_km * np.cos(U) * np.sin(V)
    y = radius_km * np.sin(U) * np.sin(V)
    z = radius_km * np.cos(V)
    return x, y, z


def _craft_edges_body_km(
    craft_scale_km: float,
    *,
    chief: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    ref_span_m = 32.0 if chief else 26.0
    s = float(craft_scale_km) / ref_span_m
    if chief:
        tmpl = bus_and_panel_edges((10.0, 4.0, 4.0), panel_length=22.0, panel_half_width=2.5)
    else:
        tmpl = bus_and_panel_edges((5.0, 2.0, 2.0), panel_length=12.0, panel_half_width=1.2)
    return scale_edges(tmpl, s)


def _earth_koz_violation(r_eci_km: np.ndarray, r_earth_km: float, min_altitude_km: float) -> bool:
    """True if spherical altitude is below 'min_altitude_km' (||r|| < R_Earth + h)."""
    r = float(np.linalg.norm(np.asarray(r_eci_km, dtype=np.float64).reshape(3)))
    return r < (float(r_earth_km) + float(min_altitude_km))


def render_orbit_eci_animation(
    ephem: dict[str, np.ndarray],
    *,
    trail_r_chief_km: np.ndarray | None = None,
    trail_r_deputy_km: np.ndarray | None = None,
    axis_length_km: float | None = None,
    craft_scale_km: float | None = None,
    eci_view: str | None = None,
    earth_wire_nu: int = 72,
    earth_wire_nv: int = 36,
    earth_koz_min_altitude_km: float | None = None,
    output_path: str | None = None,
    fps: float = 24.0,
    dpi: int = 110,
    show: bool = False,
    brt_option1: Any | None = None,
    inner_koz_formation: EllipsoidKeepOut | None = None,
    frame_log_rows: list[dict[str, Any]] | None = None,
    formation_brt_iso_resolution: tuple[int, int, int] = (18, 18, 14),
    formation_brt_margin_m: tuple[float, float, float] = (900.0, 900.0, 500.0),
) -> str | None:
    """ECI animation with global context and/or formation-following view.

    'eci_view' / env 'ORBIT_ECI_VIEW':

    - 'both' (default): left = global (Earth, trajectories, red shell at R_Earth+h);
      right = formation camera **following** chief/deputy midpoint each frame.
    - 'global': single global panel.
    - 'formation': single formation panel only.

    Earth KOZ: minimum spherical altitude 'h' km above Earth mean sphere; unsafe if
    '||r|| < R_Earth + h'. Wireframe at that radius in global view. Override with
    'earth_koz_min_altitude_km' or env 'EARTH_KOZ_MIN_ALT_KM' (default 150).

    If 'brt_option1' is set (e.g. :class:`~simulation.brt.deepreach_mpc_brt.KozDeepReachBRT') and 'ephem' contains
    'states_lvlh_m', each frame's suptitle includes learned BRT status (unsafe iff 'value <= 0').

    If 'inner_koz_formation' is set and the layout includes the formation axis, the inner
    KOZ ellipsoid is drawn in ECI attached to the chief each frame. If 'brt_option1' has
    'value_batch' (6D HJ table), a 'V=0' position isosurface at the deputy's current
    velocity is drawn (requires 'scikit-image' for marching cubes).

    When 'frame_log_rows' is a list, each animation frame appends one dict for post-run CSV.
    """
    import matplotlib

    if os.environ.get("MPLBACKEND", "").lower() == "agg" or os.environ.get("CI"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r_earth = float(R_EARTH_KM)
    h_koz = float(
        earth_koz_min_altitude_km
        if earth_koz_min_altitude_km is not None
        else os.environ.get("EARTH_KOZ_MIN_ALT_KM", "150")
    )

    times = ephem["times_s"]
    r_c = ephem["r_chief_km"]
    r_d = ephem["r_deputy_km"]
    Rc = ephem["R_chief_body_to_eci"]
    Rd = ephem["R_deputy_body_to_eci"]
    a_km = float(ephem["a_km"][0])
    n_rad_s = float(ephem["n_rad_s"][0])

    if "v_chief_km_s" in ephem and "v_deputy_km_s" in ephem:
        v_c_eph = np.asarray(ephem["v_chief_km_s"], dtype=np.float64)
        v_d_eph = np.asarray(ephem["v_deputy_km_s"], dtype=np.float64)
    else:
        v_c_eph = np.gradient(r_c, times, axis=0, edge_order=2)
        v_d_eph = np.gradient(r_d, times, axis=0, edge_order=2)

    states_lvlh = ephem.get("states_lvlh_m")
    if brt_option1 is not None and states_lvlh is None:
        warnings.warn("brt_option1 set but ephem has no 'states_lvlh_m'; BRT caption disabled.", UserWarning)

    r_c_tr = np.asarray(trail_r_chief_km, dtype=np.float64) if trail_r_chief_km is not None else r_c
    r_d_tr = np.asarray(trail_r_deputy_km, dtype=np.float64) if trail_r_deputy_km is not None else r_d
    if r_c_tr.ndim != 2 or r_c_tr.shape[1] != 3:
        raise ValueError("trail_r_chief_km must be (N, 3)")
    if r_d_tr.ndim != 2 or r_d_tr.shape[1] != 3:
        raise ValueError("trail_r_deputy_km must be (N, 3)")

    n_frames = r_c.shape[0]
    view_mode = (eci_view or os.environ.get("ORBIT_ECI_VIEW", "both")).lower()
    dual = view_mode in ("both", "dual", "split")
    single_formation = view_mode in ("formation", "zoom", "pair", "proximity")
    single_global = view_mode == "global"

    sep_tr = np.linalg.norm(r_d_tr - r_c_tr, axis=1)
    sep_anim = np.linalg.norm(r_d - r_c, axis=1)
    max_sep_km = float(max(np.max(sep_tr), np.max(sep_anim), 1e-9))

    if dual or single_formation:
        if axis_length_km is None:
            axis_length_km = min(0.01 * a_km, max(1.5e-4 * a_km, 0.42 * max_sep_km))
        if craft_scale_km is None:
            craft_scale_km = min(0.009 * a_km, max(1.2e-4 * a_km, 0.32 * max_sep_km))
    else:
        if axis_length_km is None:
            axis_length_km = 0.01 * a_km
        if craft_scale_km is None:
            craft_scale_km = 0.009 * a_km

    L = float(axis_length_km)
    c_scale = float(craft_scale_km)
    edges_chief_km = _craft_edges_body_km(c_scale, chief=True)
    edges_dep_km = _craft_edges_body_km(c_scale * 0.72, chief=False)

    pad_form = 0.14 * max_sep_km + L + c_scale
    half_form = max(0.5 * max_sep_km + pad_form, 4.0 * (L + c_scale), 0.08)

    lim_g = float(
        1.12
        * max(
            np.max(np.linalg.norm(r_c_tr, axis=1)),
            np.max(np.linalg.norm(r_d_tr, axis=1)),
            a_km,
            r_earth + h_koz + 500.0,
        )
    )
    # Velocity-direction arrows (ECI): scale to a fraction of each panel's extent.
    L_vel_global_km = 0.055 * lim_g
    L_vel_form_km = min(0.22 * half_form, 0.06 * lim_g, 2.5 * max_sep_km)

    if dual:
        fig = plt.figure(figsize=(14.2, 6.8))
        ax_g = fig.add_subplot(1, 2, 1, projection="3d")
        ax_f = fig.add_subplot(1, 2, 2, projection="3d")
    elif single_global:
        fig = plt.figure(figsize=(10.0, 8.0))
        ax_g = fig.add_subplot(111, projection="3d")
        ax_f = None
    else:
        fig = plt.figure(figsize=(10.0, 8.0))
        ax_g = None
        ax_f = fig.add_subplot(111, projection="3d")

    def _plot_trails(ax) -> None:
        ax.plot(r_c_tr[:, 0], r_c_tr[:, 1], r_c_tr[:, 2], color="0.45", linewidth=1.05, alpha=0.88)
        ax.plot(r_d_tr[:, 0], r_d_tr[:, 1], r_d_tr[:, 2], color="tab:orange", linewidth=1.15, alpha=0.92)

    if ax_g is not None:
        Ex, Ey, Ez = _earth_wireframe_km(r_earth, nu=earth_wire_nu, nv=earth_wire_nv)
        ax_g.plot_wireframe(Ex, Ey, Ez, color="C0", linewidth=0.26, alpha=0.55)
        kx, ky, kz = _earth_wireframe_km(r_earth + h_koz, nu=48, nv=24)
        ax_g.plot_wireframe(kx, ky, kz, color="tab:red", linewidth=0.2, alpha=0.42)
        _plot_trails(ax_g)
        ax_g.set_xlim(-lim_g, lim_g)
        ax_g.set_ylim(-lim_g, lim_g)
        ax_g.set_zlim(-lim_g * 0.42, lim_g * 0.42)
        ax_g.set_xlabel("x (km)")
        ax_g.set_ylabel("y (km)")
        ax_g.set_zlabel("z (km)")
        ax_g.set_title("ECI")
        try:
            ax_g.set_box_aspect((1, 1, 0.45))
        except Exception:
            pass

    if ax_f is not None:
        _plot_trails(ax_f)
        ax_f.set_xlabel("x (km)")
        ax_f.set_ylabel("y (km)")
        ax_f.set_zlabel("z (km)")
        ax_f.set_title("Formation")
        try:
            ax_f.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def _make_pack(ax, mk: int, draw_axes: bool, draw_craft: bool, draw_velocity: bool, L_vel_km: float):
        pt_c = ax.plot([], [], [], "o", color="0.15", markersize=mk, markeredgecolor="white", markeredgewidth=0.35)[0]
        pt_d = ax.plot([], [], [], "o", color="tab:orange", markersize=mk, markeredgecolor="white", markeredgewidth=0.35)[0]
        chief_lns = []
        dep_lns = []
        if draw_axes:
            for col in ("#d62728", "#2ca02c", "#1f77b4"):
                chief_lns.append(ax.plot([], [], [], color=col, linewidth=1.85)[0])
            for col in ("#8c564b", "#9467bd", "#17becf"):
                dep_lns.append(ax.plot([], [], [], color=col, linewidth=1.85)[0])
        else:
            chief_lns = [ax.plot([], [], [], color="tab:red", linewidth=0.7, alpha=0.35)[0]]
            dep_lns = [ax.plot([], [], [], color="#9467bd", linewidth=0.7, alpha=0.35)[0]]
        ln_cc = ax.plot([], [], [], color="0.25", linewidth=1.0)[0] if draw_craft else None
        ln_cd = ax.plot([], [], [], color="tab:blue", linewidth=0.95)[0] if draw_craft else None
        if draw_velocity:
            # Colors chosen to avoid deputy v blending with tab:orange trajectory / marker.
            vel_c = ax.plot(
                [],
                [],
                [],
                color="#0277bd",
                linewidth=1.75,
                solid_capstyle="round",
                marker="o",
                markersize=4.5,
                markevery=[1],
                markeredgecolor="white",
                markeredgewidth=0.65,
            )[0]
            vel_d = ax.plot(
                [],
                [],
                [],
                color="#6a1b9a",
                linewidth=1.75,
                solid_capstyle="round",
                marker="o",
                markersize=4.5,
                markevery=[1],
                markeredgecolor="white",
                markeredgewidth=0.65,
            )[0]
        else:
            vel_c = vel_d = None
        return pt_c, pt_d, chief_lns, dep_lns, ln_cc, ln_cd, vel_c, vel_d, float(L_vel_km)

    pack_g = pack_f = None
    if dual:
        pack_g = _make_pack(ax_g, 7, draw_axes=False, draw_craft=False, draw_velocity=True, L_vel_km=L_vel_global_km)
        pack_f = _make_pack(ax_f, 9, draw_axes=True, draw_craft=True, draw_velocity=True, L_vel_km=L_vel_form_km)
    elif single_global:
        pack_g = _make_pack(ax_g, 7, draw_axes=True, draw_craft=True, draw_velocity=True, L_vel_km=L_vel_global_km)
    else:
        pack_f = _make_pack(ax_f, 9, draw_axes=True, draw_craft=True, draw_velocity=True, L_vel_km=L_vel_form_km)

    legend_elems = [
        Line2D([0], [0], color="0.45", lw=1.4, label="Chief"),
        Line2D([0], [0], color="tab:orange", lw=1.4, label="Deputy"),
        Line2D([0], [0], color="C0", lw=1.0, alpha=0.55, label="Earth"),
        Line2D([0], [0], color="tab:red", lw=1.0, alpha=0.45, label="Altitude KOZ"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.18)

    formation_overlays: list = []

    def _clear_formation_overlays() -> None:
        for art in formation_overlays:
            try:
                art.remove()
            except Exception:
                pass
        formation_overlays.clear()

    def _draw_formation_brt_koz(i: int) -> None:
        if ax_f is None:
            return
        _clear_formation_overlays()
        ri = int(i) % n_frames
        rci = r_c[ri]
        Ri = Rc[ri]
        if inner_koz_formation is not None:
            try:
                Xk, Yk, Zk = inner_koz_wireframe_eci_km(inner_koz_formation, rci, Ri, nu=30, nv=18)
                wf = ax_f.plot_wireframe(
                    Xk,
                    Yk,
                    Zk,
                    color="crimson",
                    linewidth=0.55,
                    alpha=0.88,
                    rstride=1,
                    cstride=1,
                )
                formation_overlays.append(wf)
            except Exception:
                pass
        if (
            brt_option1 is not None
            and hasattr(brt_option1, "value_batch")
            and states_lvlh is not None
        ):
            x6 = np.asarray(states_lvlh[ri], dtype=np.float64).reshape(6)
            lo6 = np.asarray(getattr(brt_option1, "domain_lo"), dtype=np.float64).reshape(6)
            hi6 = np.asarray(getattr(brt_option1, "domain_hi"), dtype=np.float64).reshape(6)
            r_dep = x6[:3]
            mrg = np.asarray(formation_brt_margin_m, dtype=np.float64).reshape(3)
            # BRT around chief (KOZ target); expand search until V brackets 0, clip to formation view.
            o_chief = np.zeros(3, dtype=np.float64)
            search_half = float(max(np.max(mrg), 2.5 * float(np.linalg.norm(r_dep)), 2200.0))
            display_r = float(max(np.linalg.norm(r_dep) + np.max(mrg), 800.0))
            nx, ny, nz = formation_brt_iso_resolution
            try:
                surf = extract_brt_v0_near_center(
                    brt_option1.value_batch,
                    o_chief,
                    float(x6[3]),
                    float(x6[4]),
                    float(x6[5]),
                    domain_lo=lo6,
                    domain_hi=hi6,
                    initial_half_m=max(
                        2200.0,
                        max(2.5 * float(np.max(inner_koz_formation.semi_axes)), 80.0)
                        if inner_koz_formation is not None
                        else 120.0,
                    ),
                    max_half_m=search_half,
                    iso_resolution=(int(nx), int(ny), int(nz)),
                    display_radius_m=display_r,
                    contour_half_m=display_r,
                    contour_n2d=48,
                )
            except Exception:
                surf = {
                    "mesh_verts": None,
                    "mesh_faces": None,
                    "footprint_polys": [],
                    "contour_lines": [],
                }
            for poly in surf.get("footprint_polys") or []:
                p3 = np.asarray(poly, dtype=np.float64)
                if p3.shape[0] < 3:
                    continue
                eci_fp = lvlh_points_m_to_eci_km(rci, Ri, p3)
                coll_fp = Poly3DCollection(
                    [eci_fp],
                    facecolors=(0.55, 0.15, 0.75, 0.14),
                    edgecolors=(0.4, 0.05, 0.55, 0.4),
                    linewidths=0.2,
                )
                ax_f.add_collection3d(coll_fp)
                formation_overlays.append(coll_fp)
            vm, fm = surf.get("mesh_verts"), surf.get("mesh_faces")
            if vm is not None and fm is not None and np.asarray(fm).size > 0:
                v_eci = lvlh_points_m_to_eci_km(rci, Ri, np.asarray(vm, dtype=np.float64))
                polys = [v_eci[fk] for fk in np.asarray(fm, dtype=np.int64)]
                coll = Poly3DCollection(
                    polys,
                    facecolors=(0.55, 0.15, 0.75, 0.18),
                    edgecolors=(0.38, 0.05, 0.52, 0.28),
                    linewidths=0.12,
                )
                ax_f.add_collection3d(coll)
                formation_overlays.append(coll)
            for line in surf.get("contour_lines") or []:
                ln = np.asarray(line, dtype=np.float64)
                if ln.shape[0] < 2:
                    continue
                eci_ln = lvlh_points_m_to_eci_km(rci, Ri, ln)
                pl = ax_f.plot(
                    eci_ln[:, 0],
                    eci_ln[:, 1],
                    eci_ln[:, 2],
                    color="mediumpurple",
                    linewidth=1.1,
                    alpha=0.85,
                )[0]
                formation_overlays.append(pl)

    def _set_arrows(lines: list, origin: np.ndarray, R: np.ndarray) -> None:
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        for j, ln in enumerate(lines):
            tip = o + L * Rm[:, j]
            ln.set_data_3d([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]])

    def _set_craft(ln, origin: np.ndarray, R: np.ndarray, edges: list) -> None:
        if ln is None:
            return
        xs, ys, zs = edges_to_nan_polyline(origin, R, edges)
        ln.set_data_3d(xs, ys, zs)

    def _set_velocity_arrow(ln, r_km: np.ndarray, v_km_s: np.ndarray, L_vel_km: float) -> None:
        if ln is None:
            return
        r = np.asarray(r_km, dtype=np.float64).reshape(3)
        v = np.asarray(v_km_s, dtype=np.float64).reshape(3)
        nv = float(np.linalg.norm(v))
        if nv < 1e-15:
            ln.set_data_3d([], [], [])
            return
        d = v / nv
        tip = r + float(L_vel_km) * d
        ln.set_data_3d([r[0], tip[0]], [r[1], tip[1]], [r[2], tip[2]])

    def _upd_pack(pack, i: int, *, formation_limits_ax) -> None:
        if pack is None:
            return
        pt_c, pt_d, chief_lns, dep_lns, ln_cc, ln_cd, vel_c, vel_d, L_vel_km = pack
        pt_c.set_data_3d([r_c[i, 0]], [r_c[i, 1]], [r_c[i, 2]])
        pt_d.set_data_3d([r_d[i, 0]], [r_d[i, 1]], [r_d[i, 2]])
        if len(chief_lns) >= 3:
            _set_arrows(chief_lns, r_c[i], Rc[i])
            _set_arrows(dep_lns, r_d[i], Rd[i])
        _set_craft(ln_cc, r_c[i], Rc[i], edges_chief_km)
        _set_craft(ln_cd, r_d[i], Rd[i], edges_dep_km)
        _set_velocity_arrow(vel_c, r_c[i], v_c_eph[i], L_vel_km)
        _set_velocity_arrow(vel_d, r_d[i], v_d_eph[i], L_vel_km)
        if formation_limits_ax is not None:
            mid = 0.5 * (r_c[i] + r_d[i])
            formation_limits_ax.set_xlim(mid[0] - half_form, mid[0] + half_form)
            formation_limits_ax.set_ylim(mid[1] - half_form, mid[1] + half_form)
            formation_limits_ax.set_zlim(mid[2] - half_form, mid[2] + half_form)

    def _collect_artists() -> tuple:
        out: list = []
        for pack in (pack_g, pack_f):
            if pack is None:
                continue
            for item in pack[:-1]:
                if item is None:
                    continue
                if isinstance(item, list):
                    out.extend(item)
                else:
                    out.append(item)
        return tuple(out)

    def init() -> tuple:
        i0 = 0
        _upd_pack(pack_g, i0, formation_limits_ax=None)
        _upd_pack(pack_f, i0, formation_limits_ax=ax_f)
        _draw_formation_brt_koz(i0)
        return _collect_artists()

    def update(i: int) -> tuple:
        i = int(i) % n_frames
        _upd_pack(pack_g, i, formation_limits_ax=None)
        _upd_pack(pack_f, i, formation_limits_ax=ax_f)
        _draw_formation_brt_koz(i)
        t = float(times[i])
        T = 2.0 * np.pi / n_rad_s
        ev_c = _earth_koz_violation(r_c[i], r_earth, h_koz)
        ev_d = _earth_koz_violation(r_d[i], r_earth, h_koz)
        sep = float(np.linalg.norm(r_d[i] - r_c[i]))
        warn = "  EARTH KOZ!" if (ev_c or ev_d) else ""
        brt_note = ""
        if brt_option1 is not None and states_lvlh is not None:
            try:
                u_brt = bool(brt_option1.is_unsafe(np.asarray(states_lvlh[i], dtype=np.float64).reshape(6)))
                brt_note = " | BRT unsafe" if u_brt else " | BRT safe"
            except Exception:
                pass
        fig.suptitle(
            f"t = {t/60:.1f} min | sep = {sep:.1f} km{warn}{brt_note}",
            fontsize=10,
            y=0.98,
        )
        if ax_f is not None:
            ax_f.set_title("Formation", fontsize=9)

        if frame_log_rows is not None:
            x6l = (
                np.asarray(states_lvlh[i], dtype=np.float64).reshape(6)
                if states_lvlh is not None
                else np.full(6, np.nan, dtype=np.float64)
            )
            row: dict[str, Any] = {
                "frame": int(i),
                "time_s": t,
                "x_m": float(x6l[0]),
                "y_m": float(x6l[1]),
                "z_m": float(x6l[2]),
                "vx_m_s": float(x6l[3]),
                "vy_m_s": float(x6l[4]),
                "vz_m_s": float(x6l[5]),
                "sep_km": sep,
                "earth_koz_violation_chief": int(ev_c),
                "earth_koz_violation_deputy": int(ev_d),
            }
            if inner_koz_formation is not None and states_lvlh is not None:
                row["inner_koz_shape"] = float(inner_koz_formation.shape_value(x6l[:3]))
                row["inner_koz_inside"] = int(inner_koz_formation.is_inside(x6l[:3]))
            else:
                row["inner_koz_shape"] = float("nan")
                row["inner_koz_inside"] = 0
            if brt_option1 is not None and states_lvlh is not None:
                try:
                    row["brt_value"] = float(brt_option1.value(x6l))
                    row["brt_unsafe"] = int(bool(brt_option1.is_unsafe(x6l)))
                except Exception:
                    row["brt_value"] = float("nan")
                    row["brt_unsafe"] = -1
            else:
                row["brt_value"] = float("nan")
                row["brt_unsafe"] = -1
            frame_log_rows.append(row)

        return _collect_artists()

    interval_ms = max(1, int(1000.0 / float(fps)))
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval_ms,
        blit=False,
    )

    written: str | None = None
    if output_path:
        output_path = os.path.abspath(output_path)
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".gif":
            anim.save(output_path, writer="pillow", fps=float(fps), dpi=dpi)
            written = output_path
        elif ext in (".mp4", ".mov", ".mkv"):
            try:
                anim.save(
                    output_path,
                    writer="ffmpeg",
                    fps=float(fps),
                    dpi=dpi,
                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                )
                written = output_path
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"ffmpeg export failed ({exc}); install ffmpeg or use .gif", UserWarning)
        else:
            raise ValueError("output_path must end with .gif or .mp4 (etc.)")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return written


def sample_uniform_times(duration_s: float, n_frames: int) -> np.ndarray:
    """Sample ``[0, duration_s)`` with ``n_frames`` evenly spaced instants (no wrap duplicate)."""
    duration_s = float(duration_s)
    n_frames = int(n_frames)
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")
    return np.linspace(0.0, duration_s, n_frames, endpoint=False, dtype=np.float64)


def sample_times_uniform(duration_s: float, fps: float) -> np.ndarray:
    """Deprecated convenience: use ``sample_uniform_times(duration, int(duration*fps))``."""
    duration_s = float(duration_s)
    n = max(2, int(np.ceil(duration_s * float(fps))))
    return sample_uniform_times(duration_s, n)
