"""Robust V=0 position-slice extraction for 3D BRT visualization (marching cubes + 2D fallbacks)."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np


def choose_contour_level(z: np.ndarray, preferred: float = 0.0) -> tuple[float, str]:
    """Pick an isovalue for plotting when ``preferred`` may not be bracketed on the slice.

    Coarse HJ grids often never hit 'V=0' even though the KOZ terminal set is 's-1'; then we
    draw a low-'V' envelope ('min_envelope') so the boundary is still visible.
    """
    finite = np.asarray(z, dtype=np.float64)[np.isfinite(z)]
    if finite.size == 0:
        return float(preferred), "empty"
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin <= preferred <= vmax:
        return float(preferred), "zero"
    if vmin > 0:
        return vmin + 0.06 * (vmax - vmin), "min_envelope"
    return vmax - 0.06 * (vmax - vmin), "max_envelope"


def sample_v_on_position_grid(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return '(xs, ys, zs, X, Y, Z, vol)' with 'vol' shape '(nx, ny, nz)'."""
    xs = np.linspace(float(x_lo), float(x_hi), int(nx), dtype=np.float64)
    ys = np.linspace(float(y_lo), float(y_hi), int(ny), dtype=np.float64)
    zs = np.linspace(float(z_lo), float(z_hi), int(nz), dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            np.full(X.size, float(vx)),
            np.full(X.size, float(vy)),
            np.full(X.size, float(vz)),
        ],
        axis=-1,
    )
    vol = np.asarray(value_on_grid(pts), dtype=np.float64).reshape(nx, ny, nz)
    return xs, ys, zs, X, Y, Z, vol


def position_box_brackets_level(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    vx: float,
    vy: float,
    vz: float,
    *,
    level: float = 0.0,
    initial_half_m: float,
    max_half_m: float,
    domain_lo: np.ndarray | None = None,
    domain_hi: np.ndarray | None = None,
    probe_n: int = 9,
) -> tuple[float, float, float] | None:
    """Expand a cube around 'center_m' until 'V' on the probe grid brackets 'level'.

    Returns '(x_lo, x_hi, half_used)' for a symmetric cube, or 'None' if no bracket by 'max_half_m'.
    """
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    half = float(max(initial_half_m, 1.0))
    max_half = float(max(max_half_m, half))
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6) if domain_lo is not None else None
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6) if domain_hi is not None else None

    while half <= max_half * (1.0 + 1e-9):
        x_lo = float(c[0] - half)
        x_hi = float(c[0] + half)
        y_lo = float(c[1] - half)
        y_hi = float(c[1] + half)
        z_lo = float(c[2] - half)
        z_hi = float(c[2] + half)
        if lo6 is not None and hi6 is not None:
            x_lo = max(x_lo, float(lo6[0]))
            x_hi = min(x_hi, float(hi6[0]))
            y_lo = max(y_lo, float(lo6[1]))
            y_hi = min(y_hi, float(hi6[1]))
            z_lo = max(z_lo, float(lo6[2]))
            z_hi = min(z_hi, float(hi6[2]))
        if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
            half *= 1.45
            continue
        _, _, _, _, _, _, vol = sample_v_on_position_grid(
            value_on_grid, vx, vy, vz, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, probe_n, probe_n, probe_n
        )
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            half *= 1.45
            continue
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin <= level <= vmax:
            return x_lo, x_hi, half
        half *= 1.45
    return None


def decimate_mesh_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample faces so matplotlib 3D export stays responsive."""
    f = np.asarray(faces, dtype=np.int64)
    if f.shape[0] <= int(max_faces):
        return np.asarray(verts, dtype=np.float64), f
    step = max(1, int(np.ceil(f.shape[0] / float(max_faces))))
    return np.asarray(verts, dtype=np.float64), f[::step]


def _nearest_axis_index(axis: np.ndarray, target: float) -> int:
    a = np.asarray(axis, dtype=np.float64).reshape(-1)
    return int(np.argmin(np.abs(a - float(target))))


def _binary_close_unsafe(unsafe: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Fill 1-cell gaps so marching-cubes yields a watertight shell on a coarse grid."""
    if iterations <= 0:
        return np.asarray(unsafe, dtype=bool)
    try:
        from scipy import ndimage
    except ImportError:
        return np.asarray(unsafe, dtype=bool)
    structure = np.ones((3, 3, 3), dtype=bool)
    closed = np.asarray(unsafe, dtype=bool)
    for _ in range(int(iterations)):
        closed = ndimage.binary_closing(closed, structure=structure)
    return closed


def _march_brt_shell_from_volume(
    field: np.ndarray,
    axes_xyz: list[np.ndarray],
    *,
    level: float = 0.0,
    binary_indicator: bool = False,
    pad_voxels: int = 1,
    max_faces: int = 12000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Marching cubes on a crop around the sub-level set (avoids open cuts at the HJ domain box)."""
    vol = np.asarray(field, dtype=np.float64)
    if vol.ndim != 3:
        return None
    if binary_indicator:
        inside = vol > 0.5
        march_level = 0.5
    else:
        inside = vol <= float(level)
        march_level = float(level)
    if not np.any(inside) or np.all(inside):
        return None
    idx = np.argwhere(inside)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    pad = max(0, int(pad_voxels))
    slices = tuple(
        slice(max(0, int(lo[d]) - pad), min(vol.shape[d], int(hi[d]) + pad + 1)) for d in range(3)
    )
    sub = vol[slices]
    sub_axes = [np.asarray(axes_xyz[d][slices[d]], dtype=np.float64) for d in range(3)]
    dx = float(sub_axes[0][1] - sub_axes[0][0]) if sub_axes[0].size > 1 else 1.0
    dy = float(sub_axes[1][1] - sub_axes[1][0]) if sub_axes[1].size > 1 else 1.0
    dz = float(sub_axes[2][1] - sub_axes[2][0]) if sub_axes[2].size > 1 else 1.0
    try:
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(sub, level=march_level, spacing=(dx, dy, dz))
    except Exception:
        return None
    origin = np.array([sub_axes[0][0], sub_axes[1][0], sub_axes[2][0]], dtype=np.float64)
    return decimate_mesh_faces(verts + origin, faces, max_faces=max_faces)


def _mesh_watertight_edge_count(faces: np.ndarray) -> int:
    from collections import Counter

    edges: Counter[tuple[int, int]] = Counter()
    for tri in np.asarray(faces, dtype=np.int64):
        for i in range(3):
            e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edges[e] += 1
    return sum(1 for c in edges.values() if c != 2)


def _matplotlib_xy_contours(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    field: np.ndarray,
    level: float,
    z_fix: float,
) -> list[np.ndarray]:
    """``V=level`` polylines on the x-y plane (works without scikit-image)."""
    import matplotlib.pyplot as plt

    U, V = np.meshgrid(x_coords, y_coords, indexing="ij")
    fig, ax = plt.subplots(figsize=(4, 3))
    lines: list[np.ndarray] = []
    try:
        cs = ax.contour(U, V, field, levels=[float(level)])
        for seg_list in cs.allsegs:
            for seg in seg_list:
                seg = np.asarray(seg, dtype=np.float64)
                if seg.shape[0] < 2:
                    continue
                ln = np.zeros((seg.shape[0], 3), dtype=np.float64)
                ln[:, 0] = seg[:, 0]
                ln[:, 1] = seg[:, 1]
                ln[:, 2] = float(z_fix)
                lines.append(ln)
    except Exception:
        pass
    finally:
        plt.close(fig)
    return lines


def extract_brt_from_hj_native(
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    grid_shape: tuple[int, ...],
    values_final: np.ndarray,
    center_m: np.ndarray,
    vx: float,
    vy: float,
    vz: float,
    *,
    display_radius_m: float,
    max_draw_faces: int = 8000,
) -> dict[str, Any]:
    """Fast BRT shell from stored HJ grid nodes (no dense re-interpolation)."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    shape = tuple(int(x) for x in grid_shape)
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    axes = [
        np.linspace(lo6[d], hi6[d], shape[d], endpoint=True, dtype=np.float64) for d in range(6)
    ]
    vol = np.asarray(values_final, dtype=np.float64).reshape(shape)
    ivx = _nearest_axis_index(axes[3], vx)
    ivy = _nearest_axis_index(axes[4], vy)
    ivz = _nearest_axis_index(axes[5], vz)
    iz = _nearest_axis_index(axes[2], float(c[2]))
    pos_vol = vol[:, :, :, ivx, ivy, ivz]
    unsafe = pos_vol <= 0.0
    dx = float(axes[0][1] - axes[0][0]) if shape[0] > 1 else 1.0
    dy = float(axes[1][1] - axes[1][0]) if shape[1] > 1 else 1.0
    dz = float(axes[2][1] - axes[2][0]) if shape[2] > 1 else 1.0

    mesh_verts: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    slice_loops_3d: list[np.ndarray] = []
    closing_iters = int(os.environ.get("BRT_SNAPSHOT_MESH_CLOSING", "1"))
    if np.any(unsafe) and not np.all(unsafe):
        unsafe_mc = _binary_close_unsafe(unsafe, closing_iters)
        mc = _march_brt_shell_from_volume(
            unsafe_mc.astype(np.float64),
            axes[:3],
            level=0.5,
            binary_indicator=True,
            pad_voxels=1,
            max_faces=max_draw_faces,
        )
        if mc is not None:
            mesh_verts, mesh_faces = mc

    # x-y slice works with matplotlib alone; 3D shell needs marching cubes (scikit-image).
    # Without it, stack BRT contours on every native z grid plane so the 3D view still shows structure.
    xy_slice = pos_vol[:, :, iz]
    level_use, level_tag = choose_contour_level(xy_slice, 0.0)
    if int(unsafe.sum()) < 8:
        finite_sl = xy_slice[np.isfinite(xy_slice)]
        if finite_sl.size > 0:
            vmin_sl, vmax_sl = float(np.min(finite_sl)), float(np.max(finite_sl))
            if vmax_sl > vmin_sl:
                level_use = vmin_sl + 0.06 * (vmax_sl - vmin_sl)
                level_tag = "min_envelope"
    if mesh_verts is None:
        for k, z_coord in enumerate(axes[2]):
            sl = pos_vol[:, :, k]
            if not np.any(np.isfinite(sl)):
                continue
            slice_loops_3d.extend(_matplotlib_xy_contours(axes[0], axes[1], sl, level_use, float(z_coord)))

    contour_lines: list[np.ndarray] = []
    z_fix = float(axes[2][iz])
    if np.any(np.isfinite(xy_slice)):
        contour_lines = _matplotlib_xy_contours(axes[0], axes[1], xy_slice, level_use, z_fix)

    # Filled x-y patches read as a flat "cutting plane" in 3D when a shell is already drawn.
    footprint_polys: list[np.ndarray] = []
    envelope_polys: list[np.ndarray] = []
    if mesh_verts is None:
        unsafe_xy = unsafe[:, :, iz].astype(np.float64)
        if int(unsafe.sum()) >= 3:
            footprint_polys = [
                ln
                for ln in _matplotlib_xy_contours(axes[0], axes[1], unsafe_xy, 0.5, z_fix)
                if ln.shape[0] >= 3
            ]
        if not footprint_polys and contour_lines:
            envelope_polys = [ln for ln in contour_lines if ln.shape[0] >= 3]

    view_half = float(display_radius_m)
    if mesh_verts is not None and mesh_verts.size > 0:
        view_half = max(view_half, float(np.max(np.linalg.norm(mesh_verts - c, axis=1))) * 1.05)
    elif footprint_polys or envelope_polys:
        pts = np.vstack((footprint_polys or []) + (envelope_polys or []))
        view_half = max(view_half, float(np.max(np.linalg.norm(pts[:, :2] - c[:2], axis=1))) * 1.05)
    elif contour_lines:
        pts = np.vstack(contour_lines)
        view_half = max(view_half, float(np.max(np.linalg.norm(pts[:, :2] - c[:2], axis=1))) * 1.05)
    elif slice_loops_3d:
        pts = np.vstack(slice_loops_3d)
        view_half = max(view_half, float(np.max(np.linalg.norm(pts - c, axis=1))) * 1.05)

    tag = "hj_native_unsafe" if mesh_verts is not None else "hj_native_footprint_only"
    if mesh_verts is None and not footprint_polys and not envelope_polys and not slice_loops_3d:
        tag = "hj_native_empty"
    elif mesh_verts is None and envelope_polys and level_tag != "zero":
        tag = f"hj_native_{level_tag}"
    elif mesh_verts is None and slice_loops_3d:
        tag = f"hj_native_stacked_slices_{level_tag}"

    bounds_lo = bounds_hi = None
    if mesh_verts is not None and mesh_verts.size > 0:
        bounds_lo = np.min(mesh_verts, axis=0)
        bounds_hi = np.max(mesh_verts, axis=0)

    return {
        "mesh_verts": mesh_verts,
        "mesh_faces": mesh_faces,
        "footprint_polys": footprint_polys if footprint_polys else envelope_polys,
        "slice_loops_3d": slice_loops_3d,
        "contour_lines": contour_lines,
        "bracket_half_m": None,
        "view_half_m": view_half,
        "mesh_bounds_lo": bounds_lo,
        "mesh_bounds_hi": bounds_hi,
        "contour_level_tag": tag,
        "contour_level": float(level_use),
        "contour_level_name": level_tag,
        "mesh_backend": "marching_cubes" if mesh_verts is not None else ("stacked_slices" if slice_loops_3d else "none"),
    }


def unsafe_set_mesh_lvlh(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    vx: float,
    vy: float,
    vz: float,
    box_half_m: float,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    grid_n: tuple[int, int, int],
    *,
    box_half_xyz: tuple[float, float, float] | None = None,
    unsafe_level: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Marching-cubes mesh of the **physical BRT**: '{ x : V(x) <= unsafe_level }' in position at fixed 'v'.

    Unlike an isosurface at 'V=0' when the whole local box is negative, this always produces a
    boundary when both safe and unsafe cells exist.
    """
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    if box_half_xyz is not None:
        hx, hy, hz = (float(x) for x in box_half_xyz)
    else:
        h = float(box_half_m)
        hx = hy = hz = h
    x_lo = max(float(lo6[0]), float(c[0] - hx))
    x_hi = min(float(hi6[0]), float(c[0] + hx))
    y_lo = max(float(lo6[1]), float(c[1] - hy))
    y_hi = min(float(hi6[1]), float(c[1] + hy))
    z_lo = max(float(lo6[2]), float(c[2] - hz))
    z_hi = min(float(hi6[2]), float(c[2] + hz))
    if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
        return None
    nx, ny, nz = (max(8, int(grid_n[i])) for i in range(3))
    _, _, _, _, _, _, vol = sample_v_on_position_grid(
        value_on_grid, vx, vy, vz, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, nx, ny, nz
    )
    finite = np.isfinite(vol)
    if not np.any(finite):
        return None
    inside = vol <= float(unsafe_level)
    if not np.any(inside) or np.all(inside):
        return None
    indicator = inside.astype(np.float64)
    xs = np.linspace(x_lo, x_hi, nx, dtype=np.float64)
    ys = np.linspace(y_lo, y_hi, ny, dtype=np.float64)
    zs = np.linspace(z_lo, z_hi, nz, dtype=np.float64)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0
    try:
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(indicator, level=0.5, spacing=(dx, dy, dz))
    except Exception:
        return None
    origin = np.array([x_lo, y_lo, z_lo], dtype=np.float64)
    return verts + origin, faces


def xy_unsafe_footprint_at_z(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    z_m: float,
    vx: float,
    vy: float,
    vz: float,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    n2d: int,
    *,
    center_m: np.ndarray | None = None,
    half_xy_m: tuple[float, float] | None = None,
    unsafe_level: float = 0.0,
) -> list[np.ndarray]:
    """Closed polygons (N,3) in LVLH for the x-y slice of '{V <= unsafe_level}' at 'z=z_m'."""
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    if center_m is not None and half_xy_m is not None:
        c = np.asarray(center_m, dtype=np.float64).reshape(3)
        hx, hy = float(half_xy_m[0]), float(half_xy_m[1])
        u = np.linspace(max(lo6[0], c[0] - hx), min(hi6[0], c[0] + hx), int(n2d), dtype=np.float64)
        v = np.linspace(max(lo6[1], c[1] - hy), min(hi6[1], c[1] + hy), int(n2d), dtype=np.float64)
    else:
        u = np.linspace(float(lo6[0]), float(hi6[0]), int(n2d), dtype=np.float64)
        v = np.linspace(float(lo6[1]), float(hi6[1]), int(n2d), dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.stack(
        [
            U.ravel(),
            V.ravel(),
            np.full(U.size, float(z_m)),
            np.full(U.size, float(vx)),
            np.full(U.size, float(vy)),
            np.full(U.size, float(vz)),
        ],
        axis=-1,
    )
    Z = np.asarray(value_on_grid(pts), dtype=np.float64).reshape(U.shape)
    if not np.any(np.isfinite(Z)):
        return []
    unsafe = (Z <= float(unsafe_level)).astype(np.float64)
    if not np.any(unsafe):
        return []
    polys: list[np.ndarray] = []
    try:
        from skimage.measure import find_contours

        for row_col in find_contours(unsafe, 0.5):
            if row_col.shape[0] < 3:
                continue
            # find_contours returns (row, col) = (i_u, i_v) for array[i,j]
            poly3 = np.zeros((row_col.shape[0], 3), dtype=np.float64)
            poly3[:, 0] = u[0] + row_col[:, 0] * (u[-1] - u[0]) / max(len(u) - 1, 1)
            poly3[:, 1] = v[0] + row_col[:, 1] * (v[-1] - v[0]) / max(len(v) - 1, 1)
            poly3[:, 2] = float(z_m)
            polys.append(poly3)
    except Exception:
        pass
    return polys


def marching_cubes_v0_lvlh(
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
    """'(verts_m, faces)' for 'V=level' via marching cubes, or 'None' if no bracket / library failure."""
    nx, ny, nz = max(4, int(nx)), max(4, int(ny)), max(4, int(nz))
    xs, ys, zs, _, _, _, vol = sample_v_on_position_grid(
        value_on_grid, vx, vy, vz, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, nx, ny, nz
    )
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return None
    level_use, _ = choose_contour_level(vol, level)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0
    try:
        from skimage.measure import marching_cubes

        verts, faces, _, _ = marching_cubes(vol, level=level_use, spacing=(dx, dy, dz))
    except Exception:
        return None
    origin = np.array([x_lo, y_lo, z_lo], dtype=np.float64)
    return verts + origin, faces


def filter_mesh_in_lvhl_box(
    verts: np.ndarray,
    faces: np.ndarray,
    center_m: np.ndarray,
    half_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep triangles whose centroid lies inside 'center +/- half_xyz' (LVLH m)."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    hx, hy, hz = (float(x) for x in half_xyz)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return verts, faces
    centroids = verts[faces].mean(axis=1)
    inside = (
        (np.abs(centroids[:, 0] - c[0]) <= hx)
        & (np.abs(centroids[:, 1] - c[1]) <= hy)
        & (np.abs(centroids[:, 2] - c[2]) <= hz)
    )
    return verts, faces[inside]


def filter_mesh_by_radius(
    verts: np.ndarray,
    faces: np.ndarray,
    center_m: np.ndarray,
    max_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop triangles whose centroid is farther than 'max_radius_m' from 'center_m'."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return verts, faces
    centroids = verts[faces].mean(axis=1)
    d = np.linalg.norm(centroids - c.reshape(1, 3), axis=1)
    keep = d <= float(max_radius_m)
    faces = faces[keep]
    return verts, faces


def _polylines_from_contour_set(
    cs: Any,
    a0: int,
    a1: int,
    normal_axis: int,
    fixed_coord: float,
) -> list[np.ndarray]:
    """Extract 3D LVLH polylines from a matplotlib 'ContourSet' (mpl 3.7-3.9+)."""
    segs: list[np.ndarray] = []
    # Matplotlib 3.8+: QuadContourSet exposes allsegs instead of collections.
    if hasattr(cs, "allsegs"):
        for level_segs in cs.allsegs:
            for xy in level_segs:
                xy = np.asarray(xy, dtype=np.float64)
                if xy.ndim != 2 or xy.shape[0] < 2:
                    continue
                line = np.zeros((xy.shape[0], 3), dtype=np.float64)
                line[:, a0] = xy[:, 0]
                line[:, a1] = xy[:, 1]
                line[:, normal_axis] = float(fixed_coord)
                segs.append(line)
        return segs
    if hasattr(cs, "collections"):
        for coll in cs.collections:
            for path in coll.get_paths():
                xy = path.vertices
                if xy.size == 0:
                    continue
                line = np.zeros((xy.shape[0], 3), dtype=np.float64)
                line[:, a0] = xy[:, 0]
                line[:, a1] = xy[:, 1]
                line[:, normal_axis] = float(fixed_coord)
                segs.append(line)
    return segs


def polylines_within_radius(
    lines: list[np.ndarray],
    center_m: np.ndarray,
    radius_m: float,
    *,
    min_points: int = 2,
) -> list[np.ndarray]:
    """Keep polyline vertices inside 'radius_m' of 'center_m' (split into contiguous runs)."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    r = float(radius_m)
    out: list[np.ndarray] = []
    for line in lines:
        ln = np.asarray(line, dtype=np.float64)
        if ln.ndim != 2 or ln.shape[0] < min_points:
            continue
        d = np.linalg.norm(ln - c.reshape(1, 3), axis=1)
        inside = d <= r
        if not np.any(inside):
            continue
        idx = np.where(inside)[0]
        # Contiguous runs of in-radius points
        start = idx[0]
        prev = idx[0]
        for k in idx[1:]:
            if k == prev + 1:
                prev = k
                continue
            if prev - start + 1 >= min_points:
                out.append(ln[start : prev + 1].copy())
            start = k
            prev = k
        if prev - start + 1 >= min_points:
            out.append(ln[start : prev + 1].copy())
    return out


def nearest_contour_distance(lines: list[np.ndarray], center_m: np.ndarray) -> float | None:
    """Minimum distance from 'center_m' to any point on the given polylines."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    best: float | None = None
    for line in lines:
        ln = np.asarray(line, dtype=np.float64)
        if ln.size == 0:
            continue
        d = float(np.min(np.linalg.norm(ln - c.reshape(1, 3), axis=1)))
        best = d if best is None else min(best, d)
    return best


def contour_segments_on_domain_plane(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    normal_axis: int,
    fixed_coord: float,
    vx: float,
    vy: float,
    vz: float,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    n2d: int,
    *,
    level: float = 0.0,
) -> list[np.ndarray]:
    """'V=level' contours on a plane using the full HJ position bounds for the two free axes."""
    import matplotlib.pyplot as plt

    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    a0, a1 = [i for i in range(3) if i != normal_axis]
    u = np.linspace(float(lo6[a0]), float(hi6[a0]), int(n2d), dtype=np.float64)
    v = np.linspace(float(lo6[a1]), float(hi6[a1]), int(n2d), dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.zeros((U.size, 6), dtype=np.float64)
    pos = np.tile(c.reshape(1, 3), (U.size, 1))
    pos[:, a0] = U.ravel()
    pos[:, a1] = V.ravel()
    pos[:, normal_axis] = float(fixed_coord)
    pts[:, :3] = pos
    pts[:, 3] = float(vx)
    pts[:, 4] = float(vy)
    pts[:, 5] = float(vz)
    for d in range(3):
        pts[:, d] = np.clip(pts[:, d], lo6[d], hi6[d])
    Z = np.asarray(value_on_grid(pts), dtype=np.float64).reshape(U.shape)
    if not np.any(np.isfinite(Z)):
        return []
    level_use, level_tag = choose_contour_level(Z, level)
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        cs = ax.contour(U, V, Z, levels=[level_use])
        segs = _polylines_from_contour_set(cs, a0, a1, normal_axis, float(fixed_coord))
    except Exception:
        segs = []
    finally:
        plt.close(fig)
    return segs


def contour_segments_3d_from_plane(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    normal_axis: int,
    fixed_coord: float,
    vx: float,
    vy: float,
    vz: float,
    half_m: float,
    n2d: int,
    *,
    level: float = 0.0,
    domain_lo: np.ndarray | None = None,
    domain_hi: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Plane contours of 'V=level' as polylines '(N, 3)' in LVLH meters (for 'plot3D').

    'normal_axis': 0 fixes 'x', 1 fixes 'y', 2 fixes 'z' to 'fixed_coord'.
    """
    import matplotlib.pyplot as plt

    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    h = float(half_m)
    a0, a1 = [i for i in range(3) if i != normal_axis]
    u = np.linspace(c[a0] - h, c[a0] + h, int(n2d), dtype=np.float64)
    v = np.linspace(c[a1] - h, c[a1] + h, int(n2d), dtype=np.float64)
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.zeros((U.size, 6), dtype=np.float64)
    pos = np.stack([c.copy()] * U.size)
    pos[:, a0] = U.ravel()
    pos[:, a1] = V.ravel()
    pos[:, normal_axis] = float(fixed_coord)
    pts[:, :3] = pos
    pts[:, 3] = float(vx)
    pts[:, 4] = float(vy)
    pts[:, 5] = float(vz)
    if domain_lo is not None and domain_hi is not None:
        lo = np.asarray(domain_lo, dtype=np.float64).reshape(6)
        hi = np.asarray(domain_hi, dtype=np.float64).reshape(6)
        for d in range(3):
            pts[:, d] = np.clip(pts[:, d], lo[d], hi[d])
    Z = np.asarray(value_on_grid(pts), dtype=np.float64).reshape(U.shape)
    if not np.any(np.isfinite(Z)):
        return []
    level_use, _ = choose_contour_level(Z, level)
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        cs = ax.contour(U, V, Z, levels=[level_use])
        segs = _polylines_from_contour_set(cs, a0, a1, normal_axis, fixed_coord)
    except Exception:
        segs = []
    finally:
        plt.close(fig)
    return segs


def _box_unsafe_fraction(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    vx: float,
    vy: float,
    vz: float,
    box_half_m: float,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    n: int = 9,
) -> float | None:
    """Fraction of grid nodes with 'V <= 0' in a position cube (quick probe)."""
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    h = float(box_half_m)
    x_lo = max(float(lo6[0]), float(c[0] - h))
    x_hi = min(float(hi6[0]), float(c[0] + h))
    y_lo = max(float(lo6[1]), float(c[1] - h))
    y_hi = min(float(hi6[1]), float(c[1] + h))
    z_lo = max(float(lo6[2]), float(c[2] - h))
    z_hi = min(float(hi6[2]), float(c[2] + h))
    if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
        return None
    nn = max(5, int(n))
    xs = np.linspace(x_lo, x_hi, nn, dtype=np.float64)
    ys = np.linspace(y_lo, y_hi, nn, dtype=np.float64)
    zs = np.linspace(z_lo, z_hi, nn, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            np.full(X.size, float(vx)),
            np.full(X.size, float(vy)),
            np.full(X.size, float(vz)),
        ],
        axis=-1,
    )
    V = np.asarray(value_on_grid(pts), dtype=np.float64)
    finite = np.isfinite(V)
    if not np.any(finite):
        return None
    return float(np.mean(V[finite] <= 0.0))


def _parse_snapshot_half_xyz(display_radius_m: float) -> tuple[float, float, float]:
    """Anisotropic LVLH half-extents (x, y along-track, z) for BRT mesh near chief."""
    env = os.environ.get("BRT_SNAPSHOT_HALF_XYZ", "").strip()
    if env:
        parts = [float(x.strip()) for x in env.split(",") if x.strip()]
        if len(parts) == 3:
            return tuple(parts)
    dr = float(display_radius_m)
    return (max(160.0, 1.2 * dr), max(900.0, 12.0 * dr), max(140.0, 1.1 * dr))


def extract_brt_v0_near_center(
    value_on_grid: Callable[[np.ndarray], np.ndarray],
    center_m: np.ndarray,
    vx: float,
    vy: float,
    vz: float,
    *,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    initial_half_m: float,
    max_half_m: float,
    iso_resolution: tuple[int, int, int],
    display_radius_m: float,
    contour_half_m: float,
    contour_n2d: int = 80,
) -> dict[str, Any]:
    """Physical BRT near ``center_m``: 3D shell of ``{V <= 0}`` plus x-y footprint at chief ``z``.

    The inner KOZ is only the **terminal** set; the BRT is the backward reachable unsafe set and is
    generally much larger. This routine marches the **unsafe-set boundary** (not only ``V=0`` when
    the local cube is entirely negative).
    """
    lo6 = np.asarray(domain_lo, dtype=np.float64).reshape(6)
    hi6 = np.asarray(domain_hi, dtype=np.float64).reshape(6)
    c = np.asarray(center_m, dtype=np.float64).reshape(3)
    cap = int(os.environ.get("BRT_SNAPSHOT_ISO_MAX", "32"))
    nx, ny, nz = (min(cap, max(12, int(iso_resolution[i]))) for i in range(3))
    n2 = min(48, int(contour_n2d))
    half_xyz = _parse_snapshot_half_xyz(display_radius_m)
    mesh_radius_m = float(os.environ.get("BRT_SNAPSHOT_MESH_RADIUS_M", str(max(half_xyz) * 1.05)))
    display_half_env = os.environ.get("BRT_SNAPSHOT_DISPLAY_HALF_XYZ", "").strip()
    display_half_xyz = half_xyz
    if display_half_env:
        parts = [float(x.strip()) for x in display_half_env.split(",") if x.strip()]
        if len(parts) == 3:
            display_half_xyz = tuple(parts)
    fp_half = (
        float(os.environ.get("BRT_SNAPSHOT_FOOTPRINT_HALF_X_M", str(half_xyz[0]))),
        float(os.environ.get("BRT_SNAPSHOT_FOOTPRINT_HALF_Y_M", str(half_xyz[1]))),
    )

    mesh_verts: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    half_used: float | None = None
    half_try = float(max(initial_half_m, max(half_xyz)))
    max_h = float(max_half_m)
    while half_try <= max_h * (1.0 + 1e-9):
        scale = half_try / max(half_xyz)
        try_xyz = tuple(float(h) * scale for h in half_xyz)
        frac = _box_unsafe_fraction(value_on_grid, c, vx, vy, vz, half_try, lo6, hi6)
        iso = unsafe_set_mesh_lvlh(
            value_on_grid,
            c,
            vx,
            vy,
            vz,
            half_try,
            lo6,
            hi6,
            (nx, ny, nz),
            box_half_xyz=try_xyz,
        )
        if iso is not None:
            vm, fm = iso
            vm, fm = filter_mesh_by_radius(vm, fm, c, mesh_radius_m)
            vm, fm = filter_mesh_in_lvhl_box(vm, fm, c, display_half_xyz)
            if fm.size == 0:
                half_try = min(max_h, half_try * 1.45)
                continue
            max_f = int(os.environ.get("BRT_SNAPSHOT_MAX_FACES", "8000"))
            mesh_verts, mesh_faces = decimate_mesh_faces(vm, fm, max_faces=max_f)
            half_used = half_try
            break
        if frac is not None and (frac <= 1e-6 or frac >= 1.0 - 1e-6):
            half_try = min(max_h, half_try * 1.45)
            continue
        half_try = min(max_h, half_try * 1.45)

    footprint_polys = xy_unsafe_footprint_at_z(
        value_on_grid,
        float(c[2]),
        vx,
        vy,
        vz,
        lo6,
        hi6,
        n2,
        center_m=c,
        half_xy_m=fp_half,
    )

    contour_lines: list[np.ndarray] = []
    if os.environ.get("BRT_SNAPSHOT_V0_CONTOUR", "0").lower() in ("1", "true", "yes"):
        contour_lines = contour_segments_on_domain_plane(
            value_on_grid,
            c,
            2,
            float(c[2]),
            vx,
            vy,
            vz,
            lo6,
            hi6,
            n2,
            level=0.0,
        )

    view_half = float(display_radius_m)
    if mesh_verts is not None and mesh_verts.size > 0:
        view_half = max(view_half, float(np.max(np.linalg.norm(mesh_verts - c, axis=1))) * 1.08)
    elif footprint_polys:
        pts = np.vstack(footprint_polys)
        view_half = max(view_half, float(np.max(np.linalg.norm(pts[:, :2] - c[:2], axis=1))) * 1.08)
    elif contour_lines:
        d_near = nearest_contour_distance(contour_lines, c)
        if d_near is not None and np.isfinite(d_near):
            view_half = max(view_half, 1.12 * float(d_near))

    contour_tag = "unsafe_V_le_0" if mesh_verts is not None else "no_shell"
    if mesh_verts is None and footprint_polys:
        contour_tag = "footprint_only"

    bounds_lo = bounds_hi = None
    if mesh_verts is not None and mesh_verts.size > 0:
        bounds_lo = np.min(mesh_verts, axis=0)
        bounds_hi = np.max(mesh_verts, axis=0)

    return {
        "mesh_verts": mesh_verts,
        "mesh_faces": mesh_faces,
        "footprint_polys": footprint_polys,
        "contour_lines": contour_lines,
        "bracket_half_m": half_used,
        "view_half_m": view_half,
        "mesh_bounds_lo": bounds_lo,
        "mesh_bounds_hi": bounds_hi,
        "contour_level_tag": contour_tag,
    }
