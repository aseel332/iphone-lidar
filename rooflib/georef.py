"""Georeference the ARKit mesh onto the DSM/mask rasters.

The problem is 4-DOF, not 7-DOF: ARKit's +Y is gravity-up (measured here to
0.66 deg), and ARKit is metric. So we solve yaw, translation (E, N, Z) and
optionally a small scale correction.

Two failure modes drive the design:

1. The footprint is nearly point-symmetric, so binary overlap cannot tell
   yaw from yaw+180 (0.978 vs 0.972 on this dataset -- useless). The
   superstructure HEIGHT map separates them cleanly (+0.844 vs -0.434).
   `solve_yaw` therefore always scores both channels and reports the margin.

2. Mapping ARKit to ENU with the wrong chirality silently mirrors the model.
   `AXIS_MAP` below is the right-handed choice and `handedness_check` proves
   it against the data rather than assuming it.
"""
from __future__ import annotations

import numpy as np

# ARKit (x right, y up, z toward viewer) -> ENU-before-yaw (east, north, up).
# east = +x, north = -z, up = +y.  Determinant +1, so chirality is preserved.
AXIS_MAP = np.array([[1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0],
                     [0.0, 1.0, 0.0]])


def estimate_deck_level(mesh, bin_size=0.25, up_thresh=0.98):
    """Find the walking surface in the ARKit frame.

    On a rooftop scan the deck is the *lowest* large horizontal surface --
    everything above it is parapet, stair headroom and tanks. Picking the
    largest horizontal band alone is not enough on buildings with a big upper
    terrace, so we take the largest band and then prefer any comparably large
    band below it.
    """
    N = mesh.face_normals()
    A = mesh.face_areas()
    C = mesh.face_centroids()
    up = N[:, 1] > up_thresh
    if up.sum() == 0:
        raise ValueError("no horizontal faces found")
    y = C[up, 1]
    w = A[up]
    lo, hi = y.min(), y.max()
    edges = np.arange(lo, hi + bin_size, bin_size)
    hist, _ = np.histogram(y, bins=edges, weights=w)
    biggest = hist.max()
    idx = [i for i, v in enumerate(hist) if v >= 0.35 * biggest]
    pick = min(idx)  # lowest band that is still a major surface
    sel = (y >= edges[pick] - bin_size) & (y < edges[pick + 1] + bin_size)
    deck = float(np.average(y[sel], weights=w[sel]))
    return deck, {
        "deck_y_local": deck,
        "deck_area_m2": float(w[sel].sum()),
        "bands": [{"y": float(edges[i]), "area_m2": float(hist[i])}
                  for i in range(len(hist)) if hist[i] > 0.5],
    }


def deck_normal(mesh, deck_y, tol=0.15, up_thresh=0.98):
    """Area-weighted normal of the deck -- our gravity witness."""
    N = mesh.face_normals()
    A = mesh.face_areas()
    C = mesh.face_centroids()
    sel = (N[:, 1] > up_thresh) & (np.abs(C[:, 1] - deck_y) < tol)
    n = (A[sel][:, None] * N[sel]).sum(0)
    n /= np.linalg.norm(n)
    return n, float(np.degrees(np.arccos(np.clip(n[1], -1, 1)))), float(A[sel].sum())


def rot_z(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_matrix(yaw_deg, scale, translation, pivot=(0.0, 0.0, 0.0),
                 tilt_correction=None):
    """Compose the 4x4 ARKit -> ENU transform.

        p_enu = scale * Rz(yaw) @ (AXIS_MAP @ p_arkit - pivot) + translation

    `pivot` lives in the post-AXIS_MAP, pre-yaw frame: (mean east, mean north,
    deck height) of the mesh. Solving around it keeps yaw and translation
    decoupled during optimisation, and makes `translation` directly readable
    as "where the deck centre sits in the world".
    """
    R = rot_z(yaw_deg) @ AXIS_MAP
    if tilt_correction is not None:
        R = tilt_correction @ R
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = np.asarray(translation, float) - scale * (rot_z(yaw_deg) @ np.asarray(pivot, float))
    return M


def apply(M, pts):
    return pts @ M[:3, :3].T + M[:3, 3]


def apply_normals(M, normals):
    """Transform normals.

    Points here use the row-vector convention (`apply` does `pts @ A.T`), so a
    normal maps as  n_row -> n_row @ inv(A)  -- NOT `@ inv(A).T`, which applies
    the inverse rotation and silently mirrors every azimuth.
    """
    A = M[:3, :3]
    N = normals @ np.linalg.inv(A)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------
# target extraction
# --------------------------------------------------------------------------
def build_target(dsm, mask, drop_below=2.0):
    """The real building top surface.

    The supplied mask is a coarse rotated rectangle that overshoots the roof
    (on this dataset ~52 of 201.6 m2 sits ~11.7 m below the roof). Cutting at
    roof_median - drop_below removes that and leaves the surface the mesh can
    actually be matched against.
    """
    D = dsm.data.astype(float)
    M = mask.data
    inside = M == 1
    roof_median = float(np.median(D[inside]))
    building = inside & (D > roof_median - drop_below)
    return {
        "building": building,
        "mask_inside": inside,
        "roof_median": roof_median,
        "mask_area_m2": float(inside.sum() * dsm.res[0] * dsm.res[1]),
        "building_area_m2": float(building.sum() * dsm.res[0] * dsm.res[1]),
        "discarded_area_m2": float((inside & ~building).sum() * dsm.res[0] * dsm.res[1]),
    }


# --------------------------------------------------------------------------
# rasterisation helpers
# --------------------------------------------------------------------------
def _height_raster(e, n, z, origin_e, origin_n, res, size):
    """Top-down max-height raster; empty cells are NaN."""
    gi = np.floor((origin_n - n) / res).astype(int) + size // 2
    gj = np.floor((e - origin_e) / res).astype(int) + size // 2
    ok = (gi >= 0) & (gi < size) & (gj >= 0) & (gj < size)
    img = np.full((size, size), -1e9)
    np.maximum.at(img, (gi[ok], gj[ok]), z[ok])
    out = np.where(img > -1e8, img, np.nan)
    return out


def _pre_enu(mesh_pts):
    """ARKit -> ENU axes, before yaw. Returns (e0, n0, up)."""
    q = mesh_pts @ AXIS_MAP.T
    return q[:, 0], q[:, 1], q[:, 2]


def _transformed(e0, n0, up, yaw, scale, deck_up):
    t = np.radians(yaw)
    c, s = np.cos(t), np.sin(t)
    return scale * (c * e0 - s * n0), scale * (s * e0 + c * n0), scale * (up - deck_up)


# --------------------------------------------------------------------------
# coarse solve
# --------------------------------------------------------------------------
def solve_yaw(mesh_pts, target, dsm, deck_up, res=0.2, size=192,
              yaw_step=1.0, scale=1.0, mirror=False, height_clip=4.0):
    """Sweep yaw; at each yaw solve translation in closed form by FFT.

    Returns a list of dicts sorted by binary overlap, each carrying both the
    overlap and the height correlation so the 180 deg pair can be separated.
    """
    building = target["building"]
    ys, xs = np.nonzero(building)
    te, tn = dsm.xy(ys, xs)
    tz = dsm.data.astype(float)[ys, xs] - target["roof_median"]
    oe, on = te.mean(), tn.mean()

    T = _height_raster(te, tn, tz, oe, on, res, size)
    Tm = np.isfinite(T)
    Tv = np.where(Tm, np.clip(np.nan_to_num(T), 0, height_clip), 0.0)
    FTm = np.fft.rfft2(Tm.astype(float))

    e0, n0, up = _pre_enu(mesh_pts)
    e0 = e0 - e0.mean()
    n0 = n0 - n0.mean()
    if mirror:
        n0 = -n0

    out = []
    for yaw in np.arange(0.0, 360.0, yaw_step):
        re, rn, rz = _transformed(e0, n0, up, yaw, scale, deck_up)
        S = _height_raster(re, rn, rz, 0.0, 0.0, res, size)
        Sm = np.isfinite(S)
        Sv = np.where(Sm, np.clip(np.nan_to_num(S), 0, height_clip), 0.0)

        corr = np.fft.irfft2(np.fft.rfft2(Sm.astype(float)) * np.conj(FTm), s=T.shape)
        k = int(np.argmax(corr))
        di, dj = np.unravel_index(k, T.shape)
        overlap = corr.flat[k] / max(min(Sm.sum(), Tm.sum()), 1)

        S2 = np.roll(np.roll(Sv, -di, 0), -dj, 1)
        S2m = np.roll(np.roll(Sm, -di, 0), -dj, 1)
        both = S2m & Tm
        if both.sum() > 30 and S2[both].std() > 1e-6 and Tv[both].std() > 1e-6:
            hcorr = float(np.corrcoef(S2[both], Tv[both])[0, 1])
            hrmse = float(np.sqrt(np.mean((S2[both] - Tv[both]) ** 2)))
        else:
            hcorr, hrmse = float("nan"), float("nan")

        dE = (dj if dj <= size // 2 else dj - size) * res
        dN = -(di if di <= size // 2 else di - size) * res
        # translation that maps the (mean-centred, yaw-rotated) mesh onto target
        out.append({
            "yaw": float(yaw), "overlap": float(overlap),
            "hcorr": hcorr, "hrmse": hrmse,
            "tE": float(oe + dE), "tN": float(on + dN),
        })
    out.sort(key=lambda d: -d["overlap"])
    return out, {"origin_e": oe, "origin_n": on}


def joint_score(cand):
    """Rank on footprint AND height together.

    Overlap alone cannot see a 180 deg flip; height correlation alone can be
    fooled by a pose that stacks the superstructures plausibly but puts the
    outline in the wrong place. The product only rewards agreeing on both.
    """
    h = cand["hcorr"]
    if not np.isfinite(h):
        h = -1.0
    return cand["overlap"] * max(h, 0.0)


def handedness_check(mesh_pts, target, dsm, deck_up, **kw):
    """Prove the axis mapping empirically instead of trusting the convention."""
    res = {}
    for mirror in (False, True):
        cand, _ = solve_yaw(mesh_pts, target, dsm, deck_up, mirror=mirror, **kw)
        best = max(cand, key=joint_score)
        best = dict(best, joint=joint_score(best))
        res["mirrored" if mirror else "right_handed"] = best
    rh, mi = res["right_handed"], res["mirrored"]
    res["verdict"] = "right_handed" if rh["joint"] > mi["joint"] else "mirrored"
    res["margin"] = float(rh["joint"] - mi["joint"])
    return res


def disambiguate_180(candidates):
    """Pick between yaw and yaw+180 using the height channel."""
    best = max(candidates, key=lambda d: d["overlap"])
    opp = min(candidates, key=lambda d: abs(((d["yaw"] - best["yaw"]) % 360) - 180))
    winner = best if (best["hcorr"] or -9) >= (opp["hcorr"] or -9) else opp
    return winner, {
        "primary": {k: best[k] for k in ("yaw", "overlap", "hcorr")},
        "opposite": {k: opp[k] for k in ("yaw", "overlap", "hcorr")},
        "overlap_margin": float(best["overlap"] - opp["overlap"]),
        "hcorr_margin": float(abs((best["hcorr"] or 0) - (opp["hcorr"] or 0))),
        "decided_by": "height_correlation",
        "chosen_yaw": float(winner["yaw"]),
    }


# --------------------------------------------------------------------------
# scale (symmetric metric -- an overlap ratio normalised by min() is biased
# toward shrinking the source and must not be used here)
# --------------------------------------------------------------------------
def scale_iou(mesh_pts, target, dsm, deck_up, yaw, scales, res=0.2, size=192):
    building = target["building"]
    ys, xs = np.nonzero(building)
    te, tn = dsm.xy(ys, xs)
    tz = dsm.data.astype(float)[ys, xs] - target["roof_median"]
    oe, on = te.mean(), tn.mean()
    T = _height_raster(te, tn, tz, oe, on, res, size)
    Tm = np.isfinite(T)
    FTm = np.fft.rfft2(Tm.astype(float))

    e0, n0, up = _pre_enu(mesh_pts)
    e0, n0 = e0 - e0.mean(), n0 - n0.mean()

    rows = []
    for sc in scales:
        re, rn, rz = _transformed(e0, n0, up, yaw, sc, deck_up)
        S = _height_raster(re, rn, rz, 0.0, 0.0, res, size)
        Sm = np.isfinite(S)
        corr = np.fft.irfft2(np.fft.rfft2(Sm.astype(float)) * np.conj(FTm), s=T.shape)
        k = int(np.argmax(corr))
        di, dj = np.unravel_index(k, T.shape)
        S2m = np.roll(np.roll(Sm, -di, 0), -dj, 1)
        inter = float((S2m & Tm).sum())
        union = float((S2m | Tm).sum())
        rows.append({"scale": float(sc), "iou": inter / max(union, 1),
                     "intersection_px": inter, "union_px": union})
    return rows


# --------------------------------------------------------------------------
# refinement
# --------------------------------------------------------------------------
def refine(mesh_pts, mesh_normals_pt, target, dsm, deck_up, yaw0, tE0, tN0,
           scale0=1.0, res=0.1, fit_scale=False, max_iter=400):
    """Polish yaw/E/N (and optionally scale) against the DSM.

    Cost blends two complementary terms:
      * chamfer between the mesh footprint boundary and the DSM building
        boundary -- this is what actually pins yaw and translation, because a
        flat roof gives the height term almost nothing to bite on;
      * height RMSE over the overlap -- keeps the superstructures registered.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from scipy.optimize import minimize

    building = target["building"]
    ys, xs = np.nonzero(building)
    r0, r1 = ys.min() - 40, ys.max() + 41
    c0, c1 = xs.min() - 40, xs.max() + 41
    r0, c0 = max(r0, 0), max(c0, 0)
    r1 = min(r1, building.shape[0])
    c1 = min(c1, building.shape[1])
    sub = building[r0:r1, c0:c1]
    subD = dsm.data.astype(float)[r0:r1, c0:c1]

    edge_t = sub & ~binary_erosion(sub, np.ones((3, 3), bool))
    dist_to_target_edge = distance_transform_edt(~edge_t) * res
    dist_outside = distance_transform_edt(~sub) * res

    gt = dsm.transform
    oe = gt[0] + c0 * gt[1]
    on = gt[3] + r0 * gt[5]
    H, W = sub.shape

    e0, n0, up = _pre_enu(mesh_pts)
    ce, cn = e0.mean(), n0.mean()
    e0, n0 = e0 - ce, n0 - cn
    up_rel = up - deck_up
    # only upward-facing points carry usable height information
    top = mesh_normals_pt[:, 1] > 0.85

    def rasterise(yaw, tE, tN, sc):
        t = np.radians(yaw)
        c, s = np.cos(t), np.sin(t)
        e = sc * (c * e0 - s * n0) + tE
        n = sc * (s * e0 + c * n0) + tN
        z = sc * up_rel
        gi = np.floor((on - n) / res).astype(int)
        gj = np.floor((e - oe) / res).astype(int)
        ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
        return gi, gj, ok, z, e, n

    def cost(p):
        yaw, tE, tN = p[0], p[1], p[2]
        sc = p[3] if fit_scale else scale0
        gi, gj, ok, z, e, n = rasterise(yaw, tE, tN, sc)
        if ok.sum() < 500:
            return 1e6
        occ = np.zeros((H, W), bool)
        occ[gi[ok], gj[ok]] = True
        edge_s = occ & ~binary_erosion(occ, np.ones((3, 3), bool))
        if edge_s.sum() < 20:
            return 1e6
        # source edge -> target edge, plus a penalty for source spilling outside
        c1_ = float(dist_to_target_edge[edge_s].mean())
        c2_ = float(dist_outside[occ].mean())
        # height agreement on upward-facing points
        okt = ok & top
        if okt.sum() > 200:
            zs = z[okt] + target["roof_median"]
            zd = subD[gi[okt], gj[okt]]
            good = np.isfinite(zd)
            resid = zs[good] - zd[good]
            resid = resid - np.median(resid)
            c3_ = float(np.sqrt(np.mean(np.clip(resid, -3, 3) ** 2)))
        else:
            c3_ = 3.0
        return c1_ + 0.5 * c2_ + 0.3 * c3_

    x0 = [yaw0, tE0, tN0] + ([scale0] if fit_scale else [])
    steps = [1.5, 0.4, 0.4] + ([0.01] if fit_scale else [])
    simplex = [np.array(x0, float)]
    for i in range(len(x0)):
        p = np.array(x0, float)
        p[i] += steps[i]
        simplex.append(p)
    r = minimize(cost, np.array(x0, float), method="Nelder-Mead",
                 options={"initial_simplex": np.array(simplex),
                          "maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-5})

    yaw, tE, tN = r.x[0], r.x[1], r.x[2]
    sc = r.x[3] if fit_scale else scale0

    # Vertical offset from upward-facing points, robust to outliers. Sample the
    # DSM bilinearly (matching verify_alignment) so the two agree.
    gi, gj, ok, z, e, n = rasterise(yaw, tE, tN, sc)
    # Tie the offset to the deck surface specifically. Parapet copings sit
    # ~0.6 m proud of the deck but the DSM renders them at roughly roof level,
    # so including them biases tZ upward by ~0.2 m.
    on_deck = ok & top & (np.abs(up_rel) < 0.25)
    sel = on_deck if on_deck.sum() > 500 else (ok & top)
    zd = dsm.sample(e[sel], n[sel])
    good = np.isfinite(zd)
    resid = zd[good] - z[sel][good]
    dz = float(np.median(resid))

    return {
        "yaw_deg": float(yaw % 360.0), "scale": float(sc),
        "tE": float(tE), "tN": float(tN), "tZ": dz,
        "pivot_pre": [float(ce), float(cn), float(deck_up)],
        "cost": float(r.fun), "cost_start": float(cost(np.array(x0, float))),
        "n_iter": int(r.nit), "success": bool(r.success),
        "deck_up_local": float(deck_up),
    }


def matrix_from_solution(sol, tilt_correction=None):
    return build_matrix(sol["yaw_deg"], sol["scale"],
                        [sol["tE"], sol["tN"], sol["tZ"]],
                        pivot=sol["pivot_pre"], tilt_correction=tilt_correction)


def verify_alignment(mesh, M, target, dsm, res=0.1):
    """Independent QC of a finished transform: footprint IoU + height residuals."""
    from scipy.ndimage import binary_fill_holes

    P, Npt, W, _ = mesh.sample_surface_with_normals(10)
    Q = apply(M, P)
    Nq = apply_normals(M, Npt)

    building = target["building"]
    ys, xs = np.nonzero(building)
    r0, r1 = max(ys.min() - 30, 0), min(ys.max() + 31, building.shape[0])
    c0, c1 = max(xs.min() - 30, 0), min(xs.max() + 31, building.shape[1])
    sub = building[r0:r1, c0:c1]
    gt = dsm.transform
    oe, on = gt[0] + c0 * gt[1], gt[3] + r0 * gt[5]
    H, W_ = sub.shape

    gi = np.floor((on - Q[:, 1]) / res).astype(int)
    gj = np.floor((Q[:, 0] - oe) / res).astype(int)
    ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W_)
    occ = np.zeros((H, W_), bool)
    occ[gi[ok], gj[ok]] = True
    occ = binary_fill_holes(occ)

    inter = float((occ & sub).sum())
    union = float((occ | sub).sum())

    top = Nq[:, 2] > 0.85
    zd = dsm.sample(Q[top, 0], Q[top, 1])
    good = np.isfinite(zd)
    resid = Q[top, 2][good] - zd[good]
    deck_pts = Q[top][good]
    near_deck = np.abs(deck_pts[:, 2] - target["roof_median"]) < 0.6

    return {
        "footprint_iou": inter / max(union, 1.0),
        "footprint_mesh_m2": float(occ.sum() * res * res),
        "footprint_target_m2": float(sub.sum() * res * res),
        "height_resid_median_m": float(np.median(resid)),
        "height_resid_mad_m": float(np.median(np.abs(resid - np.median(resid)))),
        "height_resid_p95_abs_m": float(np.percentile(np.abs(resid), 95)),
        "deck_mean_elev_m": float(deck_pts[near_deck][:, 2].mean()) if near_deck.sum() else float("nan"),
        "dsm_roof_median_m": float(target["roof_median"]),
        "n_top_points": int(good.sum()),
    }


def residual_tilt(mesh_pts_enu, mesh_normals_enu, dsm, target, max_resid=1.0):
    """Compare the mesh deck plane with the DSM roof plane after alignment.

    A mismatch here is either real drainage fall or ARKit gravity error; we
    report it and let the caller decide whether to apply it.
    """
    top = mesh_normals_enu[:, 2] > 0.98
    P = mesh_pts_enu[top]
    if len(P) < 200:
        return None
    zd = dsm.sample(P[:, 0], P[:, 1])
    ok = np.isfinite(zd)
    P = P[ok]
    # Tight band around the modal deck height: parapet copings sit only ~0.6 m
    # higher and will otherwise tilt the fitted plane.
    hist, edges = np.histogram(P[:, 2], bins=np.arange(P[:, 2].min(),
                                                       P[:, 2].max() + 0.1, 0.1))
    mode_z = edges[int(np.argmax(hist))] + 0.05
    P = P[np.abs(P[:, 2] - mode_z) < 0.30]
    if len(P) < 200:
        return None

    def plane(pts):
        A = np.c_[pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean(),
                  np.ones(len(pts))]
        coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
        azi = np.degrees(np.arctan2(-coef[0], -coef[1])) % 360
        return coef, float(tilt), float(azi)

    cm, tm, am = plane(P)
    ys, xs = np.nonzero(target["building"] &
                        (np.abs(dsm.data.astype(float) - target["roof_median"]) < 0.8))
    e, n = dsm.xy(ys, xs)
    Q = np.c_[e, n, dsm.data.astype(float)[ys, xs]]
    cd, td, ad = plane(Q)
    dgrad = np.hypot(cm[0] - cd[0], cm[1] - cd[1])
    return {
        "mesh_deck_tilt_deg": tm, "mesh_deck_downslope_grid_az": am,
        "dsm_roof_tilt_deg": td, "dsm_roof_downslope_grid_az": ad,
        "tilt_disagreement_deg": float(np.degrees(np.arctan(dgrad))),
    }


def tilt_matrix(from_normal, to_normal=(0, 0, 1)):
    """Smallest rotation taking `from_normal` onto `to_normal`."""
    a = np.asarray(from_normal, float)
    a = a / np.linalg.norm(a)
    b = np.asarray(to_normal, float)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    c = float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


# --------------------------------------------------------------------------
# north
# --------------------------------------------------------------------------
def grid_convergence(lon, lat, lon0):
    """gamma such that true_azimuth = grid_azimuth + gamma (degrees)."""
    return float(np.degrees(np.arctan(np.tan(np.radians(lon - lon0)) *
                                      np.sin(np.radians(lat)))))


def utm_zone_center(crs_wkt, fallback=75.0):
    import re
    m = re.search(r'PARAMETER\["Longitude of natural origin",\s*(-?[\d.]+)', crs_wkt)
    if m:
        return float(m.group(1))
    m = re.search(r'"central_meridian",\s*(-?[\d.]+)', crs_wkt)
    return float(m.group(1)) if m else fallback


def to_lonlat(e, n, crs_wkt):
    """UTM -> lon/lat, via pyproj if available else osgeo."""
    try:
        from pyproj import CRS, Transformer
        tr = Transformer.from_crs(CRS.from_wkt(crs_wkt), CRS.from_epsg(4326),
                                  always_xy=True)
        return tr.transform(e, n)
    except ImportError:
        pass
    try:
        from rasterio.warp import transform as rio_transform
        lon, lat = rio_transform(crs_wkt, "EPSG:4326",
                                 list(np.atleast_1d(e)), list(np.atleast_1d(n)))
        return np.asarray(lon), np.asarray(lat)
    except ImportError:
        from osgeo import osr
        src = osr.SpatialReference()
        src.ImportFromWkt(crs_wkt)
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src, dst)
        pts = ct.TransformPoints(list(zip(np.atleast_1d(e), np.atleast_1d(n))))
        arr = np.asarray(pts)
        return arr[:, 0], arr[:, 1]
