"""Segment the fused rooftop mesh into planes and objects.

Design notes, driven by what this mesh actually is:

* The 8 OBJ `o` groups are texture atlases, not parts, and the mesh is 491
  disconnected shells. So neither the file structure nor topological
  connectivity carries any semantics -- everything here is geometric.

* ~23% of the surface is transition/noise triangles whose pitch and azimuth
  histograms are near-uniform. Plane fitting on raw normals leaves ~48% of the
  area unassigned, so normals are smoothed over a spatial kNN first (the mesh
  is fragmented, so topological neighbours are not available).

* Plane finding is done by normal-direction clustering then offset histogram,
  not brute-force RANSAC: it is far cheaper on 89k faces and buildings are
  overwhelmingly made of a handful of normal directions.

* Naive "cluster everything above the deck" returns ONE blob covering the whole
  roof, because the continuous parapet ring bridges every object. Parapets are
  therefore identified and removed before objects are clustered.
"""
from __future__ import annotations

import numpy as np

HORIZONTAL_MAX_TILT = 12.0   # deg from horizontal to count as a floor/roof
VERTICAL_MIN_TILT = 72.0     # deg from horizontal to count as a wall


# --------------------------------------------------------------------------
# normal smoothing
# --------------------------------------------------------------------------
def smooth_normals(centroids, normals, areas, k=24, iterations=3, radius=0.30,
                   normal_sigma_deg=25.0):
    """Edge-preserving (bilateral) normal smoothing.

    Plain isotropic averaging is actively harmful on a rooftop: at every
    deck/parapet corner it blends a horizontal and a vertical normal into a
    spurious 45 deg one, and measurably *grows* the noise band. Neighbours are
    therefore weighted by normal similarity as well as distance, so smoothing
    happens along surfaces and stops at creases.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(centroids)
    nbrs = tree.query_ball_point(centroids, r=radius)
    cos_sigma = np.radians(normal_sigma_deg)
    N = normals.copy()
    for _ in range(iterations):
        out = np.empty_like(N)
        for i, nb in enumerate(nbrs):
            if len(nb) < 3:
                out[i] = N[i]
                continue
            nb = np.asarray(nb)
            if len(nb) > k:
                d2 = ((centroids[nb] - centroids[i]) ** 2).sum(1)
                nb = nb[np.argsort(d2)[:k]]
            ni = N[nb]
            sgn = np.sign(ni @ N[i])
            sgn[sgn == 0] = 1
            ni = sgn[:, None] * ni
            ang = np.arccos(np.clip(ni @ N[i], -1, 1))
            w = areas[nb] * np.exp(-0.5 * (ang / cos_sigma) ** 2)
            v = (w[:, None] * ni).sum(0)
            nv = np.linalg.norm(v)
            out[i] = v / nv if nv > 1e-9 else N[i]
        N = out
    return N


def assign_residual_faces(planes, centroids, normals, areas, assigned,
                          dist_tol=0.10, angle_tol_deg=18.0, rounds=3):
    """Sweep leftover faces into the nearest compatible plane.

    Direction-cluster + offset-peak finding is conservative by construction and
    leaves the crease geometry unclaimed; this recovers most of it without
    inventing new planes.
    """
    cos_tol = np.cos(np.radians(angle_tol_deg))
    normals_p = np.array([p["normal"] for p in planes])
    offsets_p = np.array([p["normal"] @ p["centroid"] for p in planes])
    for _ in range(rounds):
        free = np.nonzero(~assigned)[0]
        if len(free) == 0:
            break
        ang = np.abs(normals[free] @ normals_p.T)
        dist = np.abs(centroids[free] @ normals_p.T - offsets_p[None, :])
        ok = (ang >= cos_tol) & (dist <= dist_tol)
        score = np.where(ok, -dist, -np.inf)
        best = np.argmax(score, axis=1)
        good = np.isfinite(score[np.arange(len(free)), best])
        if not good.any():
            break
        for pid in np.unique(best[good]):
            sel = free[good & (best == pid)]
            planes[pid]["faces"] = np.concatenate([planes[pid]["faces"], sel])
            assigned[sel] = True
        # refit after absorbing new members
        for p in planes:
            w = areas[p["faces"]]
            nrm = (w[:, None] * normals[p["faces"]]).sum(0)
            n_ = np.linalg.norm(nrm)
            if n_ > 1e-9:
                p["normal"] = nrm / n_
            p["centroid"] = (w[:, None] * centroids[p["faces"]]).sum(0) / w.sum()
            p["area_m2"] = float(w.sum())
        normals_p = np.array([p["normal"] for p in planes])
        offsets_p = np.array([p["normal"] @ p["centroid"] for p in planes])
    return planes, assigned


# --------------------------------------------------------------------------
# plane extraction
# --------------------------------------------------------------------------
def _direction_clusters(normals, areas, angle_tol_deg=12.0, min_area=1.0):
    """Greedy clustering of normal directions by area."""
    cos_tol = np.cos(np.radians(angle_tol_deg))
    remaining = np.ones(len(normals), bool)
    clusters = []
    while remaining.any():
        idx = np.nonzero(remaining)[0]
        # seed on the largest-area remaining face's direction
        seed = idx[np.argmax(areas[idx])]
        d = normals[seed].copy()
        for _ in range(6):  # a few refits to centre the cluster
            sel = idx[(normals[idx] @ d) >= cos_tol]
            if len(sel) == 0:
                break
            v = (areas[sel][:, None] * normals[sel]).sum(0)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                break
            d = v / nv
        sel = idx[(normals[idx] @ d) >= cos_tol]
        if len(sel) == 0:
            remaining[seed] = False
            continue
        if areas[sel].sum() >= min_area:
            clusters.append((d, sel))
        remaining[sel] = False
    return clusters


def _split_by_offset(centroids, areas, direction, members, bin_size=0.12,
                     min_area=0.8):
    """Split a normal-direction cluster into parallel planes by offset peaks."""
    off = centroids[members] @ direction
    lo, hi = off.min(), off.max()
    if hi - lo < bin_size:
        return [(float(off.mean()), members)]
    edges = np.arange(lo, hi + bin_size, bin_size)
    hist, _ = np.histogram(off, bins=edges, weights=areas[members])
    peaks = []
    for i in range(len(hist)):
        if hist[i] <= 0:
            continue
        left = hist[i - 1] if i > 0 else 0.0
        right = hist[i + 1] if i + 1 < len(hist) else 0.0
        if hist[i] >= left and hist[i] >= right and hist[i] >= min_area * 0.5:
            peaks.append(i)
    if not peaks:
        peaks = [int(np.argmax(hist))]
    centers = np.array([edges[i] + bin_size / 2 for i in peaks])
    assign = np.argmin(np.abs(off[:, None] - centers[None, :]), axis=1)
    out = []
    for j in range(len(centers)):
        sel = members[assign == j]
        if len(sel) and areas[sel].sum() >= min_area:
            out.append((float(np.average(off[assign == j],
                                         weights=areas[sel])), sel))
    return out


def _spatial_split(centroids, areas, direction, offset, members, cell=0.3,
                   min_area=0.6):
    """Break a plane into spatially disconnected patches (two aligned walls etc.)."""
    from scipy.ndimage import label

    d = np.asarray(direction, float)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    P = centroids[members]
    a = P @ u
    b = P @ v
    ai = np.floor((a - a.min()) / cell).astype(int)
    bi = np.floor((b - b.min()) / cell).astype(int)
    H, W = ai.max() + 3, bi.max() + 3
    grid = np.zeros((H, W), bool)
    grid[ai + 1, bi + 1] = True
    lab, n = label(grid, structure=np.ones((3, 3), int))
    ids = lab[ai + 1, bi + 1]
    out = []
    for c in range(1, n + 1):
        sel = members[ids == c]
        if len(sel) and areas[sel].sum() >= min_area:
            out.append(sel)
    return out


def extract_planes(centroids, normals, areas, angle_tol_deg=12.0,
                   offset_bin=0.12, min_area=0.8, spatial_cell=0.3):
    """Full plane decomposition. Returns a list of dicts."""
    planes = []
    for direction, members in _direction_clusters(normals, areas,
                                                  angle_tol_deg, min_area):
        for offset, sub in _split_by_offset(centroids, areas, direction,
                                            members, offset_bin, min_area):
            for patch in _spatial_split(centroids, areas, direction, offset,
                                        sub, spatial_cell, min_area * 0.75):
                w = areas[patch]
                nrm = (w[:, None] * normals[patch]).sum(0)
                nrm /= np.linalg.norm(nrm)
                if nrm[2] < 0:          # keep normals pointing up / outward-up
                    pass                 # (walls keep their own orientation)
                cen = (w[:, None] * centroids[patch]).sum(0) / w.sum()
                planes.append({
                    "faces": patch,
                    "normal": nrm,
                    "centroid": cen,
                    "area_m2": float(w.sum()),
                })
    planes.sort(key=lambda p: -p["area_m2"])
    for i, p in enumerate(planes):
        p["plane_id"] = i
    return planes


def plane_geometry(plane, convergence_deg=0.0):
    """Tilt, azimuth (grid and true), and projected area for one plane."""
    n = plane["normal"]
    nz = n if n[2] >= 0 else -n          # upward-facing convention for tilt/azimuth
    tilt = float(np.degrees(np.arccos(np.clip(nz[2], -1, 1))))
    # Below ~2 deg the in-plane direction is numerically meaningless -- a "flat"
    # roof has no aspect, and reporting one invites it being used as a mounting
    # azimuth downstream.
    if tilt < 2.0:
        grid_az = float("nan")
    else:
        grid_az = float(np.degrees(np.arctan2(nz[0], nz[1])) % 360.0)
    horiz = tilt <= HORIZONTAL_MAX_TILT
    vert = tilt >= VERTICAL_MIN_TILT
    return {
        "tilt_deg": tilt,
        "grid_azimuth_deg": grid_az,
        "true_azimuth_deg": (grid_az + convergence_deg) % 360.0 if np.isfinite(grid_az) else float("nan"),
        "projected_area_m2": float(plane["area_m2"] * abs(nz[2])),
        "orientation": "horizontal" if horiz else ("vertical" if vert else "sloped"),
    }


# --------------------------------------------------------------------------
# object isolation
# --------------------------------------------------------------------------
def height_grid(points, weights_z, cell=0.1, pad=4):
    """Top-down max-height raster of a point set; returns (grid, origin, cell)."""
    e, n = points[:, 0], points[:, 1]
    e0, n1 = e.min() - pad * cell, n.max() + pad * cell
    gj = np.floor((e - e0) / cell).astype(int)
    gi = np.floor((n1 - n) / cell).astype(int)
    H = gi.max() + pad + 1
    W = gj.max() + pad + 1
    g = np.full((H, W), -1e9)
    np.maximum.at(g, (gi, gj), weights_z)
    return np.where(g > -1e8, g, np.nan), (e0, n1), cell


def footprint_mask(points_xy, cell=0.1, close_iter=3):
    """Solid top-down footprint of a point set, plus its raster origin."""
    from scipy.ndimage import binary_closing, binary_fill_holes

    e0 = points_xy[:, 0].min() - 5 * cell
    n1 = points_xy[:, 1].max() + 5 * cell
    gj = np.floor((points_xy[:, 0] - e0) / cell).astype(int)
    gi = np.floor((n1 - points_xy[:, 1]) / cell).astype(int)
    H, W = gi.max() + 6, gj.max() + 6
    occ = np.zeros((H, W), bool)
    occ[gi, gj] = True
    occ = binary_closing(occ, np.ones((2 * close_iter + 1,) * 2, bool))
    occ = binary_fill_holes(occ)
    return occ, (e0, n1), cell


def boundary_distance_fn(mask, origin, cell):
    """Callable: (E, N) -> metres from the footprint boundary (0 on the edge)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    edge = mask & ~binary_erosion(mask, np.ones((3, 3), bool))
    dist = distance_transform_edt(~edge) * cell
    e0, n1 = origin
    H, W = mask.shape

    def fn(xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        gj = np.clip(np.floor((xy[:, 0] - e0) / cell).astype(int), 0, W - 1)
        gi = np.clip(np.floor((n1 - xy[:, 1]) / cell).astype(int), 0, H - 1)
        out = dist[gi, gj]
        return out[0] if out.size == 1 else out

    return fn


def find_parapet(planes, geoms, footprint_poly_dist, deck_z,
                 max_top_above_deck=1.8, max_dist_from_edge=1.2):
    """Vertical planes hugging the roof outline and short enough to be a parapet."""
    ids = []
    for p, g in zip(planes, geoms):
        if g["orientation"] != "vertical":
            continue
        top = p["z_max"]
        if top - deck_z > max_top_above_deck:
            continue
        if footprint_poly_dist(p["centroid"][:2]) > max_dist_from_edge:
            continue
        ids.append(p["plane_id"])
    return set(ids)


def cluster_objects(face_centroids, face_areas, face_z, deck_z, exclude_faces,
                    min_height=0.35, cell=0.15, min_area=0.4, gap_cells=1):
    """Connected-component clustering of above-deck geometry.

    `exclude_faces` must already remove the parapet ring, otherwise everything
    merges into a single component.
    """
    from scipy.ndimage import binary_closing, label

    keep = (face_z - deck_z > min_height) & (~exclude_faces)
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return [], None
    P = face_centroids[idx]
    e0, n1 = P[:, 0].min() - cell, P[:, 1].max() + cell
    gj = np.floor((P[:, 0] - e0) / cell).astype(int)
    gi = np.floor((n1 - P[:, 1]) / cell).astype(int)
    H, W = gi.max() + 2, gj.max() + 2
    occ = np.zeros((H, W), bool)
    occ[gi, gj] = True
    if gap_cells:
        occ = binary_closing(occ, np.ones((2 * gap_cells + 1,) * 2, bool))
    lab, n = label(occ, structure=np.ones((3, 3), int))
    ids = lab[gi, gj]
    out = []
    for c in range(1, n + 1):
        sel = idx[ids == c]
        if len(sel) == 0 or face_areas[sel].sum() < min_area:
            continue
        out.append(sel)
    out.sort(key=lambda s: -face_areas[s].sum())
    return out, (lab, (e0, n1), cell)


def extract_objects_hierarchical(planes, geoms, centroids, areas, normals,
                                 min_terrace_area=3.0, min_height=0.35,
                                 cell=0.15, min_object_area=0.4,
                                 footprint_cell=0.15, dilate_m=0.4):
    """Peel objects off the roof one terrace at a time, highest terrace first.

    A flat "everything above the deck" clustering fails on this building twice
    over: the parapet ring bridges separate objects, and the roof is genuinely
    layered (open deck -> a ~3 m block -> water tanks standing on that block),
    so a single threshold cannot separate the tanks from the block carrying
    them.

    Working top-down fixes both. Each horizontal plane is a terrace; objects
    resting on it are the above-terrace geometry inside its footprint that no
    higher terrace has already claimed. The result is a parent/child tree
    rather than a flat list.
    """
    from scipy.ndimage import binary_dilation, binary_closing, label

    terraces = [(p, g) for p, g in zip(planes, geoms)
                if g["orientation"] == "horizontal" and p["area_m2"] >= min_terrace_area]
    terraces.sort(key=lambda pg: -pg[0]["centroid"][2])  # highest first

    used = np.zeros(len(areas), bool)
    for p, _ in terraces:
        used[p["faces"]] = True   # terrace surfaces are not objects themselves

    results = []
    for p, g in terraces:
        z_t = float(p["centroid"][2])
        fmask, origin, fcell = footprint_mask(centroids[p["faces"]][:, :2],
                                              cell=footprint_cell)
        grow = int(round(dilate_m / fcell))
        if grow > 0:
            fmask = binary_dilation(fmask, np.ones((2 * grow + 1,) * 2, bool))
        e0, n1 = origin
        H, W = fmask.shape

        gj = np.floor((centroids[:, 0] - e0) / fcell).astype(int)
        gi = np.floor((n1 - centroids[:, 1]) / fcell).astype(int)
        inside = ((gi >= 0) & (gi < H) & (gj >= 0) & (gj < W))
        over = np.zeros(len(areas), bool)
        over[inside] = fmask[gi[inside], gj[inside]]

        cand = over & (~used) & (centroids[:, 2] > z_t + min_height)
        idx = np.nonzero(cand)[0]
        children = []
        if len(idx):
            # Cluster in 3D voxels, not in top-down projection: two tanks a
            # hand's width apart share footprint cells once projected, and a
            # ring of walls encloses interior cells that merge everything into
            # one component. In 3D they stay distinct.
            P = centroids[idx]
            o = P.min(0) - cell
            vi = np.floor((P - o) / cell).astype(int)
            dims = vi.max(0) + 2
            occ = np.zeros(dims, bool)
            occ[vi[:, 0], vi[:, 1], vi[:, 2]] = True
            lab, n = label(occ, structure=np.ones((3, 3, 3), int))
            ids = lab[vi[:, 0], vi[:, 1], vi[:, 2]]
            for c in range(1, n + 1):
                sel = idx[ids == c]
                if len(sel) and areas[sel].sum() >= min_object_area:
                    children.append(sel)
            children.sort(key=lambda s: -areas[s].sum())
            for s in children:
                used[s] = True

        results.append({
            "terrace_plane_id": p["plane_id"],
            "terrace_z": z_t,
            "terrace_area_m2": p["area_m2"],
            "objects": children,
        })

    results.sort(key=lambda r: r["terrace_z"])
    return results


def classify_object(verts_enu, faces_sel, centroids, normals, areas, deck_z):
    """Heuristic label for an above-deck cluster."""
    C = centroids[faces_sel]
    A = areas[faces_sel]
    N = normals[faces_sel]
    z = C[:, 2]
    height = float(z.max() - deck_z)
    base = float(np.percentile(z, 2) - deck_z)
    # np.ptp(), not ndarray.ptp() -- the method was removed in numpy 2.0
    foot_e = float(np.ptp(C[:, 0]))
    foot_n = float(np.ptp(C[:, 1]))
    foot = max(foot_e * foot_n, 1e-6)
    tilt = np.degrees(np.arccos(np.clip(np.abs(N[:, 2]), 0, 1)))
    horiz_area = float(A[tilt < HORIZONTAL_MAX_TILT].sum())
    vert_area = float(A[tilt > VERTICAL_MIN_TILT].sum())
    total = float(A.sum())

    # rectilinearity: how concentrated the wall azimuths are into 2 orthogonal
    # families. Tanks are round, so theirs spread out.
    wall = tilt > VERTICAL_MIN_TILT
    if wall.sum() > 10:
        az = np.degrees(np.arctan2(N[wall, 0], N[wall, 1])) % 180.0
        h, _ = np.histogram(az, bins=18, range=(0, 180), weights=A[wall])
        rect = float(np.sort(h)[-2:].sum() / max(h.sum(), 1e-9))
    else:
        rect = 0.0

    # Labels are heuristic and deliberately conservative. On a noisy phone-LiDAR
    # scan the geometric signature of a stair headroom and a walled tank plinth
    # overlap heavily, so anything ambiguous stays generic and carries a low
    # confidence rather than asserting a type that downstream code would trust.
    label, confidence = "unclassified_object", 0.3
    if foot > 20.0 and height > 1.5:
        label, confidence = "superstructure_block", 0.5
    elif height > 1.0 and foot <= 6.0 and rect < 0.60:
        label, confidence = "water_tank", 0.6 if height > 1.5 else 0.45
    elif 4.0 < foot <= 20.0 and 1.6 < height < 3.4 and rect >= 0.60 and horiz_area > 0.6:
        label, confidence = "stair_headroom", 0.55
    elif height <= 1.0 and horiz_area > 0.5 * total:
        label, confidence = "raised_platform", 0.5
    elif total < 1.5:
        label, confidence = "small_object", 0.4
    if base > 0.8:
        label = label + "_elevated"
    return {
        "label": label,
        "label_confidence": confidence,
        "height_above_deck_m": height,
        "base_above_deck_m": base,
        "top_elev_m": float(z.max()),
        "base_elev_m": float(np.percentile(z, 2)),
        "footprint_bbox_m": [foot_e, foot_n],
        "surface_area_m2": total,
        "horizontal_area_m2": horiz_area,
        "vertical_area_m2": vert_area,
        "rectilinearity": rect,
        "centroid_e": float(C[:, 0].mean()),
        "centroid_n": float(C[:, 1].mean()),
    }
