"""Emit the pipeline result in the target site-analysis JSON schema.

Schema notes that drove decisions here:

* Polygons are lat/lng closed rings, so plane footprints are traced in UTM,
  simplified, then reprojected -- simplifying after reprojection would apply a
  metre-based tolerance to degrees.
* `azimuth_deg` is compass (0 = N, 90 = E, 180 = S). Our planes carry TRUE-north
  azimuths already, which is what compass means; grid azimuths would be wrong
  by the grid convergence.
* `base_height_m` is above ground level, not above the deck and not an absolute
  elevation, so it is measured from a ground datum derived from the DSM.
* Only non-vertical planes are roof faces. Walls are structure, not roof.
"""
from __future__ import annotations

import numpy as np

# The schema's obstruction vocabulary is vent | ac_unit | chimney | skylight |
# pipe | unknown. Our classifier speaks a different language (it was built for
# Indian rooftops: water tanks, stair headrooms, terrace blocks). Only map where
# the correspondence is defensible; everything else is honestly "unknown"
# rather than forced into a bucket a downstream consumer would misread.
TYPE_MAP = {
    "water_tank": "unknown",
    "water_tank_elevated": "unknown",
    "stair_headroom": "unknown",
    "superstructure_block": "unknown",
    "raised_platform": "unknown",
    "small_object": "unknown",
    "unclassified_object": "unknown",
}


def _rings_from_mask(mask, origin, cell):
    """Trace closed boundary rings of a binary mask, in UTM metres."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    e0, n1 = origin
    H, W = mask.shape
    xs = e0 + (np.arange(W) + 0.5) * cell
    ys = n1 - (np.arange(H) + 0.5) * cell
    fig = plt.figure()
    try:
        cs = plt.contour(xs, ys, mask.astype(float), levels=[0.5])
        rings = []
        for path in cs.get_paths():
            v = path.vertices
            if len(v) >= 4:
                rings.append(np.asarray(v, float))
    finally:
        plt.close(fig)
    rings.sort(key=lambda r: -_ring_area(r))
    return rings


def _ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _douglas_peucker(pts, eps):
    """Simplify a polyline; keeps ring vertex counts sane for JSON."""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    d = end - start
    n = np.linalg.norm(d)
    if n < 1e-12:
        dist = np.linalg.norm(pts - start, axis=1)
    else:
        # 2-D cross product written out: numpy 2.0 removed the 2-vector form
        v = pts - start
        dist = np.abs(d[0] * v[:, 1] - d[1] * v[:, 0]) / n
    i = int(np.argmax(dist))
    if dist[i] > eps:
        left = _douglas_peucker(pts[: i + 1], eps)
        right = _douglas_peucker(pts[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def plane_fit_stats(plane, verts_enu, faces, inlier_tol=0.05):
    """RMSE / inlier ratio / sample count of the plane fit, from real vertices."""
    vid = np.unique(faces[plane["faces"]].ravel())
    P = verts_enu[vid]
    n = np.asarray(plane["normal"], float)
    d = (P - np.asarray(plane["centroid"], float)) @ n
    rmse = float(np.sqrt(np.mean(d ** 2)))
    inlier = float(np.mean(np.abs(d) <= inlier_tol))
    return {"rmse_m": round(rmse, 4),
            "inlier_ratio": round(inlier, 4),
            "sample_count": int(len(P))}


def _face_confidence(fit, area_m2):
    """Blend fit quality with how much evidence the plane has.

    A 0.9 m^2 sliver fitted perfectly is not as trustworthy as a 40 m^2 plane
    fitted well, so area gates the score rather than only the residual.
    """
    q = fit["inlier_ratio"] * float(np.exp(-fit["rmse_m"] / 0.05))
    size = min(1.0, area_m2 / 10.0)
    return round(float(np.clip(0.35 + 0.65 * q * size, 0.0, 0.99)), 3)


def build_site_json(*, address_id, planes, geoms, plane_records, terraces,
                    object_records, verts_enu, faces, centroids, dsm, mask,
                    crs_wkt, to_lonlat, ground_elev_m, deck_z,
                    footprint_mask_fn, imagery_date=None,
                    dsm_resolution_m=0.1, provider="iphone-lidar+dsm",
                    simplify_m=0.12, min_face_area_m2=1.0,
                    max_fit_rmse_m=0.20, min_inlier_ratio=0.45,
                    extra_warnings=()):
    """Assemble the schema document."""
    # ---- roof faces: everything that is not a wall -------------------------
    face_ids = {}
    faces_out = []
    face_anchor = []          # (face_id, e, n, z) for attributing obstructions
    rejected = []
    idx = 0
    for p, g, rec in zip(planes, geoms, plane_records):
        if g["orientation"] == "vertical":
            continue
        if p["area_m2"] < min_face_area_m2:
            continue
        # Quality gate. Residual-absorption can leave a "plane" whose members
        # span metres; emitting those as roof faces would hand a consumer a
        # confident-looking surface that does not exist.
        fit_pre = plane_fit_stats(p, verts_enu, faces)
        if (fit_pre["rmse_m"] > max_fit_rmse_m
                or fit_pre["inlier_ratio"] < min_inlier_ratio):
            rejected.append((p["plane_id"], p["area_m2"], fit_pre))
            continue
        idx += 1
        fid = f"face-{idx}"
        face_ids[p["plane_id"]] = fid

        m_, orig_, cell_ = footprint_mask_fn(centroids[p["faces"]][:, :2], cell=0.15)
        rings = _rings_from_mask(m_, orig_, cell_)
        if not rings:
            polygon = []
        else:
            r = _douglas_peucker(rings[0], simplify_m)
            if not np.allclose(r[0], r[-1]):
                r = np.vstack([r, r[0]])
            lon, lat = to_lonlat(r[:, 0], r[:, 1], crs_wkt)
            lon = np.atleast_1d(lon); lat = np.atleast_1d(lat)
            polygon = [[round(float(a), 7), round(float(o), 7)]
                       for a, o in zip(lat, lon)]

        face_anchor.append((fid, float(p["centroid"][0]), float(p["centroid"][1]),
                            float(p["centroid"][2])))
        fit = fit_pre
        tilt = float(g["tilt_deg"])
        az = g["true_azimuth_deg"]
        faces_out.append({
            "id": fid,
            "polygon": polygon,
            "tilt_deg": round(tilt, 2),
            "azimuth_deg": None if not np.isfinite(az) else round(float(az), 2),
            "area_m2": round(float(p["area_m2"]), 2),
            "base_height_m": round(float(p["z_min"] - ground_elev_m), 2),
            "plane_fit": fit,
            "confidence": _face_confidence(fit, p["area_m2"]),
        })

    # ---- obstructions ------------------------------------------------------
    obs_out = []
    for i, o in enumerate(object_records, start=1):
        lon, lat = to_lonlat(np.array([o["centroid_e"]]),
                             np.array([o["centroid_n"]]), crs_wkt)
        on_face = face_ids.get(o.get("on_terrace_plane_id"), None)
        if on_face is None and face_anchor:
            # The terrace this object sits on may have been rejected by the fit
            # gate. Fall back to the nearest surviving face that is actually
            # below the object, rather than dropping the association.
            base = float(o["base_elev_m"])
            cands = [a for a in face_anchor if a[3] <= base + 0.5] or face_anchor
            on_face = min(cands, key=lambda a: np.hypot(a[1] - o["centroid_e"],
                                                        a[2] - o["centroid_n"]))[0]
            o = {**o, "_attributed_by": "nearest_face_fallback"}
        obs_out.append({
            "id": f"obs-{i}",
            "type": TYPE_MAP.get(o["label"], "unknown"),
            "source_label": o["label"],          # our richer label, kept alongside
            "centroid": {"lat": round(float(np.atleast_1d(lat)[0]), 7),
                         "lng": round(float(np.atleast_1d(lon)[0]), 7)},
            "footprint_m": {"width": round(float(o["footprint_bbox_m"][0]), 2),
                            "length": round(float(o["footprint_bbox_m"][1]), 2)},
            "height_m": round(float(o["top_elev_m"] - o["base_elev_m"]), 2),
            "on_face": on_face,
            "confidence": round(float(o["label_confidence"]), 2),
            **({"attributed_by": "nearest_face_fallback"}
               if o.get("_attributed_by") else {}),
        })

    # ---- warnings: real ones, derived from the data ------------------------
    warnings = list(extra_warnings)
    if rejected:
        area = sum(r[1] for r in rejected)
        warnings.append(
            f"planes_rejected_on_fit_quality: {len(rejected)} candidate faces "
            f"({area:.1f} m2) failed rmse<={max_fit_rmse_m} m / "
            f"inlier>={min_inlier_ratio}; these are fusion noise, not surfaces")
    unplaced = [o["id"] for o in obs_out if o["on_face"] is None]
    if unplaced:
        warnings.append(f"obstructions_not_attributed_to_a_face: {','.join(unplaced)}")
    if any(f["polygon"] == [] for f in faces_out):
        warnings.append("polygon_trace_failed_on_one_or_more_faces")
    if all(o["type"] == "unknown" for o in obs_out) and obs_out:
        warnings.append(
            "obstruction_types_unmapped: classifier vocabulary (water_tank, "
            "stair_headroom, superstructure_block) does not correspond to the "
            "schema's vent/ac_unit/chimney/skylight/pipe set; see source_label")

    lon_c, lat_c = to_lonlat(np.array([float(np.mean(verts_enu[:, 0]))]),
                             np.array([float(np.mean(verts_enu[:, 1]))]), crs_wkt)

    return {
        "address_id": address_id,
        "location": {"lat": round(float(np.atleast_1d(lat_c)[0]), 7),
                     "lng": round(float(np.atleast_1d(lon_c)[0]), 7)},
        "source": {
            "imagery_date": imagery_date,
            "dsm_resolution_m": dsm_resolution_m,
            "provider": provider,
        },
        "roof_faces": faces_out,
        "obstructions": obs_out,
        "trees": [],
        "warnings": warnings,
    }
