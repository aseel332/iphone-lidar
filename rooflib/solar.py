"""Solar-facing attributes: aspect, usable area, obstruction context, shading.

Two conventions matter and are easy to get wrong:

* Azimuths derived from a UTM grid are GRID azimuths. Solar geometry needs TRUE
  north. At this site grid convergence is -0.951 deg, so true = grid - 0.951.
  Every azimuth leaving this module is true-north referenced and labelled.

* A roof flatter than ~2 deg has no meaningful aspect. Reporting one anyway
  invites a downstream layout tool mounting panels to a number that is pure
  fitting noise, so it is returned as NaN.
"""
from __future__ import annotations

import numpy as np


def plane_solar_record(plane, geom, centroids, areas, deck_z, convergence_deg):
    """Per-plane record with everything a layout/yield step needs."""
    F = plane["faces"]
    C = centroids[F]
    z = C[:, 2]
    tilt = geom["tilt_deg"]
    flat = tilt < 2.0
    return {
        "plane_id": int(plane["plane_id"]),
        "orientation": geom["orientation"],
        "surface_area_m2": round(float(plane["area_m2"]), 3),
        "projected_area_m2": round(float(geom["projected_area_m2"]), 3),
        "tilt_deg": round(tilt, 3),
        "grid_azimuth_deg": None if flat else round(geom["grid_azimuth_deg"], 2),
        "true_azimuth_deg": None if flat else round(geom["true_azimuth_deg"], 2),
        "aspect": "flat" if flat else _compass(geom["true_azimuth_deg"]),
        "centroid_e": round(float(plane["centroid"][0]), 3),
        "centroid_n": round(float(plane["centroid"][1]), 3),
        "centroid_z": round(float(plane["centroid"][2]), 3),
        "z_min": round(float(z.min()), 3),
        "z_max": round(float(z.max()), 3),
        "height_above_deck_m": round(float(plane["centroid"][2] - deck_z), 3),
        "n_faces": int(len(F)),
    }


def _compass(az):
    if not np.isfinite(az):
        return "flat"
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int(((az % 360) + 11.25) // 22.5) % 16]


# --------------------------------------------------------------------------
# usable area on a terrace
# --------------------------------------------------------------------------
def usable_area(terrace_plane, centroids, obstruction_face_sets, cell=0.1,
                edge_setback_m=0.6, obstruction_clearance_m=0.5):
    """Deck area left after edge setback and clearance around obstructions.

    This is the number a layout step actually consumes -- gross plane area
    overstates what panels can occupy, often by 30-40 % on a small roof.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    from .segment import footprint_mask

    mask, origin, c = footprint_mask(centroids[terrace_plane["faces"]][:, :2],
                                     cell=cell)
    gross = float(mask.sum() * c * c)

    er = int(round(edge_setback_m / c))
    inner = binary_erosion(mask, np.ones((2 * er + 1,) * 2, bool)) if er > 0 else mask

    e0, n1 = origin
    H, W = mask.shape
    blocked = np.zeros_like(mask)
    for sel in obstruction_face_sets:
        P = centroids[sel]
        gj = np.floor((P[:, 0] - e0) / c).astype(int)
        gi = np.floor((n1 - P[:, 1]) / c).astype(int)
        ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
        blocked[gi[ok], gj[ok]] = True
    cl = int(round(obstruction_clearance_m / c))
    if cl > 0 and blocked.any():
        blocked = binary_dilation(blocked, np.ones((2 * cl + 1,) * 2, bool))

    free = inner & ~blocked
    return {
        "gross_area_m2": round(gross, 2),
        "after_setback_m2": round(float(inner.sum() * c * c), 2),
        "usable_area_m2": round(float(free.sum() * c * c), 2),
        "blocked_by_objects_m2": round(float((inner & blocked).sum() * c * c), 2),
        "edge_setback_m": edge_setback_m,
        "obstruction_clearance_m": obstruction_clearance_m,
        "_mask": free,
        "_origin": origin,
        "_cell": c,
    }


# --------------------------------------------------------------------------
# shading
# --------------------------------------------------------------------------
def horizon_profile(dsm, origin_e, origin_n, origin_z, n_sectors=36,
                    max_dist_m=60.0, step_m=0.5, convergence_deg=0.0):
    """Sky-obstruction elevation angle per azimuth sector, from the DSM.

    The DSM is the only input that sees beyond the building, so it is the only
    source for neighbour shading. Returned azimuths are true-north referenced.
    """
    out = []
    dists = np.arange(2.0, max_dist_m + step_m, step_m)
    for k in range(n_sectors):
        grid_az = 360.0 * k / n_sectors
        th = np.radians(grid_az)
        e = origin_e + dists * np.sin(th)
        n = origin_n + dists * np.cos(th)
        z = dsm.sample(e, n)
        good = np.isfinite(z)
        if not good.any():
            out.append({"true_azimuth_deg": (grid_az + convergence_deg) % 360,
                        "horizon_elev_deg": 0.0, "distance_m": None,
                        "obstruction_height_m": 0.0})
            continue
        ang = np.degrees(np.arctan2(z[good] - origin_z, dists[good]))
        i = int(np.argmax(ang))
        out.append({
            "true_azimuth_deg": round((grid_az + convergence_deg) % 360, 2),
            "horizon_elev_deg": round(float(max(ang[i], 0.0)), 3),
            "distance_m": round(float(dists[good][i]), 2),
            "obstruction_height_m": round(float(z[good][i] - origin_z), 3),
        })
    return out


def sector_summary(dsm, mask, target, deck_z, n_sectors=12, max_dist_m=50.0,
                   min_dist_m=8.0, convergence_deg=0.0):
    """Coarse per-sector obstruction table around the building."""
    D = dsm.data.astype(float)
    ys, xs = np.nonzero(mask.data == 1)
    cy, cx = ys.mean(), xs.mean()
    Y, X = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    res = dsm.res[0]
    dist = np.hypot((Y - cy) * res, (X - cx) * res)
    grid_bearing = np.degrees(np.arctan2((X - cx) * res, -(Y - cy) * res)) % 360
    outside = mask.data != 1
    width = 360.0 / n_sectors
    rows = []
    for k in range(n_sectors):
        a0 = k * width
        sel = outside & (grid_bearing >= a0) & (grid_bearing < a0 + width) \
            & (dist < max_dist_m) & (dist > min_dist_m)
        if sel.sum() == 0:
            continue
        zz = D[sel]
        i = int(np.argmax(zz))
        rows.append({
            "sector_true_az_from": round((a0 + convergence_deg) % 360, 2),
            "sector_true_az_to": round((a0 + width + convergence_deg) % 360, 2),
            "max_elev_m": round(float(zz.max()), 2),
            "above_deck_m": round(float(zz.max() - deck_z), 2),
            "distance_at_max_m": round(float(dist[sel][i]), 1),
            "elev_angle_deg": round(float(np.degrees(np.arctan2(
                max(zz.max() - deck_z, 0.0), max(dist[sel][i], 1e-6)))), 2),
            "px_above_deck": int((zz > deck_z).sum()),
        })
    return rows


def solar_position_summary(lat, lon):
    """Noon sun elevation at solstices/equinox -- a sanity frame for shading.

    Deliberately a closed-form declination model, not an ephemeris: it is used
    to sanity-check that computed horizon angles matter, not to drive yield.
    """
    out = {}
    for name, decl in (("summer_solstice", 23.44), ("equinox", 0.0),
                       ("winter_solstice", -23.44)):
        elev = 90.0 - abs(lat - decl)
        out[name] = {
            "solar_declination_deg": decl,
            "solar_noon_elevation_deg": round(elev, 2),
            "shadow_length_per_m": round(1.0 / np.tan(np.radians(elev)), 3)
            if elev > 1 else None,
        }
    out["latitude_deg"] = round(lat, 6)
    out["longitude_deg"] = round(lon, 6)
    out["hemisphere"] = "northern" if lat >= 0 else "southern"
    out["optimal_fixed_tilt_deg_rule_of_thumb"] = round(abs(lat) * 0.87, 1)
    out["optimal_azimuth_true_deg"] = 180.0 if lat >= 0 else 0.0
    return out


def interrow_spacing(obstruction_height_m, min_sun_elev_deg=25.0):
    """Shadow length cast by an obstruction at a given sun elevation."""
    if obstruction_height_m <= 0:
        return 0.0
    return float(obstruction_height_m / np.tan(np.radians(min_sun_elev_deg)))
