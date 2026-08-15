"""Export the segmented, georeferenced model.

Two coordinate flavours are written, as requested:

* local ENU -- origin at the building centroid, metres. Vertex values stay
  small, which matters because most 3D viewers store positions as float32:
  at UTM easting 250811 / northing 2546703 the representable step is already
  ~0.25 m, so an absolute-coordinate mesh visibly jitters and z-fights.
* absolute UTM 43N -- for GIS interop, where that precision loss does not
  apply because GIS stores doubles.

`origin_utm` in the sidecar JSON converts one to the other.
"""
from __future__ import annotations

import json

import numpy as np


def write_obj(path, verts, faces, groups=None, group_names=None,
              header_comment=None):
    """Write an OBJ, optionally splitting faces into named `o` groups."""
    with open(path, "w") as fh:
        if header_comment:
            for line in header_comment.splitlines():
                fh.write(f"# {line}\n")
        for v in verts:
            fh.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        if groups is None:
            for f in faces:
                fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))
        else:
            groups = np.asarray(groups)
            for gid in np.unique(groups):
                name = (group_names or {}).get(int(gid), f"segment_{int(gid)}")
                fh.write(f"o {name}\n")
                for f in faces[groups == gid]:
                    fh.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))


def write_ply_labelled(path, verts, faces, face_labels, palette=None):
    """ASCII PLY with a per-face colour -- opens in MeshLab/CloudCompare."""
    labels = np.asarray(face_labels)
    uniq = np.unique(labels)
    if palette is None:
        rng = np.random.default_rng(7)
        palette = {int(u): tuple(int(x) for x in rng.integers(60, 235, 3))
                   for u in uniq}
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(verts)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {len(faces)}\n")
        fh.write("property list uchar int vertex_index\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for v in verts:
            fh.write("%.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for f, l in zip(faces, labels):
            r, g, b = palette[int(l)]
            fh.write("3 %d %d %d %d %d %d\n" % (f[0], f[1], f[2], r, g, b))


def write_geojson(path, features, crs_epsg=32643):
    fc = {
        "type": "FeatureCollection",
        "name": "roof_segments",
        "crs": {"type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"}},
        "features": features,
    }
    with open(path, "w") as fh:
        json.dump(fc, fh, indent=2)


def mask_to_polygons(mask, origin, cell, min_area_cells=4):
    """Trace raster components into (already simplified) rectilinear rings.

    Marching-squares would give smoother outlines but pulls in extra
    dependencies; per-component boundary tracing is enough for a footprint
    polygon at 0.1 m and keeps the export dependency-free.
    """
    from scipy.ndimage import label

    e0, n1 = origin
    lab, n = label(mask, structure=np.ones((3, 3), int))
    polys = []
    for c in range(1, n + 1):
        sel = lab == c
        if sel.sum() < min_area_cells:
            continue
        ii, jj = np.nonzero(sel)
        ring = []
        # boundary cells only, ordered by angle about the centroid: adequate
        # for convex-ish roof parts, which is what these are
        from scipy.ndimage import binary_erosion
        edge = sel & ~binary_erosion(sel, np.ones((3, 3), bool))
        ei, ej = np.nonzero(edge)
        ex = e0 + (ej + 0.5) * cell
        ny = n1 - (ei + 0.5) * cell
        cx, cy = ex.mean(), ny.mean()
        order = np.argsort(np.arctan2(ny - cy, ex - cx))
        ring = [[float(ex[k]), float(ny[k])] for k in order]
        if len(ring) < 3:
            continue
        ring.append(ring[0])
        polys.append({"ring": ring, "area_m2": float(sel.sum() * cell * cell)})
    return polys


def transform_sidecar(solution, matrix, origin_utm, crs_wkt, convergence_deg,
                      diagnostics, notes=None):
    """Everything needed to reproduce or invert the georeferencing."""
    return {
        "schema": "rooflib/transform/1",
        "description": "ARKit mesh -> georeferenced ENU / UTM",
        "source_frame": {
            "name": "ARKit world (gravity-aligned)",
            "axes": "x right, y up (anti-gravity), z toward viewer",
            "units": "metres",
        },
        "axis_map_arkit_to_enu": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "axis_map_note": "east=+x, north=-z, up=+y; determinant +1 "
                         "(chirality preserved). Using north=+z mirrors the "
                         "model and flips every azimuth.",
        "solution": {
            "yaw_deg_grid": solution["yaw_deg"],
            "scale": solution["scale"],
            "translation_e": solution["tE"],
            "translation_n": solution["tN"],
            "translation_z": solution["tZ"],
            "pivot_pre_enu": solution["pivot_pre"],
        },
        "matrix_arkit_to_utm_4x4": np.asarray(matrix).tolist(),
        "local_enu": {
            "origin_utm": list(map(float, origin_utm)),
            "note": "local = utm - origin_utm; keeps float32 viewers precise",
        },
        "crs": {
            "horizontal_wkt": crs_wkt,
            "horizontal_epsg": 32643,
            "vertical_datum": "UNDECLARED in source DSM -- elevations are "
                              "carried through as-is from dsm.tif. Could be "
                              "ellipsoidal or a geoid/MSL height. Does not "
                              "affect tilt, azimuth or yield.",
        },
        "north": {
            "grid_convergence_deg": convergence_deg,
            "relation": "true_azimuth = grid_azimuth + grid_convergence_deg",
            "note": "all exported azimuths are TRUE north referenced",
        },
        "diagnostics": diagnostics,
        "notes": notes or [],
    }


def save_json(path, obj):
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"{type(o)} not serialisable")

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


def write_csv(path, rows, columns=None):
    import csv
    if not rows:
        open(path, "w").close()
        return
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
