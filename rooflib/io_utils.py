"""Input loading for the rooftop pipeline.

Deliberately dependency-light: rasters go through a shim that prefers rasterio
(what Colab gets from `pip install rasterio`) and falls back to osgeo/GDAL.
The OBJ reader is hand-rolled numpy so the pipeline does not need trimesh or
open3d, neither of which is preinstalled anywhere we run.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# rasters
# --------------------------------------------------------------------------
@dataclass
class Raster:
    """A georeferenced grid. `transform` is the 6-tuple GDAL geotransform."""

    data: np.ndarray  # (H, W) or (B, H, W)
    transform: tuple
    crs_wkt: str
    nodata: float | None = None

    @property
    def shape(self):
        return self.data.shape[-2:]

    @property
    def res(self):
        return abs(self.transform[1]), abs(self.transform[5])

    def xy(self, rows, cols, center=True):
        """Pixel indices -> world (easting, northing)."""
        gt = self.transform
        off = 0.5 if center else 0.0
        e = gt[0] + (np.asarray(cols) + off) * gt[1] + (np.asarray(rows) + off) * gt[2]
        n = gt[3] + (np.asarray(cols) + off) * gt[4] + (np.asarray(rows) + off) * gt[5]
        return e, n

    def rowcol(self, e, n):
        """World (easting, northing) -> fractional pixel indices."""
        gt = self.transform
        det = gt[1] * gt[5] - gt[2] * gt[4]
        de, dn = np.asarray(e) - gt[0], np.asarray(n) - gt[3]
        col = (de * gt[5] - dn * gt[2]) / det
        row = (dn * gt[1] - de * gt[4]) / det
        return row - 0.5, col - 0.5

    def sample(self, e, n, order=1):
        """Bilinear sample of a single-band raster at world coordinates.

        Returns NaN outside the grid so callers can mask cleanly.
        """
        arr = self.data if self.data.ndim == 2 else self.data[0]
        r, c = self.rowcol(e, n)
        h, w = arr.shape
        if order == 0:
            ri, ci = np.round(r).astype(int), np.round(c).astype(int)
            ok = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
            out = np.full(np.shape(e), np.nan, float)
            out[ok] = arr[ri[ok], ci[ok]]
            return out
        r0 = np.floor(r).astype(int)
        c0 = np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        out = np.full(np.shape(e), np.nan, float)
        ok = (r0 >= 0) & (r0 + 1 < h) & (c0 >= 0) & (c0 + 1 < w)
        if not np.any(ok):
            return out
        r0o, c0o, fro, fco = r0[ok], c0[ok], fr[ok], fc[ok]
        v = (
            arr[r0o, c0o] * (1 - fro) * (1 - fco)
            + arr[r0o + 1, c0o] * fro * (1 - fco)
            + arr[r0o, c0o + 1] * (1 - fro) * fco
            + arr[r0o + 1, c0o + 1] * fro * fco
        )
        out[ok] = v
        return out


def read_raster(path) -> Raster:
    """Read a GeoTIFF via rasterio if present, else osgeo."""
    try:
        import rasterio

        with rasterio.open(path) as ds:
            data = ds.read()
            gt = ds.transform.to_gdal()
            wkt = ds.crs.to_wkt() if ds.crs else ""
            nod = ds.nodata
    except ImportError:
        from osgeo import gdal

        gdal.UseExceptions()
        ds = gdal.Open(str(path))
        data = ds.ReadAsArray()
        gt = ds.GetGeoTransform()
        wkt = ds.GetProjection()
        nod = ds.GetRasterBand(1).GetNoDataValue()
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    return Raster(np.asarray(data), tuple(gt), wkt, nod)


def write_raster(path, arr, transform, crs_wkt, nodata=None):
    """Write a single- or multi-band GeoTIFF (rasterio preferred)."""
    arr = np.asarray(arr)
    bands = 1 if arr.ndim == 2 else arr.shape[0]
    a3 = arr[None] if arr.ndim == 2 else arr
    try:
        import rasterio
        from rasterio.transform import Affine

        tr = Affine.from_gdal(*transform)
        with rasterio.open(
            path, "w", driver="GTiff", height=a3.shape[1], width=a3.shape[2],
            count=bands, dtype=a3.dtype, crs=crs_wkt or None, transform=tr,
            nodata=nodata, compress="deflate",
        ) as ds:
            ds.write(a3)
    except ImportError:
        from osgeo import gdal, gdal_array

        gdal.UseExceptions()
        drv = gdal.GetDriverByName("GTiff")
        gtype = gdal_array.NumericTypeCodeToGDALTypeCode(a3.dtype.type)
        ds = drv.Create(str(path), a3.shape[2], a3.shape[1], bands, gtype,
                        options=["COMPRESS=DEFLATE"])
        ds.SetGeoTransform(transform)
        ds.SetProjection(crs_wkt)
        for i in range(bands):
            b = ds.GetRasterBand(i + 1)
            b.WriteArray(a3[i])
            if nodata is not None:
                b.SetNoDataValue(float(nodata))
        ds.FlushCache()


# --------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------
@dataclass
class Mesh:
    V: np.ndarray            # (nv, 3) vertices, ARKit frame (x right, y up, z toward viewer)
    F: np.ndarray            # (nf, 3) triangle vertex indices
    VT: np.ndarray | None = None   # (nvt, 2) texture coords
    FT: np.ndarray | None = None   # (nf, 3) texture indices
    face_material: np.ndarray | None = None   # (nf,) material name per face
    face_group: np.ndarray | None = None      # (nf,) `o` group per face
    materials: dict = field(default_factory=dict)  # name -> texture relative path

    # -- derived quantities -------------------------------------------------
    def triangles(self):
        return self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]

    def face_normals(self, normalize=True):
        p0, p1, p2 = self.triangles()
        n = np.cross(p1 - p0, p2 - p0)
        if not normalize:
            return n
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(ln, 1e-12)

    def face_areas(self):
        p0, p1, p2 = self.triangles()
        return np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / 2.0

    def face_centroids(self):
        p0, p1, p2 = self.triangles()
        return (p0 + p1 + p2) / 3.0

    def sample_surface(self, n_per_face=10, seed=0, include_vertices=True):
        """Barycentric point sampling so top-down rasters come out solid."""
        p0, p1, p2 = self.triangles()
        rng = np.random.default_rng(seed)
        out = [self.V] if include_vertices else []
        for _ in range(n_per_face):
            a = rng.random((len(self.F), 1))
            b = rng.random((len(self.F), 1))
            flip = (a + b) > 1
            a = np.where(flip, 1 - a, a)
            b = np.where(flip, 1 - b, b)
            out.append(p0 + a * (p1 - p0) + b * (p2 - p0))
        return np.vstack(out)

    def sample_surface_with_normals(self, n_per_face=10, seed=0):
        """Points plus the normal and area-weight of the face each came from."""
        p0, p1, p2 = self.triangles()
        nrm = self.face_normals()
        area = self.face_areas()
        rng = np.random.default_rng(seed)
        pts, nn, ww, ff = [], [], [], []
        idx = np.arange(len(self.F))
        for _ in range(n_per_face):
            a = rng.random((len(self.F), 1))
            b = rng.random((len(self.F), 1))
            flip = (a + b) > 1
            a = np.where(flip, 1 - a, a)
            b = np.where(flip, 1 - b, b)
            pts.append(p0 + a * (p1 - p0) + b * (p2 - p0))
            nn.append(nrm)
            ww.append(area / n_per_face)
            ff.append(idx)
        return (np.vstack(pts), np.vstack(nn),
                np.concatenate(ww), np.concatenate(ff))

    def connected_shells(self):
        """Union-find over shared vertices -> per-face shell id.

        This mesh is heavily fragmented, so the result is informational; do not
        rely on it for segmentation.
        """
        parent = np.arange(len(self.V))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for f in self.F:
            for x, y in ((f[0], f[1]), (f[1], f[2])):
                ra, rb = find(x), find(y)
                if ra != rb:
                    parent[ra] = rb
        roots = np.array([find(i) for i in range(len(self.V))])
        _, inv = np.unique(roots, return_inverse=True)
        return inv[self.F[:, 0]]


def load_obj_from_zip(zip_path, obj_name=None) -> tuple[Mesh, dict]:
    """Read the OBJ (and its MTL) straight out of the zip, no extraction."""
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    if obj_name is None:
        cands = [n for n in names if n.lower().endswith(".obj")]
        if not cands:
            raise FileNotFoundError(f"no .obj inside {zip_path}")
        obj_name = cands[0]
    text = zf.read(obj_name).decode("utf-8", "replace")

    V, VT, F, FT = [], [], [], []
    fmat, fgrp = [], []
    cur_mat, cur_grp, mtllib = None, None, None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        tag, _, rest = line.partition(" ")
        if tag == "v":
            p = rest.split()
            V.append((float(p[0]), float(p[1]), float(p[2])))
        elif tag == "vt":
            p = rest.split()
            VT.append((float(p[0]), float(p[1])))
        elif tag == "f":
            toks = rest.split()
            vi, ti = [], []
            for t in toks:
                parts = t.split("/")
                vi.append(int(parts[0]) - 1)
                ti.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
            # fan-triangulate any n-gons
            for k in range(1, len(vi) - 1):
                F.append((vi[0], vi[k], vi[k + 1]))
                FT.append((ti[0], ti[k], ti[k + 1]))
                fmat.append(cur_mat)
                fgrp.append(cur_grp)
        elif tag == "o" or tag == "g":
            cur_grp = rest.strip()
        elif tag == "usemtl":
            cur_mat = rest.strip()
        elif tag == "mtllib":
            mtllib = rest.strip()

    materials = {}
    if mtllib and mtllib in names:
        cur = None
        for line in zf.read(mtllib).decode("utf-8", "replace").splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "newmtl":
                cur = p[1]
            elif p[0] == "map_Kd" and cur:
                materials[cur] = " ".join(p[1:])

    mesh = Mesh(
        V=np.asarray(V, float),
        F=np.asarray(F, np.int64),
        VT=np.asarray(VT, float) if VT else None,
        FT=np.asarray(FT, np.int64) if FT else None,
        face_material=np.asarray(fmat, object),
        face_group=np.asarray(fgrp, object),
        materials=materials,
    )
    meta = {
        "zip": str(zip_path),
        "obj": obj_name,
        "mtl": mtllib,
        "textures": {k: v for k, v in materials.items()},
        "n_vertices": int(len(mesh.V)),
        "n_faces": int(len(mesh.F)),
    }
    return mesh, meta


def load_textures_from_zip(zip_path, materials, max_side=None):
    """Decode texture JPEGs into numpy arrays (only needed for ortho render)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    zf = zipfile.ZipFile(zip_path)
    out = {}
    for name, rel in materials.items():
        if rel not in zf.namelist():
            continue
        im = Image.open(io.BytesIO(zf.read(rel))).convert("RGB")
        if max_side and max(im.size) > max_side:
            sc = max_side / max(im.size)
            im = im.resize((int(im.width * sc), int(im.height * sc)), Image.BILINEAR)
        out[name] = np.asarray(im)
    return out


def save_json(path, obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
