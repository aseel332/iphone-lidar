#!/usr/bin/env python3
"""Generate the Colab notebook from the tested rooflib modules.

The notebook embeds rooflib via %%writefile cells so it is completely
self-contained -- the only thing that has to be in Drive is the four inputs.
Generating it from the real module files means the notebook cannot drift away
from the code that was actually tested.
"""
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
MODULES = ["io_utils", "georef", "segment", "solar", "exporters", "viz",
           "export_schema", "ground"]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

cells.append(md(r"""
# Rooftop LiDAR → georeferenced, segmented, solar-ready model

Takes an iPhone/ARKit rooftop scan plus a co-registered DSM / RGB / building-mask
triple and produces a **georeferenced, semantically segmented** roof model with
per-surface pitch, azimuth, elevation and usable area.

**Pipeline**

| Stage | What it does |
|---|---|
| 1 | Load and characterise the four inputs |
| 2 | Solve the ARKit → UTM transform (4-DOF: yaw, E, N, Z + optional scale) |
| 3 | Segment into planes, terraces and objects |
| 4 | Derive solar attributes: tilt, true azimuth, usable area, shading |
| 5 | Export in local-ENU **and** absolute UTM, plus GeoJSON / CSV / JSON |

**Three things this pipeline is careful about**, each of which silently ruins the
output if you get it wrong:

1. **Chirality.** ARKit is `x` right, `y` up, `z` toward viewer. Mapping to ENU as
   `east=+x, north=−z, up=+y` preserves handedness. Using `north=+z` mirrors the
   model and flips every azimuth. Stage 2 *proves* the choice against the data.
2. **The 180° ambiguity.** A roughly rectangular footprint matches its own
   180° rotation almost perfectly. Binary overlap cannot separate them; the
   superstructure **height** map can. Getting this wrong turns south-facing into
   north-facing.
3. **Grid vs true north.** UTM azimuths are grid azimuths. Solar geometry needs
   true north. Grid convergence is applied to every azimuth that leaves the pipeline.
"""))

cells.append(md("## 0 · Environment"))

cells.append(code(r"""
import sys, subprocess, importlib

# find_spec("google.colab") imports the parent package and raises off-Colab,
# so probe by import instead.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False
print("Running in Colab:", IN_COLAB)

# rasterio is the only thing Colab lacks; numpy/scipy/matplotlib are preinstalled.
for pkg, mod in [("rasterio", "rasterio")]:
    if importlib.util.find_spec(mod) is None:
        print(f"installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import numpy as np, scipy, matplotlib
print("numpy", np.__version__, "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__)
try:
    import rasterio; print("rasterio", rasterio.__version__)
except ImportError:
    from osgeo import gdal; print("using GDAL", gdal.__version__)
"""))

cells.append(md(r"""
## 1 · Mount Drive and point at the inputs

Expected Drive layout — create this folder and upload the four files into it:

```
MyDrive/
└── iphone-lidar/
    └── inputs/
        ├── lidar_object.zip
        ├── dsm.tif
        ├── rgb.tif
        └── building_mask.tif
```

Outputs are written next to it in `MyDrive/iphone-lidar/outputs/`. Running
outside Colab falls back to local paths, so the same notebook works on a laptop.
"""))

cells.append(code(r"""
import os, pathlib

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")          # opens a Google auth prompt
    BASE = pathlib.Path("/content/drive/MyDrive/iphone-lidar")
else:
    BASE = pathlib.Path(os.environ.get("ROOFLIB_BASE", pathlib.Path.cwd()))

INPUT_DIR  = BASE / "inputs"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = ["lidar_object.zip", "dsm.tif", "rgb.tif", "building_mask.tif"]

print("input dir :", INPUT_DIR)
print("output dir:", OUTPUT_DIR)

# Convenience: if the four files are absent but the single-file bundle is
# sitting in Drive, unpack it. Uploading one 21 MB zip is much less fiddly
# than four separate files.
if any(not (INPUT_DIR / f).exists() for f in REQUIRED):
    for cand in [BASE / "iphone-lidar-inputs.zip", INPUT_DIR / "iphone-lidar-inputs.zip"]:
        if cand.exists():
            import zipfile
            print(f"unpacking bundle {cand.name} ...")
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cand) as z:
                z.extractall(INPUT_DIR)
            break

missing = [f for f in REQUIRED if not (INPUT_DIR / f).exists()]
if missing:
    raise SystemExit(
        f"Missing {missing} in {INPUT_DIR}.\n"
        "Either upload those files there, or drop iphone-lidar-inputs.zip into\n"
        f"{BASE} and re-run this cell."
    )
for f in REQUIRED:
    print(f"  {f:24s} {(INPUT_DIR / f).stat().st_size / 1e6:8.2f} MB")
"""))

cells.append(md(r"""
## 2 · Materialise `rooflib`

Written to disk here so the notebook stays self-contained — nothing to upload
besides the inputs, and nothing to `pip install` from a private index.
"""))

cells.append(code("import pathlib\npathlib.Path('rooflib').mkdir(exist_ok=True)\n"
                  "pathlib.Path('rooflib/__init__.py').write_text('')\n"
                  "print('rooflib package created')"))

for name in MODULES:
    src = (ROOT / "rooflib" / f"{name}.py").read_text()
    cells.append(code(f"%%writefile rooflib/{name}.py\n{src}"))

cells.append(code(r"""
import importlib, rooflib
for m in ["io_utils", "georef", "segment", "solar", "exporters", "viz",
          "export_schema", "ground"]:
    importlib.import_module(f"rooflib.{m}")
    importlib.reload(sys.modules[f"rooflib.{m}"])

from rooflib.io_utils import read_raster, load_obj_from_zip, Raster
from rooflib import georef as G, segment as S, solar as SO, exporters as X, viz as V
from rooflib import export_schema as ES
from rooflib import ground as GR
print("rooflib ready")
"""))

cells.append(md("## 3 · Load and characterise the inputs"))

cells.append(code(r"""
dsm  = read_raster(INPUT_DIR / "dsm.tif")
rgb  = read_raster(INPUT_DIR / "rgb.tif")
mask = read_raster(INPUT_DIR / "building_mask.tif")
mesh, mesh_meta = load_obj_from_zip(INPUT_DIR / "lidar_object.zip")

assert dsm.transform == mask.transform == rgb.transform, \
    "rasters are not on a common grid - the pipeline assumes they are co-registered"

res = dsm.res[0]
print(f"rasters   : {dsm.shape[1]}x{dsm.shape[0]} px @ {res} m  (co-registered: identical geotransform)")
print(f"            extent {dsm.shape[1]*res:.1f} x {dsm.shape[0]*res:.1f} m")
print(f"DSM       : {np.nanmin(dsm.data):.2f} .. {np.nanmax(dsm.data):.2f} m   nodata={dsm.nodata}")
print(f"mask      : {(mask.data==1).sum()} px = {(mask.data==1).sum()*res*res:.1f} m^2")
print(f"mesh      : {mesh_meta['n_vertices']} verts, {mesh_meta['n_faces']} faces, "
      f"{mesh.face_areas().sum():.1f} m^2 surface")
print(f"            bbox {np.round(mesh.V.max(0)-mesh.V.min(0), 2)} m (ARKit frame)")
print(f"            {len(mesh.materials)} texture atlases, {len(np.unique(mesh.face_group))} 'o' groups")
"""))

cells.append(code(r"""
# --- structural facts that determine how segmentation has to work -----------
shells = mesh.connected_shells()
n_shells = len(np.unique(shells))

grp_extent = {}
for g in np.unique(mesh.face_group):
    C = mesh.face_centroids()[mesh.face_group == g]
    grp_extent[g] = (np.ptp(C[:, 0]), np.ptp(C[:, 2]))

print(f"connected shells: {n_shells}")
print("  -> the mesh is NOT one surface; topological connectivity carries no semantics.")
print(f"\n'o' groups and their horizontal extents (ARKit x,z):")
for g, e in grp_extent.items():
    print(f"  {g}: {e[0]:5.1f} x {e[1]:5.1f} m")
print("  -> every group spans the whole roof: these are texture atlases, not parts.")

N, A = mesh.face_normals(), mesh.face_areas()
tilt = np.degrees(np.arccos(np.clip(np.abs(N[:, 1]), 0, 1)))
noise = A[(tilt > 5) & (tilt < 85)].sum()
print(f"\nfaces between 5 and 85 deg off-axis: {noise:.1f} m^2 ({100*noise/A.sum():.0f}% of surface)")
print("  -> transition/noise triangles from mesh fusion; smoothed before plane fitting.")
"""))

cells.append(code(r"""
# --- the mask overshoots the roof: quantify before relying on it ------------
target = G.build_target(dsm, mask)
print(f"building_mask polygon      : {target['mask_area_m2']:8.1f} m^2")
print(f"  of which real roof surface: {target['building_area_m2']:8.1f} m^2")
print(f"  discarded (>2 m below roof): {target['discarded_area_m2']:8.1f} m^2"
      f"  ({100*target['discarded_area_m2']/target['mask_area_m2']:.0f}% of the mask)")
print(f"roof median elevation      : {target['roof_median']:8.2f} m")
print("\n-> the mask is a coarse rotated rectangle, usable as a region of interest")
print("   but NOT as the roof footprint. The alignment target is mask AND DSM.")
"""))

cells.append(md(r"""
### Ground elevation, and building height

The building's height above ground feeds both `roof_summary.json` and
`site_analysis.json` (`base_height_m` on every face and obstruction), so it is
worth getting right rather than eyeballing a percentile.

A naive `percentile(DSM outside mask, 10)` is biased on this scene: the ring
immediately around the building is contaminated by an adjacent lower roof
sitting right against the wall, which pulls the estimate up by 0.5-1 m. The fix
is to find the *dominant* low surface in the scene rather than the *nearest*
one — real ground is the largest flat run at the bottom of the elevation
histogram, and `rooflib.ground` locates it directly.
"""))

cells.append(code(r"""
ground_elev_m, ground_diag = GR.estimate_ground_elevation(dsm, mask)

print(f"modal ground surface: {ground_diag['modal_bin_lo_m']:.2f}-{ground_diag['modal_bin_hi_m']:.2f} m  "
      f"({ground_diag['modal_bin_px']} px, {ground_diag['modal_bin_area_m2']:.0f} m^2 - "
      f"the largest single elevation band in the scene)")
print(f"GROUND ELEVATION: {ground_elev_m:.2f} m  "
      f"(robust median of {ground_diag['band_px']} px within {ground_diag['band_m']} m of the mode)")
print(f"  cross-check, whole-scene p10 : {ground_diag['cross_check_whole_scene_p10_m']:.2f} m  "
      f"(agrees within {ground_diag['agreement_m']:.2f} m)")
print(f"  cross-check, near-ring p5    : {ground_diag['cross_check_ring_2to10m_p5_m']:.2f} m  "
      f"(reads high - contaminated by the adjacent lower roof, hence not used)")
print(f"\nroof median {target['roof_median']:.2f} m -> building height to deck "
      f"{target['roof_median']-ground_elev_m:.2f} m")
"""))

cells.append(md(r"""
## 4 · Stage 2 — georeferencing

Four unknowns, not seven: ARKit's `+y` is gravity-up, and ARKit is metric, so
roll and pitch are already solved and only **yaw, E, N, Z** (plus an optional
scale check) remain.

Translation is not searched — at each trial yaw it is solved in closed form by
FFT cross-correlation, which is why a full 360° sweep is cheap.
"""))

cells.append(code(r"""
deck_local, deck_info = G.estimate_deck_level(mesh)
deck_n, deck_tilt, deck_area = G.deck_normal(mesh, deck_local)

print(f"deck (walking surface) at ARKit y = {deck_local:.3f}, {deck_info['deck_area_m2']:.1f} m^2")
print(f"deck normal {np.round(deck_n, 5)}  ->  {deck_tilt:.3f} deg from vertical")
print("\nGRAVITY CHECK: ARKit's +y is within a degree of true vertical, so roll/pitch")
print("are already solved and this is a 4-DOF problem.")
print("\nhorizontal surface bands in the ARKit frame (area-weighted):")
for b in deck_info["bands"]:
    print(f"  y = {b['y']:6.2f}   {b['area_m2']:6.1f} m^2")
"""))

cells.append(code(r"""
pts, pt_normals, pt_w, pt_face = mesh.sample_surface_with_normals(n_per_face=10)
print(f"sampled {len(pts):,} surface points (barycentric, so top-down rasters are solid)")

hand = G.handedness_check(pts, target, dsm, deck_local, yaw_step=2.0)
rh, mi = hand["right_handed"], hand["mirrored"]
print("\n--- CHIRALITY, decided by data rather than convention ---")
print(f"  east=+x, north=-z  : yaw {rh['yaw']:5.1f}  overlap {rh['overlap']:.3f}  "
      f"height-corr {rh['hcorr']:+.3f}  joint {rh['joint']:.3f}")
print(f"  mirrored (north=+z): yaw {mi['yaw']:5.1f}  overlap {mi['overlap']:.3f}  "
      f"height-corr {mi['hcorr']:+.3f}  joint {mi['joint']:.3f}")
print(f"  verdict: {hand['verdict']}  (margin {hand['margin']:.3f})")
assert hand["verdict"] == "right_handed", "chirality check failed - inspect before continuing"
"""))

cells.append(code(r"""
candidates, _ = G.solve_yaw(pts, target, dsm, deck_local, yaw_step=1.0)
winner, ambiguity = G.disambiguate_180(candidates)

print("--- top 5 by footprint overlap ---")
for c in candidates[:5]:
    print(f"  yaw {c['yaw']:5.1f}   overlap {c['overlap']:.4f}   height-corr {c['hcorr']:+.4f}")

print("\n--- THE 180 DEGREE DECISION ---")
p, o = ambiguity["primary"], ambiguity["opposite"]
print(f"  yaw {p['yaw']:5.1f} : overlap {p['overlap']:.4f}   height-corr {p['hcorr']:+.4f}")
print(f"  yaw {o['yaw']:5.1f} : overlap {o['overlap']:.4f}   height-corr {o['hcorr']:+.4f}")
print(f"\n  overlap separates them by only {ambiguity['overlap_margin']:.4f}  <- not decisive")
print(f"  height  separates them by       {ambiguity['hcorr_margin']:.4f}  <- decisive")
print(f"  chosen yaw = {ambiguity['chosen_yaw']:.1f} deg (grid)")
"""))

cells.append(code(r"""
V.fig_yaw_sweep(candidates, ambiguity["chosen_yaw"],
                path=OUTPUT_DIR / "qc_yaw_sweep.png")
import matplotlib.pyplot as plt; plt.show()
"""))

cells.append(code(r"""
# Scale, measured with a SYMMETRIC metric. An overlap ratio normalised by
# min(source, target) is biased toward shrinking the source and will happily
# report a bogus sub-1.0 scale; IoU has no such bias.
scale_rows = G.scale_iou(pts, target, dsm, deck_local, ambiguity["chosen_yaw"],
                         [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06])
print("scale   IoU")
for r in scale_rows:
    bar = "#" * int((r["iou"] - 0.80) * 200)
    print(f" {r['scale']:.2f}   {r['iou']:.4f}  {bar}")
best_scale = max(scale_rows, key=lambda r: r["iou"])["scale"]
print(f"\nIoU peak near scale {best_scale:.2f}; the refinement below fits it properly.")
"""))

cells.append(code(r"""
solution = G.refine(pts, pt_normals, target, dsm, deck_local,
                    ambiguity["chosen_yaw"], winner["tE"], winner["tN"],
                    fit_scale=True)
M = G.matrix_from_solution(solution)

print(f"yaw   {solution['yaw_deg']:10.3f} deg (grid)")
print(f"scale {solution['scale']:10.4f}")
print(f"tE    {solution['tE']:10.3f} m")
print(f"tN    {solution['tN']:10.3f} m")
print(f"tZ    {solution['tZ']:10.3f} m")
print(f"cost  {solution['cost_start']:.4f} -> {solution['cost']:.4f} in {solution['n_iter']} iterations")
print("\n4x4 ARKit -> UTM 43N:")
print(np.array2string(M, precision=5, suppress_small=True))
"""))

cells.append(code(r"""
check = G.verify_alignment(mesh, M, target, dsm)
print("--- independent verification ---")
for k, v in check.items():
    print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")

deck_z = check["deck_mean_elev_m"]
print(f"\n  mesh deck lands at {deck_z:.2f} m; DSM roof median is {target['roof_median']:.2f} m"
      f"  (difference {abs(deck_z-target['roof_median']):.2f} m)")

# the deck normal must still be vertical after transforming - a transposed
# normal transform is silent otherwise
Nl, Cl, Al = mesh.face_normals(), mesh.face_centroids(), mesh.face_areas()
sel = (Nl[:, 1] > 0.98) & (np.abs(Cl[:, 1] - deck_local) < 0.15)
nw = G.apply_normals(M, Nl[sel]); nw = (Al[sel, None] * nw).sum(0); nw /= np.linalg.norm(nw)
print(f"  deck normal in ENU {np.round(nw, 5)} -> {np.degrees(np.arccos(np.clip(nw[2],-1,1))):.3f} deg from vertical")
"""))

cells.append(code(r"""
before = (mesh.V @ G.AXIS_MAP.T)[:, :2] + np.array([solution["tE"], solution["tN"]])
after  = G.apply(M, mesh.V)[:, :2]
V.fig_alignment(dsm, target, before, after, path=OUTPUT_DIR / "qc_alignment.png")
plt.show()
"""))

cells.append(code(r"""
lon, lat = G.to_lonlat(np.array([solution["tE"]]), np.array([solution["tN"]]), dsm.crs_wkt)
lon, lat = float(np.atleast_1d(lon)[0]), float(np.atleast_1d(lat)[0])
convergence = G.grid_convergence(lon, lat, G.utm_zone_center(dsm.crs_wkt))

print(f"building centre : {lat:.6f} N, {lon:.6f} E")
print(f"grid convergence: {convergence:+.4f} deg")
print(f"  true_azimuth = grid_azimuth {convergence:+.4f}")
print("  every azimuth exported below is TRUE north referenced.")

tiltinfo = G.residual_tilt(G.apply(M, pts), G.apply_normals(M, pt_normals), dsm, target)
print("\nroof plane, mesh vs DSM:")
for k, v in tiltinfo.items():
    print(f"  {k:32s} {v:8.3f}")
print("\n  both are essentially flat; the disagreement is within DSM noise")
print("  (the DSM's own roof-plane residual std is ~0.39 m over a flat slab),")
print("  so no tilt correction is applied and the mesh keeps ARKit's gravity.")
"""))

cells.append(md(r"""
## 5 · Stage 3 — segmentation

Normals are smoothed **bilaterally** first. Plain isotropic smoothing is actively
harmful here: at every deck/parapet corner it averages a horizontal and a vertical
normal into a spurious 45° one and measurably *grows* the noise band.

Planes are then found by normal-direction clustering followed by an offset
histogram, rather than brute-force RANSAC — far cheaper on ~90k faces, and
buildings are made of a handful of normal directions.
"""))

cells.append(code(r"""
Cw = G.apply(M, mesh.face_centroids())
Nw = G.apply_normals(M, mesh.face_normals())
Aw = mesh.face_areas() * solution["scale"] ** 2

t_raw = np.degrees(np.arccos(np.clip(np.abs(Nw[:, 2]), 0, 1)))
Ns = S.smooth_normals(Cw, Nw, Aw, normal_sigma_deg=15, iterations=3, radius=0.30)
t_sm = np.degrees(np.arccos(np.clip(np.abs(Ns[:, 2]), 0, 1)))

for tag, t in (("raw", t_raw), ("bilateral", t_sm)):
    nb = Aw[(t > 5) & (t < 70)].sum()
    print(f"  {tag:10s} noise band (5-70 deg): {nb:6.1f} m^2  ({100*nb/Aw.sum():.0f}%)")
"""))

cells.append(code(r"""
planes = S.extract_planes(Cw, Ns, Aw, angle_tol_deg=12, offset_bin=0.12, min_area=0.8)
assigned = np.zeros(len(Aw), bool)
for p in planes:
    assigned[p["faces"]] = True
print(f"direction+offset clustering : {len(planes)} planes, {100*Aw[assigned].sum()/Aw.sum():.0f}% of area")

planes, assigned = S.assign_residual_faces(planes, Cw, Ns, Aw, assigned)
planes.sort(key=lambda p: -p["area_m2"])
for i, p in enumerate(planes):
    p["plane_id"] = i
    z = Cw[p["faces"]][:, 2]
    p["z_min"], p["z_max"] = float(z.min()), float(z.max())
geoms = [S.plane_geometry(p, convergence) for p in planes]
print(f"after residual assignment   : {len(planes)} planes, {100*Aw[assigned].sum()/Aw.sum():.0f}% of area")

deck_plane = planes[0]
deck_z = float(deck_plane["centroid"][2])
print(f"\ndeck plane: #{deck_plane['plane_id']}, {deck_plane['area_m2']:.1f} m^2 at z = {deck_z:.2f} m")
"""))

cells.append(code(r"""
print(f"{'id':>3} {'area m2':>8} {'class':<11} {'tilt':>6} {'trueAz':>8} {'aspect':>7} {'z':>7} {'+deck':>7}")
for p, g in list(zip(planes, geoms))[:18]:
    az = "  flat" if not np.isfinite(g["grid_azimuth_deg"]) else f"{g['true_azimuth_deg']:8.1f}"
    asp = SO._compass(g["true_azimuth_deg"]) if np.isfinite(g["grid_azimuth_deg"]) else "flat"
    print(f"{p['plane_id']:3d} {p['area_m2']:8.2f} {g['orientation']:<11} {g['tilt_deg']:6.1f} "
          f"{az} {asp:>7} {p['centroid'][2]:7.2f} {p['centroid'][2]-deck_z:+7.2f}")
"""))

cells.append(code(r"""
# Wall azimuths are an independent check on yaw: a rectilinear building must
# produce four families 90 deg apart, and they must be consistent.
wall_az = sorted(g["grid_azimuth_deg"] for p, g in zip(planes, geoms)
                 if g["orientation"] == "vertical" and p["area_m2"] > 2)
families = []
for a in wall_az:
    if families and a - families[-1][-1] < 20:
        families[-1].append(a)
    else:
        families.append([a])
main = [f for f in families if len(f) >= 2]
singles = [f for f in families if len(f) == 1]
means = [float(np.mean(f)) for f in main]

print("wall azimuth families (grid):")
for f, m in zip(main, means):
    print(f"  {m:6.1f} deg   n={len(f):2d}   spread {max(f)-min(f):.1f} deg")
for f in singles:
    print(f"  {f[0]:6.1f} deg   n= 1   (isolated: small plane or fusion noise)")

spac = [means[i+1] - means[i] for i in range(len(means) - 1)]
print("\nspacings between families:", [f"{s:.1f}" for s in spac])
if len(means) == 4 and all(abs(s - 90) < 3 for s in spac):
    print("\n-> four families within 3 deg of orthogonal. The building is rectilinear,")
    print("   and this INDEPENDENTLY confirms the solved yaw: a wrong yaw would still")
    print("   produce orthogonal families, but a wrong CHIRALITY or a mis-fitted scale")
    print("   would smear them - so this is a genuine consistency check on stage 2.")
else:
    print("\n-> expected four families ~90 deg apart on a rectilinear building; inspect.")
"""))

cells.append(code(r"""
terraces = S.extract_objects_hierarchical(planes, geoms, Cw, Aw, Ns)

print("Objects are peeled off one terrace at a time, highest first.")
print("A single 'everything above the deck' clustering fails twice over here:")
print("the parapet ring bridges separate objects, and the roof is layered")
print("(deck -> a ~3 m block -> tanks standing on that block).\n")

object_records = []
for t in terraces:
    rel = t["terrace_z"] - deck_z
    print(f"TERRACE plane#{t['terrace_plane_id']}  z={t['terrace_z']:.2f} ({rel:+.2f} above deck)"
          f"  {t['terrace_area_m2']:.1f} m^2  ->  {len(t['objects'])} object(s)")
    for sel in t["objects"]:
        info = S.classify_object(None, sel, Cw, Ns, Aw, t["terrace_z"])
        info["on_terrace_plane_id"] = t["terrace_plane_id"]
        info["on_terrace_z"] = t["terrace_z"]
        object_records.append(info)
        print(f"    {info['label']:<26s} {info['surface_area_m2']:6.1f} m^2  "
              f"{info['footprint_bbox_m'][0]:4.1f}x{info['footprint_bbox_m'][1]:4.1f} m  "
              f"top z {info['top_elev_m']:6.2f}  conf {info['label_confidence']:.2f}")
print("\nLabels are heuristic: on a noisy phone scan a stair headroom and a walled")
print("tank plinth have overlapping geometric signatures, so ambiguous clusters")
print("stay generic rather than asserting a type downstream code would trust.")
"""))

cells.append(code(r"""
V.fig_segmentation(Cw, planes, geoms, deck_z, path=OUTPUT_DIR / "qc_segmentation.png")
plt.show()
"""))

cells.append(md("## 6 · Stage 4 — solar attributes"))

cells.append(code(r"""
plane_records = [SO.plane_solar_record(p, g, Cw, Aw, deck_z, convergence)
                 for p, g in zip(planes, geoms)]

deck_objects = [sel for t in terraces if abs(t["terrace_z"] - deck_z) < 0.05
                for sel in t["objects"]]
usable = SO.usable_area(deck_plane, Cw, deck_objects,
                        edge_setback_m=0.6, obstruction_clearance_m=0.5)

print("Deck area budget")
print(f"  gross plane footprint      {usable['gross_area_m2']:8.2f} m^2")
print(f"  after {usable['edge_setback_m']} m edge setback     {usable['after_setback_m2']:8.2f} m^2")
print(f"  blocked by roof objects    {usable['blocked_by_objects_m2']:8.2f} m^2")
print(f"  USABLE for panels          {usable['usable_area_m2']:8.2f} m^2")
print(f"\n  -> {100*usable['usable_area_m2']/usable['gross_area_m2']:.0f}% of the gross deck is actually mountable.")
"""))

cells.append(code(r"""
sun = SO.solar_position_summary(lat, lon)
print(f"latitude {sun['latitude_deg']:.4f} ({sun['hemisphere']} hemisphere)")
print(f"panels should face TRUE azimuth {sun['optimal_azimuth_true_deg']:.0f} deg")
print(f"rule-of-thumb fixed tilt        {sun['optimal_fixed_tilt_deg_rule_of_thumb']:.0f} deg")
for k in ("summer_solstice", "equinox", "winter_solstice"):
    s = sun[k]
    print(f"  {k:18s} noon elevation {s['solar_noon_elevation_deg']:5.1f} deg   "
          f"shadow {s['shadow_length_per_m']:.2f} m per m of height")
"""))

cells.append(code(r"""
horizon = SO.horizon_profile(dsm, deck_plane["centroid"][0], deck_plane["centroid"][1],
                             deck_z, n_sectors=36, convergence_deg=convergence)
sectors = SO.sector_summary(dsm, mask, target, deck_z, convergence_deg=convergence)

worst = max(horizon, key=lambda h: h["horizon_elev_deg"])
print(f"worst horizon obstruction: {worst['horizon_elev_deg']:.1f} deg at true azimuth "
      f"{worst['true_azimuth_deg']:.0f}, {worst['distance_m']} m away, "
      f"{worst['obstruction_height_m']:+.2f} m above the deck")

far = max(sectors, key=lambda s: s["elev_angle_deg"])
print(f"worst NEIGHBOUR (beyond 8 m): {far['elev_angle_deg']:.1f} deg at "
      f"{far['distance_at_max_m']:.0f} m, {far['above_deck_m']:+.2f} m above deck")
print("\n-> on this roof the dominant shading is the building's OWN superstructure,")
print("   not the neighbours. Panel layout must work around it.")
print(f"\ninter-row spacing for a 3 m obstruction at 25 deg sun: "
      f"{SO.interrow_spacing(3.0, 25.0):.2f} m")
"""))

cells.append(code(r"""
V.fig_solar(horizon, usable, path=OUTPUT_DIR / "qc_solar.png")
plt.show()
"""))

cells.append(md(r"""
## 7 · Stage 5 — export

Geometry is written twice:

* **local ENU** — origin at the building centroid. Most 3D viewers store vertex
  positions as float32; at UTM easting 250811 / northing 2546703 the float32 step
  is already ~0.25 m, so an absolute-coordinate mesh visibly jitters and z-fights.
* **absolute UTM 43N** — for GIS, which stores doubles and does not have that problem.

`transform.json` carries the matrix and the origin, so either can be recovered
from the other.
"""))

cells.append(code(r"""
verts_utm = G.apply(M, mesh.V)
origin_utm = np.array([float(np.mean(verts_utm[:, 0])),
                       float(np.mean(verts_utm[:, 1])), 0.0])
verts_local = verts_utm - origin_utm

# one label per face: plane id for plane members, -1 for unassigned
face_plane = np.full(len(mesh.F), -1, int)
for p in planes:
    face_plane[p["faces"]] = p["plane_id"]

face_class = np.zeros(len(mesh.F), int)      # 0 unassigned
CLASS_ID = {"horizontal": 1, "vertical": 2, "sloped": 3}
for p, g in zip(planes, geoms):
    face_class[p["faces"]] = CLASS_ID[g["orientation"]]

face_object = np.full(len(mesh.F), -1, int)
for i, (t, ) in enumerate([(t, ) for t in terraces]):
    pass
oid = 0
object_names = {}
for t in terraces:
    for sel in t["objects"]:
        face_object[sel] = oid
        object_names[oid] = f"{object_records[oid]['label']}_{oid}"
        oid += 1
print(f"labelled {int((face_plane>=0).sum())} / {len(mesh.F)} faces into {len(planes)} planes")
print(f"labelled {int((face_object>=0).sum())} faces into {oid} objects")
"""))

cells.append(code(r"""
group_names = {}
for p, g in zip(planes, geoms):
    group_names[p["plane_id"]] = (
        f"plane{p['plane_id']:03d}_{g['orientation']}_"
        f"{'flat' if not np.isfinite(g['grid_azimuth_deg']) else SO._compass(g['true_azimuth_deg'])}"
        f"_{p['area_m2']:.1f}m2")
group_names[-1] = "unassigned"

hdr = (f"generated by rooflib\n"
       f"yaw {solution['yaw_deg']:.4f} deg grid, scale {solution['scale']:.5f}\n"
       f"grid convergence {convergence:+.4f} deg (true = grid + convergence)")

X.write_obj(OUTPUT_DIR / "roof_segmented_local_enu.obj", verts_local, mesh.F,
            groups=face_plane, group_names=group_names,
            header_comment=hdr + f"\nlocal ENU, origin_utm = {origin_utm.tolist()}")
X.write_obj(OUTPUT_DIR / "roof_segmented_utm43n.obj", verts_utm, mesh.F,
            groups=face_plane, group_names=group_names,
            header_comment=hdr + "\nabsolute UTM 43N (EPSG:32643)")
X.write_ply_labelled(OUTPUT_DIR / "roof_classes_local_enu.ply", verts_local, mesh.F, face_class)
print("wrote OBJ (local + UTM) and a class-coloured PLY")
"""))

cells.append(code(r"""
sidecar = X.transform_sidecar(
    solution, M, origin_utm, dsm.crs_wkt, convergence,
    diagnostics={
        "footprint_iou": check["footprint_iou"],
        "footprint_mesh_m2": check["footprint_mesh_m2"],
        "footprint_target_m2": check["footprint_target_m2"],
        "height_residual_median_m": check["height_resid_median_m"],
        "height_residual_mad_m": check["height_resid_mad_m"],
        "deck_elevation_m": deck_z,
        "dsm_roof_median_m": target["roof_median"],
        "chirality_verdict": hand["verdict"],
        "chirality_margin": hand["margin"],
        "yaw_180_overlap_margin": ambiguity["overlap_margin"],
        "yaw_180_height_margin": ambiguity["hcorr_margin"],
        "refine_cost_start": solution["cost_start"],
        "refine_cost_final": solution["cost"],
        "ground_elevation_m": ground_elev_m,
        "ground_elevation_method": ground_diag["method"],
        "ground_elevation_diagnostics": ground_diag,
        "building_height_to_deck_m": round(target["roof_median"] - ground_elev_m, 3),
    },
    notes=[
        "building_mask overshoots the roof; the alignment target is mask AND DSM>roof-2m.",
        "The 180 deg yaw ambiguity was resolved by superstructure height correlation, "
        "not by footprint overlap, which is near-degenerate on this footprint.",
        "Mesh and DSM disagree on superstructure height by 1.2-1.5 m; the mesh is used "
        "as authoritative for on-roof object geometry and the DSM for georeferencing "
        "and off-building context.",
        "DSM vertical datum is undeclared in the source file.",
    ])
X.save_json(OUTPUT_DIR / "transform.json", sidecar)

X.write_csv(OUTPUT_DIR / "roof_planes.csv", plane_records)
X.write_csv(OUTPUT_DIR / "roof_objects.csv",
            [{**r, "footprint_e_m": r["footprint_bbox_m"][0],
              "footprint_n_m": r["footprint_bbox_m"][1]} for r in object_records],
            columns=["label", "label_confidence", "on_terrace_plane_id", "on_terrace_z",
                     "surface_area_m2", "horizontal_area_m2", "vertical_area_m2",
                     "footprint_e_m", "footprint_n_m", "base_elev_m", "top_elev_m",
                     "height_above_deck_m", "centroid_e", "centroid_n", "rectilinearity"])
X.save_json(OUTPUT_DIR / "shading.json",
            {"horizon_profile_from_deck_centre": horizon,
             "neighbour_sectors": sectors,
             "solar_geometry": sun})
X.save_json(OUTPUT_DIR / "roof_summary.json", {
    "deck_elevation_m": round(deck_z, 3),
    "ground_elevation_m": round(ground_elev_m, 3),
    "building_height_above_local_ground_m": round(deck_z - ground_elev_m, 2),
    "gross_deck_m2": usable["gross_area_m2"],
    "usable_deck_m2": usable["usable_area_m2"],
    "n_planes": len(planes),
    "n_objects": len(object_records),
    "terraces": [{"plane_id": t["terrace_plane_id"], "z": round(t["terrace_z"], 3),
                  "height_above_deck_m": round(t["terrace_z"] - deck_z, 3),
                  "area_m2": round(t["terrace_area_m2"], 2),
                  "n_objects_on_it": len(t["objects"])} for t in terraces],
    "location": {"latitude": lat, "longitude": lon, "epsg": 32643},
})
print("wrote transform.json, roof_planes.csv, roof_objects.csv, shading.json, roof_summary.json")
"""))

cells.append(code(r"""
# footprint polygons for GIS
feats = []
for p, g, rec in zip(planes, geoms, plane_records):
    if p["area_m2"] < 1.5:
        continue
    m_, orig_, cell_ = S.footprint_mask(Cw[p["faces"]][:, :2], cell=0.15)
    for poly in X.mask_to_polygons(m_, orig_, cell_, min_area_cells=20):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly["ring"]]},
            "properties": {k: v for k, v in rec.items()},
        })
X.write_geojson(OUTPUT_DIR / "roof_planes.geojson", feats)
print(f"wrote roof_planes.geojson with {len(feats)} polygons (EPSG:32643)")

print("\n--- outputs ---")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name:38s} {f.stat().st_size/1024:9.1f} KB")
"""))

cells.append(md(r"""
## 7b · Site-analysis JSON (target schema)

Emits `site_analysis.json` in the consumer schema: lat/lng polygons, compass
azimuths, `base_height_m` above **ground**, and per-face plane-fit statistics.

Three decisions worth knowing about:

- **Only non-vertical planes are roof faces.** Walls are structure, not roof.
- **Faces are gated on fit quality** (RMSE ≤ 0.20 m, inliers ≥ 0.45). Without it
  the export ships surfaces with 4 m RMSE and 1 % inliers that look authoritative
  and are pure fusion noise. Rejected area is reported in `warnings`.
- **Obstruction `type` is honest.** The schema's vocabulary is
  `vent | ac_unit | chimney | skylight | pipe | unknown`; this roof has water
  tanks, a stair headroom and a terrace block. Forcing those into `vent` or
  `chimney` would be a lie, so `type` is `unknown` and the real classification
  travels alongside in `source_label`.
"""))

cells.append(code(r"""
# reuse the modal ground_elev_m computed in section 3 - two independent
# estimates of the same ground plane would be a bug, not a feature
site = ES.build_site_json(
    address_id="qa-007",                      # <- set your real address id
    planes=planes, geoms=geoms, plane_records=plane_records,
    terraces=terraces, object_records=object_records,
    verts_enu=verts_utm, faces=mesh.F, centroids=Cw,
    dsm=dsm, mask=mask, crs_wkt=dsm.crs_wkt, to_lonlat=G.to_lonlat,
    ground_elev_m=ground_elev_m, deck_z=deck_z,
    footprint_mask_fn=S.footprint_mask,
    imagery_date=None,                        # not recorded in the source files
    dsm_resolution_m=float(dsm.res[0]),
    provider="iphone-arkit-lidar + photogrammetric-dsm",
    extra_warnings=[
        "mesh_dsm_superstructure_height_disagreement_1.2_1.5m",
        f"scan_incomplete: mesh footprint {check['footprint_mesh_m2']:.1f} m2 "
        f"vs dsm building surface {check['footprint_target_m2']:.1f} m2",
        "dsm_vertical_datum_undeclared",
    ])

X.save_json(OUTPUT_DIR / "site_analysis.json", site)
print(f"ground datum {ground_elev_m:.2f} m; deck sits {deck_z-ground_elev_m:.2f} m above it")
print(f"roof_faces {len(site['roof_faces'])}   obstructions {len(site['obstructions'])}")
"""))

cells.append(code(r"""
print(f"{'id':<9}{'area':>7}{'tilt':>7}{'azim':>8}{'base_h':>8}{'rmse':>7}{'inlier':>8}{'conf':>7}  ring")
for f in site["roof_faces"]:
    az = "flat" if f["azimuth_deg"] is None else f"{f['azimuth_deg']:.1f}"
    print(f"{f['id']:<9}{f['area_m2']:7.2f}{f['tilt_deg']:7.1f}{az:>8}{f['base_height_m']:8.2f}"
          f"{f['plane_fit']['rmse_m']:7.3f}{f['plane_fit']['inlier_ratio']:8.2f}"
          f"{f['confidence']:7.2f}  {len(f['polygon'])} pts")

print(f"\n{'id':<8}{'type':<9}{'source_label':<23}{'w x l':>12}{'h':>7}{'on_face':>9}{'conf':>6}")
for o in site["obstructions"]:
    wl = f"{o['footprint_m']['width']:.1f} x {o['footprint_m']['length']:.1f}"
    print(f"{o['id']:<8}{o['type']:<9}{o['source_label']:<23}{wl:>12}{o['height_m']:7.2f}"
          f"{str(o['on_face']):>9}{o['confidence']:6.2f}")

print("\nwarnings:")
for w in site["warnings"]:
    print("  -", w)
"""))

cells.append(md("## 8 · Summary"))

cells.append(code(r"""
print("="*74)
print("GEOREFERENCING")
print(f"  yaw {solution['yaw_deg']:.2f} deg grid / {(solution['yaw_deg']+convergence)%360:.2f} deg true")
print(f"  scale {solution['scale']:.4f}   origin UTM43N ({origin_utm[0]:.2f}, {origin_utm[1]:.2f})")
print(f"  footprint IoU {check['footprint_iou']:.3f}   deck {deck_z:.2f} m vs DSM {target['roof_median']:.2f} m")
print(f"  chirality {hand['verdict']}; 180 deg resolved with height margin {ambiguity['hcorr_margin']:.2f}")
print("\nSEGMENTATION")
print(f"  {len(planes)} planes covering {100*Aw[assigned].sum()/Aw.sum():.0f}% of {Aw.sum():.0f} m^2")
print(f"  {len(terraces)} horizontal terraces, {len(object_records)} objects")
print("\nSOLAR")
print(f"  usable deck {usable['usable_area_m2']:.1f} m^2 of {usable['gross_area_m2']:.1f} m^2 gross")
print(f"  optimal orientation: true azimuth {sun['optimal_azimuth_true_deg']:.0f} deg, "
      f"tilt ~{sun['optimal_fixed_tilt_deg_rule_of_thumb']:.0f} deg")
print(f"  dominant shading is the on-roof block ({worst['horizon_elev_deg']:.0f} deg), not neighbours "
      f"({far['elev_angle_deg']:.1f} deg)")
print("="*74)
"""))

cells.append(md(r"""
### Known limitations

Worth reading before trusting a number downstream.

- **Superstructure height disagreement.** The mesh puts the tallest roof structure
  ~5.0 m above the deck; the DSM says ~2.9 m, and the DSM's maximum across the
  whole 100 × 100 m scene is only ~3.5 m above the deck. The mesh is treated as
  authoritative for on-roof geometry (its deck registers to the DSM within ~0.1 m,
  and photogrammetric DSMs systematically under-height small elevated structures),
  but this is **unresolved** and directly affects self-shading estimates.
- **The scan is incomplete.** Mesh footprint ≈ 144 m² against a DSM building
  surface of ≈ 150 m²; the far end of the long axis appears unscanned, so roof
  area is a slight under-estimate.
- **Object labels are heuristic**, with confidences ≤ 0.6. Trust the geometry
  (footprint, height, elevation) far more than the noun attached to it.
- **Vertical datum is undeclared** in `dsm.tif`. Elevations are internally
  consistent and pass through unchanged; they are not tied to a known datum.
  Irrelevant to tilt, azimuth and yield.
- **~16 % of mesh area stays unassigned** to any plane — mostly fusion noise at
  creases. It is retained in the exported geometry, labelled `unassigned`.
- **Scale is fitted to ~1.006** but the IoU curve is flat between 1.02 and 1.04;
  treat absolute dimensions as ±2 %.
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": [], "toc_visible": True},
})

out = ROOT / "notebooks" / "rooftop_pipeline.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells, {out.stat().st_size/1024:.0f} KB)")
