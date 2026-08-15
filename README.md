# iPhone LiDAR Rooftop Pipeline

Takes an iPhone/ARKit rooftop scan plus a co-registered DSM / RGB / building-mask
raster triple and produces a **georeferenced, semantically segmented** roof model
with per-surface pitch, true azimuth, elevation, usable area, and shading.

The deliverable is a single self-contained Jupyter notebook
([`notebooks/rooftop_pipeline.ipynb`](notebooks/rooftop_pipeline.ipynb)) that runs
either in Google Colab or locally, end-to-end in about 3–4 minutes on CPU.

## Pipeline

| Stage | What it does |
|---|---|
| 1 | Load and characterise the four inputs |
| 2 | Solve the ARKit → UTM transform (4-DOF: yaw, E, N, Z + optional scale) |
| 3 | Segment the mesh into planes, terraces and objects |
| 4 | Derive solar attributes: tilt, true azimuth, usable area, shading |
| 5 | Export local-ENU **and** absolute UTM geometry, plus GeoJSON / CSV / JSON |

Three things it is careful about, each of which silently ruins the output if
mishandled:

1. **Chirality** — ARKit is `x` right, `y` up, `z` toward viewer. Mapping to ENU
   as `east=+x, north=−z, up=+y` preserves handedness; `north=+z` would mirror
   the model and flip every azimuth. Stage 2 proves the choice against the data
   instead of assuming it.
2. **The 180° ambiguity** — a roughly rectangular footprint matches its own
   180°-rotated self almost perfectly, so binary footprint overlap can't tell
   yaw from yaw+180. The superstructure **height** map can, and is used to
   disambiguate.
3. **Grid vs. true north** — UTM azimuths are grid azimuths. Solar geometry
   needs true north, so grid convergence is applied to every azimuth that
   leaves the pipeline.

## Inputs

Four files, all pixel-aligned on the same raster grid where applicable
(EPSG:32643 / UTM 43N):

| File | Format | What it is |
|---|---|---|
| `inputs/lidar_object.zip` | OBJ + MTL + JPEG texture atlases | iPhone/ARKit textured mesh of the roof (~56.7k verts, ~89.4k triangles) |
| `inputs/dsm.tif` | Float32 GeoTIFF, 0.1 m GSD | Digital surface model covering the site |
| `inputs/rgb.tif` | 3-band uint8 GeoTIFF, same grid as the DSM | Orthophoto of the site |
| `inputs/building_mask.tif` | Byte {0,1} GeoTIFF, same grid | Mask isolating the building footprint from the surrounding scene |

`inputs/*.tif.aux.xml` are QGIS-generated statistics sidecars, kept alongside
the rasters for convenience — the pipeline itself doesn't read them.

`iphone-lidar-inputs.zip` at the repo root is a convenience bundle of the same
four files, for uploading to Google Drive in a single step when running in
Colab (see [SETUP.md](SETUP.md)).

`inputs/Geo reference las.zip` is a georeferenced LAS point cloud collected
alongside the other inputs, originally intended to supply the heading
(north direction) for an earlier version of the pipeline design. The
implemented pipeline derives yaw from footprint/height matching against the
DSM instead ([`rooflib/georef.py`](rooflib/georef.py)), so this file is **not
read by any code in this repo** — it's kept only as raw provenance data.

## Outputs

Everything below is written to `outputs/` by a pipeline run (already present
in this repo from the last local run):

| File | Contents |
|---|---|
| `roof_segmented_local_enu.obj` | Segmented mesh, origin at the building centroid (small coordinates, safe for float32 viewers) |
| `roof_segmented_utm43n.obj` | Same geometry in absolute EPSG:32643 coordinates (for GIS) |
| `roof_classes_local_enu.ply` | Class-coloured point/mesh export — opens in MeshLab/CloudCompare |
| `transform.json` | The solved 4×4 ARKit→ENU/UTM transform, origin, CRS, grid convergence, and solve diagnostics |
| `roof_planes.csv` | One row per fitted plane: orientation, area, tilt, grid/true azimuth, elevations, height above deck |
| `roof_objects.csv` | One row per detected object: label, confidence, footprint, elevations, height above deck |
| `roof_planes.geojson` | Plane footprints as GIS polygons (EPSG:32643) |
| `shading.json` | Horizon profile from the deck centre, neighbouring obstruction sectors, solar geometry |
| `site_analysis.json` | Roof faces / obstructions / trees in the target site-analysis schema (see [`rooflib/export_schema.py`](rooflib/export_schema.py)) |
| `roof_summary.json` | Headline numbers: deck elevation, building height, gross/usable deck area, plane & object counts, location |
| `qc_alignment.png` | QC figure: mesh-to-DSM alignment |
| `qc_segmentation.png` | QC figure: plane/object segmentation map |
| `qc_solar.png` | QC figure: solar exposure per plane |
| `qc_yaw_sweep.png` | QC figure: the yaw-solve score curve, including the 180° margin |

**Why two OBJ files:** most 3D viewers store vertex positions as float32. At
this site's UTM easting (~250,860) the representable float32 step is already
~0.25 m, so an absolute-coordinate mesh visibly jitters and z-fights. Local ENU
keeps the numbers small for viewing; the UTM copy is for GIS, which uses
doubles. `transform.json` converts between the two.

## Repo layout

```
iphone-lidar/
├── inputs/                       the four source files (+ raw LAS provenance)
├── iphone-lidar-inputs.zip       single-file Drive upload bundle
├── outputs/                      results from the last pipeline run
├── notebooks/
│   └── rooftop_pipeline.ipynb    the deliverable — self-contained, runs in Colab or locally
├── rooflib/                      the tested library the notebook embeds
│   ├── io_utils.py               raster shim (rasterio, falls back to GDAL) + hand-rolled OBJ reader
│   ├── georef.py                 4-DOF solve (yaw, E/N/Z, scale), chirality proof, 180° disambiguation
│   ├── ground.py                 ground-elevation estimate from the DSM (dominant-low-surface method)
│   ├── segment.py                plane fitting, terrace grouping, hierarchical object detection
│   ├── solar.py                  tilt/true-azimuth, usable area, shading geometry
│   ├── exporters.py               OBJ/PLY/GeoJSON/CSV/JSON writers (local-ENU + UTM)
│   ├── export_schema.py          site-analysis JSON schema emitter
│   └── viz.py                    QC figure generation
├── build_notebook.py             regenerates notebooks/rooftop_pipeline.ipynb from rooflib/
├── SETUP.md                      Colab / Drive / VS Code setup walkthrough
└── README.md                     this file
```

**`rooflib/` is the source of truth.** `build_notebook.py` reads those modules
and embeds them into the notebook via `%%writefile` cells, so the notebook
cannot drift from tested code. Edit the modules and run `build_notebook.py` to
regenerate the notebook — don't edit the embedded copies inside the `.ipynb`
directly, they'll be overwritten.

## Running it

Full walkthrough (Colab, Drive upload, and local VS Code setup) is in
[SETUP.md](SETUP.md). Short version for running locally:

```bash
python -m venv .venv
.venv/bin/pip install rasterio scipy matplotlib nbformat numpy
.venv/bin/python build_notebook.py     # regenerate the notebook from rooflib/, optional
```

Then open `notebooks/rooftop_pipeline.ipynb` in VS Code or Jupyter, select the
`.venv` kernel, and run all cells. It detects it isn't running in Colab and
falls back to local paths under the repo root (override with the
`ROOFLIB_BASE` environment variable).

### What a successful run looks like

- Chirality verdict `right_handed`, with the solve margin printed
- The 180° table: footprint-overlap margin near-zero (not decisive) vs.
  height margin clearly decisive
- Footprint IoU ~0.9, deck elevation within a few centimetres of the DSM roof
  median
- Wall azimuth families ~90° apart
- Four QC figures written to `outputs/`

## Known limitations

- **Mesh and DSM disagree on superstructure height by 1.2–1.5 m**, unresolved.
  This directly affects self-shading, which on this roof is the dominant
  shading term.
- **Object labels are heuristic** (confidence ≤ 0.6 in `roof_objects.csv`).
  The geometry is solid; the class names are best-effort guesses.

See the *Known limitations* section at the end of the notebook for the full
list before trusting the numbers for anything downstream.
