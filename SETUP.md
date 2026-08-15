# Setup: Colab, Drive, and VS Code

## What I could not do for you

I have no browser and no access to your Google credentials, so **I could not log
into your Colab account, upload anything to your Drive, or authorise the Drive
mount.** Those three steps need you. Everything else is done and tested.

There is also **no official "Colab in VS Code" integration** — Google does not
publish one. Section 3 gives you the two arrangements that actually work, and
says plainly what each costs.

---

## 1 · Upload the inputs to Drive

I packaged all four inputs into a single file so this is one upload rather than
four:

```
iphone-lidar-inputs.zip          21 MB
```

Steps:

1. Open <https://drive.google.com>
2. Create a folder in **My Drive** named exactly `iphone-lidar`
3. Drag `iphone-lidar-inputs.zip` into it

That is all. The notebook detects the bundle and unpacks it into
`iphone-lidar/inputs/` on first run — I tested that path end-to-end from a
completely empty directory.

If you would rather upload the four files individually, create
`iphone-lidar/inputs/` and put `lidar_object.zip`, `dsm.tif`, `rgb.tif` and
`building_mask.tif` there. The notebook accepts either layout.

Final Drive layout after the first run:

```
My Drive/
└── iphone-lidar/
    ├── iphone-lidar-inputs.zip
    ├── inputs/          (unpacked automatically)
    └── outputs/         (written by the notebook)
```

---

## 2 · Run the notebook in Colab

1. Go to <https://colab.research.google.com>
2. **File → Upload notebook** → choose `notebooks/rooftop_pipeline.ipynb`
3. **Runtime → Run all**
4. At the Drive cell, Colab opens a Google auth popup — pick your account and
   allow access. This is the step only you can do.

No GPU needed; a standard CPU runtime is fine. End-to-end runtime is about
**3–4 minutes**, most of it in the yaw sweep and normal smoothing.

The notebook is **self-contained** — it writes its own `rooflib` package via
`%%writefile` cells, so there is nothing to `pip install` from anywhere private
and nothing else to upload. The only external install is `rasterio`, which the
first cell handles.

### What you should see

The run is instrumented so you can check it rather than trust it:

- Chirality verdict `right_handed`, with the margin printed
- The 180° table: overlap margin ~0.006 (not decisive) vs height margin ~1.27 (decisive)
- Footprint IoU ~0.906, deck within ~0.11 m of the DSM roof median
- Wall azimuth families ~90° apart
- Four QC figures

If the chirality assert trips or the IoU comes out low, stop and look — the
inputs are probably not the ones this was tuned against.

### Outputs

Written to `My Drive/iphone-lidar/outputs/`:

| File | Contents |
|---|---|
| `roof_segmented_local_enu.obj` | Segmented mesh, origin at building centroid |
| `roof_segmented_utm43n.obj` | Same geometry, absolute EPSG:32643 |
| `roof_classes_local_enu.ply` | Class-coloured, opens in MeshLab/CloudCompare |
| `transform.json` | 4×4 matrix, origin, CRS, convergence, diagnostics |
| `roof_planes.csv` | Per-plane tilt, true azimuth, areas, elevations |
| `roof_objects.csv` | Per-object label, confidence, footprint, heights |
| `roof_planes.geojson` | Plane footprints for GIS (EPSG:32643) |
| `shading.json` | Horizon profile, neighbour sectors, solar geometry |
| `roof_summary.json` | Headline numbers |
| `qc_*.png` | Four QC figures |

**Why two OBJ files.** Most 3D viewers store vertex positions as float32. At
easting 250811 / northing 2546703 the float32 step is already ~0.25 m, so an
absolute-coordinate mesh visibly jitters and z-fights. Local ENU keeps the
numbers small for viewing; the UTM copy is for GIS, which uses doubles.
`transform.json` converts between them.

---

## 3 · VS Code

Pick one of these. I'd recommend the first.

### Option A — edit locally, run in Colab (recommended)

VS Code is the editor, Colab is the runtime, Drive is the sync.

1. Install **Google Drive for Desktop**: <https://www.google.com/drive/download/>
2. It mounts your Drive as a drive letter on Windows (usually `G:`)
3. Open the notebook from there in VS Code, edit, save
4. Refresh the Colab tab — your edits are there

Install these VS Code extensions (Ctrl+Shift+X):

```
ms-toolsai.jupyter          Jupyter notebook editing
ms-python.python            Python language support
```

You can install both from the command line:

```bash
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.python
```

**Why this one:** no tunnels, nothing to keep alive, and it survives Colab
disconnecting. The cost is that you can't step through a cell in VS Code against
Colab's kernel.

### Option B — just run it locally

This pipeline is small: ~90k triangles and a 1 Mpx raster. **It runs fine on your
machine** — that is exactly how I built and tested it. Colab buys you nothing here
except not having to install anything.

I already created a working venv for you:

```bash
cd ~/projects/iphone-lidar
.venv/bin/python -c "import rasterio, scipy, matplotlib; print('ok')"
```

To run the notebook locally in VS Code:

1. Open `notebooks/rooftop_pipeline.ipynb`
2. Click **Select Kernel** (top right) → **Python Environments** →
   `.venv/bin/python`
3. Run all

It detects it is not in Colab, skips the Drive mount, and falls back to local
paths (override with the `ROOFLIB_BASE` environment variable).

To run it headless:

```bash
cd ~/projects/iphone-lidar
.venv/bin/python build_notebook.py     # regenerate the notebook from rooflib/
```

### What about connecting VS Code *to* a Colab runtime?

People do this by running an SSH server inside the Colab VM and tunnelling out
with `cloudflared` or `ngrok`, then attaching VS Code Remote-SSH. It works, but:

- it goes against Colab's terms of use for non-interactive/tunnelled access, and
  accounts do get flagged
- the tunnel dies whenever the runtime recycles, which is often
- you gain nothing for a 4-minute CPU job

I'm not setting that up. Option A or B covers the real need.

---

## 4 · Repo layout

```
iphone-lidar/
├── inputs/                      the four source files
├── iphone-lidar-inputs.zip      single-file Drive upload bundle
├── insights.md                  what the inputs are and how they connect
├── SETUP.md                     this file
├── build_notebook.py            regenerates the notebook from rooflib/
├── notebooks/
│   └── rooftop_pipeline.ipynb   the deliverable — self-contained
├── rooflib/                     the tested library the notebook embeds
│   ├── io_utils.py              raster shim + OBJ reader (no trimesh/open3d)
│   ├── georef.py                4-DOF solve, chirality, 180° disambiguation
│   ├── segment.py               planes, terraces, hierarchical objects
│   ├── solar.py                 tilt/azimuth, usable area, shading
│   ├── exporters.py             OBJ/PLY/GeoJSON/CSV/JSON
│   └── viz.py                   QC figures
├── outputs/                     results from the local test run
└── .venv/                       local Python environment
```

**`rooflib/` is the source of truth.** `build_notebook.py` reads those files and
embeds them into the notebook, so edit the modules and regenerate — don't edit the
embedded copies inside the `.ipynb`, they'll be overwritten.

---

## 5 · Before trusting the numbers

Read the *Known limitations* section at the end of the notebook. The two that
matter most:

- **Mesh and DSM disagree on superstructure height by 1.2–1.5 m**, unresolved.
  This directly affects self-shading, which on this roof is the dominant shading
  term.
- **Object labels are heuristic** (confidence ≤ 0.6). The geometry is solid; the
  nouns are guesses.
