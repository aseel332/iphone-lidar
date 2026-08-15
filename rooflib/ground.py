"""Ground-elevation estimation from the DSM.

The naive approach -- `percentile(dsm[outside_mask], 10)` -- is what the
pipeline used at first, and it is wrong on this scene: the 0-10 m ring around
the building is contaminated by an adjacent lower roof sitting right against
the wall, which pulls the estimate up by ~0.5-1 m.

The fix is to find the DOMINANT low surface in the scene, not the nearest one.
Real ground is the largest flat run at the bottom of the elevation histogram;
a 0.1 m-binned histogram of everything outside the building mask has one
unmistakable peak there (46k px / 462 m^2 in the 63.5-63.6 m bin on this
dataset, dwarfing every other bin below 70 m), and a robust median around that
peak is the ground elevation.
"""
from __future__ import annotations

import numpy as np


def estimate_ground_elevation(dsm, mask, bin_m=0.1, band_m=0.6):
    """Modal ground elevation from DSM pixels outside the building mask.

    Returns (ground_m, diagnostics). diagnostics includes the modal bin, the
    pixel/area count backing it, and the cross-checks that would catch this
    estimate being wrong (whole-scene p10, nearest-ring p5/p10) so a caller can
    tell a confident estimate from a shaky one instead of trusting a bare float.
    """
    D = dsm.data.astype(float)
    outside = mask.data != 1
    z = D[outside]
    res2 = dsm.res[0] * dsm.res[1]

    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    hist, _ = np.histogram(z, bins=edges)
    i = int(np.argmax(hist))
    mode = edges[i] + bin_m / 2

    band = z[np.abs(z - mode) < band_m]
    ground = float(np.median(band))

    # Cross-checks: cheap alternative estimators. Large disagreement between
    # these and `ground` is the signal that this scene doesn't have a clean
    # single ground plane and the result needs a human look.
    p10_whole = float(np.percentile(z, 10))
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(outside) * dsm.res[0]
    ring = outside & (dist > 2) & (dist <= 10)
    p5_ring = float(np.percentile(D[ring], 5)) if ring.sum() > 50 else float("nan")

    diagnostics = {
        "method": "modal_surface",
        "bin_m": bin_m,
        "band_m": band_m,
        "modal_bin_lo_m": round(float(edges[i]), 3),
        "modal_bin_hi_m": round(float(edges[i + 1]), 3),
        "modal_bin_px": int(hist[i]),
        "modal_bin_area_m2": round(float(hist[i] * res2), 1),
        "band_px": int(band.size),
        "band_area_m2": round(float(band.size * res2), 1),
        "cross_check_whole_scene_p10_m": round(p10_whole, 3),
        "cross_check_ring_2to10m_p5_m": round(p5_ring, 3),
        "agreement_m": round(abs(ground - p10_whole), 3),
    }
    return ground, diagnostics
