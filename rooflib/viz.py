"""QC figures.

Palette is the validated 3-slot categorical set (blue/orange/aqua); it clears
the all-pairs CVD and normal-vision floors in light mode, which matters here
because the segmentation map is a spatial plot where any pair of classes can
end up adjacent. Aqua sits below 3:1 against the surface, so every class is
also carried by a visible legend label -- identity is never colour-alone.
"""
from __future__ import annotations

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a86"
GRID = "#e3e3df"

# categorical slots, fixed order, never cycled
C1 = "#2a78d6"   # blue
C2 = "#eb6834"   # orange
C3 = "#1baf7a"   # aqua
C4 = "#eda100"   # yellow
C8 = "#e34948"   # red

CLASS_COLORS = {"horizontal": C1, "vertical": C2, "sloped": C3}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8)
    return ax


def fig_yaw_sweep(candidates, chosen_yaw, path=None):
    """Overlap and height-correlation vs yaw.

    Both series are dimensionless and share the -1..1 range, so they go on ONE
    axis -- the point of the figure is precisely that overlap alone is flat
    across the 180 deg pair while the height channel is not.
    """
    import matplotlib.pyplot as plt

    yaw = np.array([c["yaw"] for c in candidates])
    ov = np.array([c["overlap"] for c in candidates])
    hc = np.array([c["hcorr"] for c in candidates])
    o = np.argsort(yaw)
    yaw, ov, hc = yaw[o], ov[o], hc[o]

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    _style(ax, "Yaw search: footprint overlap cannot separate the 180° pair; height correlation can",
           "yaw about vertical (degrees, grid north)", "score (dimensionless, −1…1)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.plot(yaw, ov, color=C1, linewidth=2, label="footprint overlap")
    ax.plot(yaw, hc, color=C2, linewidth=2, label="height correlation")
    opp = (chosen_yaw + 180) % 360
    for x, lab, col in ((chosen_yaw, f"chosen {chosen_yaw:.0f}°", C1),
                        (opp, f"rejected {opp:.0f}°", C8)):
        ax.axvline(x, color=col, linewidth=1.2, linestyle="--", alpha=0.9)
        # inside the axes, not above them -- at y=1.02 this collided with the title
        ax.annotate(lab, (x, 0.94), xycoords=("data", "axes fraction"),
                    color=col, fontsize=8, ha="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xlim(0, 360)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(range(0, 361, 45))
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def fig_alignment(dsm, target, mesh_xy_before, mesh_xy_after, path=None):
    """Mesh footprint against the DSM target, before and after solving."""
    import matplotlib.pyplot as plt

    building = target["building"]
    ys, xs = np.nonzero(building)
    r0, r1 = ys.min() - 25, ys.max() + 26
    c0, c1 = xs.min() - 25, xs.max() + 26
    sub = dsm.data.astype(float)[r0:r1, c0:c1]
    e0, n1 = dsm.xy(np.array([r0]), np.array([c0]))
    e1, n0 = dsm.xy(np.array([r1]), np.array([c1]))
    extent = [float(e0[0]), float(e1[0]), float(n0[0]), float(n1[0])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE)
    for ax, xy, ttl in ((axes[0], mesh_xy_before, "Before: ARKit frame (arbitrary yaw & origin)"),
                        (axes[1], mesh_xy_after, "After: solved 4-DOF transform")):
        _style(ax, ttl, "easting (m, EPSG:32643)", "northing (m)")
        ax.imshow(sub, extent=extent, origin="upper", cmap="Greys_r",
                  alpha=0.85, aspect="equal")
        cs = ax.contour(np.linspace(extent[0], extent[1], sub.shape[1]),
                        np.linspace(extent[3], extent[2], sub.shape[0]),
                        building[r0:r1, c0:c1].astype(float), levels=[0.5],
                        colors=[C3], linewidths=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c=C2, alpha=0.35, linewidths=0,
                   label="mesh footprint")
        ax.plot([], [], color=C3, linewidth=2, label="DSM building target")
        # These axes carry a dark raster underneath, so an unframed legend is
        # unreadable; give it the chart surface as a backing plate.
        leg = ax.legend(fontsize=8, loc="upper right", framealpha=0.92,
                        facecolor=SURFACE, edgecolor=GRID)
        for t in leg.get_texts():
            t.set_color(INK)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def fig_segmentation(centroids, planes, geoms, deck_z, path=None,
                     min_terrace_area=3.0):
    """Plan view coloured by surface class, plus the terrace elevation ladder."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = _style(axes[0], "Segmented surfaces (plan view)",
                "easting (m)", "northing (m)")
    seen = set()
    for p, g in zip(planes, geoms):
        C = centroids[p["faces"]]
        cls = g["orientation"]
        ax.scatter(C[:, 0], C[:, 1], s=1.1, c=CLASS_COLORS[cls], linewidths=0,
                   alpha=0.75, label=cls if cls not in seen else None)
        seen.add(cls)
    ax.set_aspect("equal")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left",
                    markerscale=8, title="surface class")
    leg.get_title().set_color(INK_2)
    leg.get_title().set_fontsize(8)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax = _style(axes[1], f"Horizontal terraces by elevation (≥ {min_terrace_area:g} m²)",
                "plane area (m²)", "height above deck (m)")
    # Threshold matches the one the object hierarchy uses. Plotting every
    # sliver instead produces a pile of colliding labels around +2.5 m.
    terr = [(p, g) for p, g in zip(planes, geoms)
            if g["orientation"] == "horizontal" and p["area_m2"] >= min_terrace_area]
    terr.sort(key=lambda pg: pg[0]["centroid"][2])
    ys = [p["centroid"][2] - deck_z for p, _ in terr]
    ws = [p["area_m2"] for p, _ in terr]
    ax.barh(ys, ws, height=0.13, color=C1, edgecolor=SURFACE, linewidth=1.5)
    for y, w in zip(ys, ws):
        ax.annotate(f"{w:.1f} m²  ·  +{y:.2f} m", (w, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK_2)
    ax.set_xlim(0, max(ws) * 1.5)
    ax.set_ylim(-0.4, max(ys) + 0.5)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def fig_solar(horizon, usable, path=None):
    """Horizon obstruction (polar) and the usable-area map."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 5.2), facecolor=SURFACE)
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_facecolor(SURFACE)
    az = np.radians([h["true_azimuth_deg"] for h in horizon] +
                    [horizon[0]["true_azimuth_deg"]])
    el = np.array([h["horizon_elev_deg"] for h in horizon] +
                  [horizon[0]["horizon_elev_deg"]])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(az, el, color=C2, linewidth=2)
    ax.fill(az, el, color=C2, alpha=0.18)
    ax.set_title("Horizon obstruction from deck centre\n(true-north azimuth, degrees elevation)",
                 color=INK, fontsize=10, pad=14)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_rlabel_position(135)
    ax.spines["polar"].set_color(GRID)

    ax2 = _style(fig.add_subplot(1, 2, 2), None, "easting (m)", "northing (m)")
    m = usable["_mask"]
    e0, n1 = usable["_origin"]
    c = usable["_cell"]
    extent = [e0, e0 + m.shape[1] * c, n1 - m.shape[0] * c, n1]
    ax2.imshow(np.where(m, 1.0, np.nan), extent=extent, origin="upper",
               cmap=_mono_cmap(C1), vmin=0, vmax=1, aspect="equal")
    ax2.set_title(
        f"Deck area available for panels — {usable['usable_area_m2']:.1f} m² "
        f"of {usable['gross_area_m2']:.1f} m² gross",
        color=INK, fontsize=10, loc="left", pad=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def _mono_cmap(hex_color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("m", [hex_color, hex_color])
