"""
Step 5 — Analysis and maps of 3D climate velocity

Companion code to Rizzi et al. (2026)


What this script does
---------------------
For the chosen analogue threshold, it summarises and maps the step-3 results for the whole water column, the surface layer, the seabed and each biozone.
Step 3 gives displacements over the period (ab_dist_fw, delta_xy, delta_z): here they are divided by n_years to obtain velocities (per year):
    - statistics of climate distance and velocity (per year);
    - type of displacement (horizontal only, vertical only, 3D, none);
    - availability of analogues (no analogue cells);
    - direction of horizontal displacement (rose plots, quadrant counts);
    - maps of 3D, horizontal and vertical velocity;
    - similarity between depth layers within each biozone (heatmaps);
    - depth profiles of no analogue / null velocity cells and of the mean vertical shift.


Inputs (in data_dir)
------
- {prefix}_dft1.csv                     : present table with coordinates (step 3)
- 3d_package.gpkg, layer "geodf"        : cell positions (step 2)
- {range}_dist_vel_{threshold}.nc       : step-3 results (full, bottom, depth ranges)
- {range}_no_analogue_{threshold}.csv   : step-3 no-analogue cells
- data/shapefiles/med_sea.shp           : Mediterranean Sea polygon


Outputs (in data_dir)
-------
PNG figures

"""


## IMPORTS
import os
 
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import seaborn as sns
import cartopy.crs as ccrs
import pyproj
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D



## CONFIGURATION ## 
# Must match steps 1-3
prefix       = "3d_2006-2055"
n_years      = 25       # length of each period (years); converts distances to annual rates
threshold    = 0.1      # analogue threshold chosen in step 3
threshold_label = f"t{threshold}".replace(".", "")   # e.g. "t01"
 
data_dir      = "data/outputs"
package_file  = os.path.join(data_dir, "3d_package.gpkg")
med_shapefile = "data/shapefiles/med_sea.shp"
 
# Depth of the surface layer analysed separately (m, a model depth level)
surface_depth = 0.5
 
# Decimals used to match lat/lon when selecting bottom cells (as in step 3)
round_decimals = 1
 
# Depth ranges of step 3 (m), one per biozone
depth_ranges = {
    "euphotic":           (0,    50),
    "mesophotic":         (30,   250),
    "mesopelagic":        (200,  1250),
    "bathyabyssopelagic": (1000, 5500),
}
 
# Depth bounds used to assign each depth level to one biozone in the depth profile figures (m). 
# Levels outside all bounds are left out, except that the bathyabyssopelagic file provides every level down to its maximum.
profile_bounds = {
    "euphotic":           (0,    40),
    "mesophotic":         (50,   200),
    "mesopelagic":        (250,  1000),
    "bathyabyssopelagic": (1250, 5500),
}
profile_bg_colors = ["#f2f8ff", "#eaf5ea", "#f7f0ff", "#fff7ea"]   # one per biozone
 
# Maps: Web Mercator projection
map_crs = ccrs.GOOGLE_MERCATOR
 
# Velocity colour scale: grey for zero, then yellow to red, brown, green and blue
climate_cmap = LinearSegmentedColormap.from_list("climate_warmcool_grey0", [
    (0.00, "#bdbdbd"),
    (0.01, "#ffd700"),
    (0.15, "#fecc5c"),
    (0.30, "#fd8d3c"),
    (0.45, "#e31a1c"),
    (0.60, "#8c510a"),
    (0.75, "#4d9221"),
    (0.90, "#4eb3d3"),
    (1.00, "#2166ac"),
], N=256)
 
# Velocity maps: variable -> title, colour-bar legend, unit, colours, file suffix
velocity_maps = {
    "ab_dist_fw": {"title": "3D climate velocity", "legend": "Forward velocity", "uom": "km/year",
                   "cmap": climate_cmap, "vmin": None, "vmax": 50, "suffix": "vxyz_map"},
    "delta_xy":     {"title": "Horizontal velocity", "legend": "Velocity XY", "uom": "km/year",
                   "cmap": climate_cmap, "vmin": None, "vmax": 50, "suffix": "vxy_map"},
    "delta_z":    {"title": "Vertical velocity", "legend": "Velocity Z", "uom": "m/year",
                   "cmap": "coolwarm", "vmin": -1.5, "vmax": 1.5, "suffix": "vz_map"},
}
 
# Direction quadrants (degrees, 0 = North, clockwise)
quadrant_labels = ["North (NW,N,NE)", "East", "South (SE,S,SW)", "West"]
quadrant_ranges = [(292.5, 67.5), (67.5, 112.5), (112.5, 247.5), (247.5, 292.5)]
 
show_figures = False   # True to display figures while running



## FILE NAMES ##
def velocity_file(range_label):
    """Step-3 result file for a run ("full", "bottom" or "min-max")."""
    return os.path.join(data_dir, f"{range_label}_dist_vel_{threshold_label}.nc")
 
 
def no_analogue_file(range_label):
    """Step-3 no-analogue file for a run ("full", "bottom" or "min-max")."""
    return os.path.join(data_dir, f"{range_label}_no_analogue_{threshold_label}.csv")
 
 
def savepath(label, suffix):
    """Output path of a figure."""
    return os.path.join(data_dir, f"{label}_{suffix}.png")
 
 
def _finish(fig, save):
    """Save, optionally display, and close a figure."""
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)



## ANALYSIS ##
def analyze_dataset(data, missing_coords_df, dft1, mode="full", depth_range=None):
    """Statistics, displacement types, no-analogue and zero-movement masks, and directions for one step-3 result."""
    print(f"\n=== RUNNING ANALYSIS FOR {mode.upper()} ===")
    if mode == "range" and depth_range:
        print(f"Depth range: {depth_range[0]} - {depth_range[1]} m")
 
    has_3d    = "delta_xy" in data and "delta_z" in data
    variables = ["cc_dist", "ab_dist_fw"] + [v for v in ("delta_xy", "delta_z") if v in data]
    results   = {}
 
    # --- Statistics ---
    # Displacements are converted to velocities (per year) and reported under
    # velocity names: ab_dist_fw -> vel_3d (km/yr), delta_xy -> vel_xy (km/yr),
    # delta_z -> vel_z (m/yr). cc_dist is reported as is.
    velocity_names = {"ab_dist_fw": "vel_3d", "delta_xy": "vel_xy", "delta_z": "vel_z"}
 
    print("\n=== DATASET STATISTICS ===")
    for var in variables:
        name  = velocity_names.get(var, var)
        scale = n_years if var in velocity_names else 1
        stats = {
            "mean": data[var].mean().item() / scale,
            "std":  data[var].std().item() / scale,
            "min":  data[var].min().item() / scale,
            "max":  data[var].max().item() / scale,
        }
        for key, value in stats.items():
            results[f"{name}_{key}"] = value
        print(f"{name}: mean = {stats['mean']:.3f}, std = {stats['std']:.3f}, "
              f"min = {stats['min']:.3f}, max = {stats['max']:.3f}")
 
        # Upward (positive) and downward (negative) vertical velocity separately
        if var == "delta_z":
            pos_vals = data[var].where(data[var] > 0)
            neg_vals = data[var].where(data[var] < 0)
            results["vel_z_pos_min"] = pos_vals.min().item() / n_years
            results["vel_z_pos_max"] = pos_vals.max().item() / n_years
            results["vel_z_neg_min"] = neg_vals.min().item() / n_years
            results["vel_z_neg_max"] = neg_vals.max().item() / n_years
            print(f"   upward movement: {results['vel_z_pos_min']:.3f} to {results['vel_z_pos_max']:.3f}")
            print(f"   downward movement: {results['vel_z_neg_min']:.3f} to {results['vel_z_neg_max']:.3f}")
 
        # Cells whose analogue is the cell itself
        if var == "ab_dist_fw":
            results["ab_dist_fw_null"] = (data[var] == 0).sum().item()
            print(f"Points with ab_dist_fw=0 --> {results['ab_dist_fw_null']}")
 
    # --- Displacement types ---
    print("\n=== MOVEMENT TYPES ===")
    mask_valid = data["ab_dist_fw"].notnull()
 
    if has_3d:
        mask_moving = mask_valid & (data["ab_dist_fw"] != 0)
        results.update({
            "count_vertical_only":   (mask_moving & (data["delta_xy"] == 0) & (data["delta_z"] != 0)).sum().item(),
            "count_horizontal_only": (mask_moving & (data["delta_z"] == 0) & (data["delta_xy"] != 0)).sum().item(),
            "count_proper_3d":       (mask_moving & (data["delta_xy"] != 0) & (data["delta_z"] != 0)).sum().item(),
            "count_null_vel":        (mask_valid & (data["delta_xy"] == 0) & (data["delta_z"] == 0)).sum().item(),
            # Consistency check: moving cells with no horizontal or vertical component
            "count_check":           (mask_moving & (data["delta_xy"] == 0) & (data["delta_z"] == 0)).sum().item(),
        })
        print(f"Vertical only --> {results['count_vertical_only']}")
        print(f"Horizontal only --> {results['count_horizontal_only']}")
        print(f"3D movement --> {results['count_proper_3d']}")
        print(f"No movement --> {results['count_null_vel']}")
        print(f"Check --> {results['count_check']} - IT MUST BE 0")
    else:
        results["count_null_vel"] = (mask_valid & (data["ab_dist_fw"] == 0)).sum().item()
        print("Dataset missing 'delta_xy' or 'delta_z'")
        print(f"No movement --> {results['count_null_vel']}")
 
    # --- Analogue availability ---
    results["total_points"]            = len(dft1)
    results["points_with_no_analogue"] = len(missing_coords_df)
    results["points_with_analogue"]    = len(dft1) - len(missing_coords_df)
 
    print("\n=== ANALOGUE AVAILABILITY ===")
    print(f"Total points: {results['total_points']}")
    print(f"Points with analogue: {results['points_with_analogue']}")
    print(f"Points with no analogue: {results['points_with_no_analogue']}")
 
    # --- Masks: no-analogue cells and zero movement ---
    template     = data["delta_xy"] if has_3d else data["ab_dist_fw"]
    missing_mask = xr.full_like(template, False, dtype=bool)
    use_depth    = "depth" in missing_mask.dims and "depth" in missing_coords_df.columns
    for _, row in missing_coords_df.iterrows():
        if use_depth:
            missing_mask.loc[dict(depth=row["depth"], lat=row["lat"], lon=row["lon"])] = True
        else:
            missing_mask.loc[dict(lat=row["lat"], lon=row["lon"])] = True
 
    zero_mask = ((data["delta_xy"] == 0) & (data["delta_z"] == 0)) if has_3d else (data["ab_dist_fw"] == 0)
 
    results["missing_mask"]    = missing_mask
    results["zero_mask"]       = zero_mask
    results["num_no_analogue"] = missing_mask.sum().item()
    results["num_zero"]        = zero_mask.sum().item()
    data["missing_mask"]       = missing_mask
    data["zero_mask"]          = zero_mask
 
    # --- Direction of displacement ---
    if "analogue_x" in data and "analogue_y" in data:
        # Present-cell coordinates on the result grid
        dft1_on_grid = dft1.to_xarray().interp_like(data, method="nearest")
        x0 = dft1_on_grid["x"] * 1000   # km -> m
        y0 = dft1_on_grid["y"] * 1000
 
        dx = data["analogue_x"] - x0
        dy = data["analogue_y"] - y0
 
        # Horizontal direction (0 = North, 90 = East, 180 = South, 270 = West),
        # only for cells that move horizontally and have an analogue
        mask_nonzero = ((dx != 0) | (dy != 0)) & (data["ab_dist_fw"] != 0) & (~missing_mask)
        xy_dir = xr.full_like(dx, np.nan).where(~mask_nonzero, (90 - np.degrees(np.arctan2(dy, dx))) % 360)
 
        results.update({"dx": dx, "dy": dy, "xy_dir": xy_dir})
        data["dx"], data["dy"], data["xy_dir"] = dx, dy, xy_dir
 
        # Vertical direction: +1 analogue deeper, -1 shallower, 0 same depth
        if "analogue_z" in data:
            z0    = dft1_on_grid["z"] * 1000
            dz    = data["analogue_z"] - z0
            z_dir = xr.where(dz.isnull(), np.nan, xr.where(dz > 0, 1, xr.where(dz < 0, -1, 0)))
            results.update({"dz": dz, "z_dir": z_dir})
            data["dz"], data["z_dir"] = dz, z_dir
 
    # --- Direction counts by quadrant ---
    results["quadrant_labels"] = quadrant_labels
    if "xy_dir" in data:
        dirs = data["xy_dir"].values.flatten()
        dirs = dirs[~np.isnan(dirs)]
 
        counts = []
        for start, end in quadrant_ranges:
            if start < end:
                counts.append(np.sum((dirs >= start) & (dirs < end)))
            else:   # quadrant across North (0°)
                counts.append(np.sum((dirs >= start) | (dirs < end)))
        results["quadrant_counts"] = counts
 
        print("\n=== Movement Direction Counts by Quadrant ===")
        for qlab, qcount in zip(quadrant_labels, counts):
            print(f"{qlab}: {qcount}")
        print(f"Total in all quadrants: {np.sum(counts)}")
    else:
        results["quadrant_counts"] = [np.nan] * len(quadrant_labels)
 
    return results, data
 
 
def compute_depth_range_mean(da, depth_min, depth_max):
    """Mean of a variable across the depth levels of a range."""
    if depth_max:
        depth_filtered = da.sel(depth=slice(depth_min, depth_max))
    else:
        depth_filtered = da.sel(depth=slice(depth_min, da["depth"].max().item()))
    return depth_filtered.mean(dim="depth", skipna=True)
 
 
def select_bottom_points(dft, geodf_bottom):
    """Rows of a (depth, lat, lon) table that are bottom cells, matching lat/lon on rounded values."""
    def _rounded(index):
        return pd.MultiIndex.from_tuples(
            [(d, round(la, round_decimals), round(lo, round_decimals)) for d, la, lo in index],
            names=index.names
        )
    return dft[_rounded(dft.index).isin(_rounded(geodf_bottom.index))]
 
 
def filter_by_depth(obj, depth_min, depth_max, level_name="depth"):
    """Rows of a table with a depth index level within a depth range (bounds included)."""
    lev = obj.index.get_level_values(level_name)
    if depth_max:
        return obj[(lev >= depth_min) & (lev <= depth_max)]
    return obj[lev >= depth_min]



## FIGURES ##
def _add_grid_and_north(ax):
    """Add lat/lon gridlines and a north arrow to a map."""
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                      linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}
    gl.xlocator     = mticker.MaxNLocator(nbins=5)
    gl.ylocator     = mticker.MaxNLocator(nbins=5)
 
    x, y, arrow_length = 0.08, 0.08, 0.06
    ax.annotate("N", xy=(x, y + arrow_length), xytext=(x, y),
                arrowprops=dict(facecolor="black", width=3, headwidth=10),
                ha="center", va="center", fontsize=14, fontweight="bold",
                xycoords=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
 
 
def plot_climate_map(name, da, med_gdf, plot_type="single", depths=None, legend=None, uom=None,
                     cmap="pink_r", vmin=None, vmax=None, normalize=False,
                     missing_coords_df=None, geodf_bottom=None,
                     shapefile=None, region_extent=None, figsize=(12, 7), save=None):
    """Map a variable for one depth layer ("single"), the mean of a depth range ("range") or the seabed ("bottom"); no-analogue cells in purple."""
    if plot_type not in ["single", "range", "bottom"]:
        raise ValueError("Invalid plot_type. Use 'single', 'range' or 'bottom'.")
 
    # Colour scale capped at vmax (99th percentile if not given)
    norm = None
    if normalize:
        vmax = vmax if vmax is not None else np.nanpercentile(da.values, 99)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
 
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": map_crs})
    points  = None   # no-analogue cells to mark
 
    if plot_type == "single":
        # Depth level closest to the requested depth
        if "depth" in da.dims:
            closest_idx = np.argmin(np.abs(da["depth"].values - depths))
            plot_slice  = da.isel(depth=closest_idx)
            layer_depth = da["depth"].values[closest_idx]
        else:
            plot_slice  = da
            layer_depth = depths
 
        if missing_coords_df is not None:
            points = (missing_coords_df[missing_coords_df["depth"] == layer_depth]
                      if "depth" in missing_coords_df.columns else missing_coords_df)
        title, title_size = name, 20
 
    elif plot_type == "range":
        depth_min, depth_max = depths
        plot_slice = compute_depth_range_mean(da, depth_min, depth_max)
 
        # Water columns where most levels with no-analogue cells are no-analogue:
        # blanked on the map and marked as no-analogue
        if missing_coords_df is not None:
            mc_range = missing_coords_df[(missing_coords_df["depth"] >= depth_min) &
                                         (missing_coords_df["depth"] <= depth_max)]
            counts = mc_range.groupby(["lat", "lon"])["depth"].count().reset_index(name="no_analogue_count")
            total_layers = mc_range["depth"].nunique()
            points = counts[counts["no_analogue_count"] > total_layers / 2]
 
            for _, row in points.iterrows():
                if row["lat"] in plot_slice["lat"].values and row["lon"] in plot_slice["lon"].values:
                    plot_slice.loc[dict(lat=row["lat"], lon=row["lon"])] = np.nan
        title, title_size = f"{name} (mean {depth_min}-{depth_max or 'max'}m)", None
 
    else:   # bottom: value of the deepest cell of each water column
        bottom_data = [(0, lat, lon, da.sel(depth=d, lat=lat, lon=lon).values.item())
                       for (d, lat, lon) in geodf_bottom.index
                       if not np.isnan(da.sel(depth=d, lat=lat, lon=lon).values)]
        df_bottom  = pd.DataFrame(bottom_data, columns=["depth", "lat", "lon", "value"])
        plot_slice = df_bottom.set_index(["depth", "lat", "lon"]).to_xarray()["value"]
 
        if missing_coords_df is not None and not missing_coords_df.empty:
            points = missing_coords_df
        title, title_size = name, None
 
    # Reproject to the map projection and draw
    plot_slice = (plot_slice.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                  .rio.write_crs("epsg:4326")
                  .rio.reproject(map_crs.proj4_init))
    im = plot_slice.plot(ax=ax, transform=plot_slice.rio.crs.to_epsg(), cmap=cmap, norm=norm, add_colorbar=False)
 
    if points is not None and not points.empty:
        gpd.GeoDataFrame(geometry=gpd.points_from_xy(points.lon, points.lat), crs="epsg:4326") \
           .to_crs(map_crs.proj4_init).plot(ax=ax, color="purple", markersize=10, alpha=0.6)
 
    med_gdf.boundary.plot(ax=ax, color="black")
    _add_grid_and_north(ax)
 
    cbar_ax = fig.add_axes([0.213, 0.08, 0.6, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal").set_label(
        f"{legend or name} ({uom})" if uom else name, fontsize=16)
    cbar_ax.tick_params(labelsize=14)
    ax.set_title(title, fontsize=title_size)
 
    # Optional extra outlines and zoom (extent as lon_min, lon_max, lat_min, lat_max)
    if shapefile is not None:
        shapefile.to_crs(map_crs.proj4_init).boundary.plot(ax=ax, color="black")
    if region_extent is not None:
        min_lon, max_lon, min_lat, max_lat = region_extent
        project = pyproj.Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)
        minx, miny = project.transform(min_lon, min_lat)
        maxx, maxy = project.transform(max_lon, max_lat)
        ax.set_extent([minx, maxx, miny, maxy], crs=map_crs)
 
    _finish(fig, save)
 
 
def movement_direction_rose(xy_dir, results, depth_label="", bins=36, save=None):
    """Rose plot of horizontal displacement directions, with quadrant and no-analogue proportions."""
    directions   = xy_dir.values.flatten() if isinstance(xy_dir, xr.DataArray) else xy_dir.flatten()
    directions   = directions[~np.isnan(directions)]
    total_points = results.get("total_points", np.nan)
 
    fig = plt.figure(figsize=(12, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.0], wspace=0.05)
 
    # Histogram of directions as proportion of all cells
    ax = fig.add_subplot(gs[0, 0], polar=True)
    counts, bin_edges = np.histogram(np.deg2rad(directions), bins=bins)
    proportions = counts / total_points if total_points > 0 else counts * np.nan
    ax.bar(bin_edges[:-1], proportions, width=np.diff(bin_edges), bottom=0.0, align="edge",
           color="steelblue", edgecolor="k", alpha=0.7)
 
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Movement Direction Rose ({depth_label})", fontsize=18, pad=20)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
 
    # Text box with proportions of all cells
    def fmt_prop(x):
        return f"{x:.2f}" if not np.isnan(x) else "n/a"
 
    quad_props = (np.array(results.get("quadrant_counts", [])) / total_points
                  if total_points > 0 else np.nan)
    text_lines = [f"{l}: {fmt_prop(p)}" for l, p in zip(results.get("quadrant_labels", []), quad_props)]
    text_lines += [
        "",
        f"No analogue: {fmt_prop(results.get('points_with_no_analogue', np.nan) / total_points)}",
        f"Null velocity: {fmt_prop(results.get('ab_dist_fw_null', np.nan) / total_points)}",
        f"Vertical only: {fmt_prop(results.get('count_vertical_only', np.nan) / total_points)}",
    ]
 
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    ax_text.text(0.0, 0.5, "\n".join(text_lines), va="center", ha="left", fontsize=20, linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=1.2", facecolor="white", edgecolor="0.3", alpha=0.95))
 
    fig.subplots_adjust(left=0.05, right=0.95)
    _finish(fig, save)
 
 
def depth_dissimilarity_heatmaps(data, depth_min, depth_max, save=None):
    """Pairwise dissimilarity between depth layers of a range, for horizontal and vertical velocity."""
    data_range = data.sel(depth=slice(depth_min, depth_max))
    depths     = data_range["depth"].values
    n_depths   = len(depths)
 
    def preprocess_layer(da, mask):
        """Annual values of a layer, with no-analogue cells set to 1.5 x the layer maximum."""
        arr  = da.values / n_years
        vmax = np.nanmax(arr)
        return np.where(mask.values, vmax * 1.5, arr).flatten()
 
    def compute_normalized_dissimilarity(layers, masks):
        """Normalised RMS difference between layers; a cell with no analogue in only one layer counts as maximum dissimilarity (1)."""
        n      = len(layers)
        matrix = np.zeros((n, n))
 
        # Normalisation by the overall range of values
        all_values = np.concatenate([v[~np.isnan(v)] for v in layers if np.any(~np.isnan(v))])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
        norm = vmax - vmin if vmax != vmin else 1
 
        for i in range(n):
            for j in range(i, n):
                vi, vj = layers[i], layers[j]
                mi, mj = masks[i], masks[j]
 
                valid_mask       = ~np.isnan(vi) & ~np.isnan(vj)
                missing_mismatch = (mi & ~mj & ~np.isnan(vj)) | (mj & ~mi & ~np.isnan(vi))
 
                if np.sum(valid_mask) == 0 and np.sum(missing_mismatch) == 0:
                    matrix[i, j] = matrix[j, i] = np.nan
                    continue
 
                d_valid = 0
                if np.sum(valid_mask) > 0:
                    d_valid = (np.linalg.norm((vi[valid_mask] - vj[valid_mask]) / norm)
                               / np.sqrt(np.sum(valid_mask)))
 
                # Weighted mean of value differences and missing-analogue mismatches
                matrix[i, j] = matrix[j, i] = (
                    (d_valid * np.sum(valid_mask) + 1.0 * np.sum(missing_mismatch))
                    / (np.sum(valid_mask) + np.sum(missing_mismatch))
                )
 
        return matrix
 
    mask_layers = [data_range["missing_mask"].isel(depth=i).values.flatten() for i in range(n_depths)]
    dissim = {}
    for var in ["delta_xy", "delta_z"]:
        if var in data_range:
            layers = [preprocess_layer(data_range[var].isel(depth=i), data_range["missing_mask"].isel(depth=i))
                      for i in range(n_depths)]
            dissim[var] = compute_normalized_dissimilarity(layers, mask_layers)
        else:
            dissim[var] = None
 
    # Heatmaps
    titles = {"delta_xy": "Horizontal velocity", "delta_z": "Vertical velocity"}
    cmap   = "RdYlBu_r"
    ncols  = 2 if dissim["delta_z"] is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 14))
    axes = np.atleast_1d(axes)
 
    for ax, var in zip(axes, ["delta_xy", "delta_z"]):
        if dissim[var] is None:
            ax.axis("off")
            continue
        sns.heatmap(dissim[var], ax=ax,
                    xticklabels=[int(d) for d in depths], yticklabels=[int(d) for d in depths],
                    cmap=cmap, square=True, vmin=0, vmax=1, cbar=False)
        ax.set_title(f"Dissimilarity between depths: {titles[var]}", fontsize=18)
        ax.set_xlabel("Depth (m)", fontsize=16)
        ax.set_ylabel("Depth (m)", fontsize=16)
        ax.tick_params(axis="x", labelsize=14, rotation=45)
        ax.tick_params(axis="y", labelsize=14, rotation=0)
 
    fig.subplots_adjust(bottom=0.18)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.06, pad=0.08)
    cbar.set_label("Dissimilarity", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
 
    _finish(fig, save)
    return dissim
 
 
def assign_depths_to_biozones(data_dict):
    """Map each depth level to the biozone file it is taken from, following profile_bounds."""
    depth_source = {}
    for label, ds in data_dict.items():
        dmin, dmax = profile_bounds[label]
        for depth in ds["depth"].values:
            if dmin <= depth <= dmax or (label == "bathyabyssopelagic" and depth <= dmax):
                depth_source[depth] = label
    return depth_source
 
 
def _layer_in_region(ds, depth, region_extent=None):
    """One depth level of a result dataset, optionally restricted to a region (lon_min, lon_max, lat_min, lat_max)."""
    if region_extent is None:
        return ds.sel(depth=depth)
    lon_min, lon_max, lat_min, lat_max = region_extent
    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max), depth=depth)
 
 
def plot_no_analogue_profile(data_dict, missing_dict, depth_source, region_extent=None,
                             save=os.path.join(data_dir, "prop_noanalogue_nullvel.png")):
    """Proportion of no-analogue and null-velocity cells at each depth level, optionally within a region."""
    all_depths = np.sort(list(depth_source.keys()))
    rows = []
 
    for depth in all_depths:
        label    = depth_source[depth]
        ds_layer = _layer_in_region(data_dict[label], depth, region_extent)
 
        missing = missing_dict[label]
        missing = missing[missing["depth"] == depth]
        if region_extent is not None:
            lon_min, lon_max, lat_min, lat_max = region_extent
            missing = missing[missing["lat"].between(lat_min, lat_max) & missing["lon"].between(lon_min, lon_max)]
 
        # Proportions of all grid cells of the layer
        total_points  = np.prod(ds_layer["delta_xy"].shape)
        n_null        = ((ds_layer["delta_xy"] == 0) & (ds_layer["delta_z"] == 0)).sum().item()
        n_no_analogue = len(missing)
 
        rows.append({"depth": depth,
                     "No Analogue": n_no_analogue / total_points if total_points > 0 else np.nan,
                     "Null Velocity": n_null / total_points if total_points > 0 else np.nan})
 
    df_plot = pd.DataFrame(rows)
 
    fig, ax = plt.subplots(figsize=(6, 10))
    width = 0.5
    y = np.arange(len(df_plot))
 
    ax.barh(y - width / 2, df_plot["No Analogue"], height=width, color="purple", label="No Analogue")
    ax.barh(y + width / 2, df_plot["Null Velocity"], height=width, color="grey", label="Null Velocity")
 
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["depth"])
    ax.invert_yaxis()
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Depth (m)")
    ax.legend()
    plt.tight_layout()
    _finish(fig, save)
 
 
def plot_vertical_shift_profile(data_dict, depth_source, region_extent=None, eps=1e-6,
                                save=os.path.join(data_dir, "total_vertical_shift.png")):
    """Mean vertical shift at each depth level, optionally within a region; colour = dominant direction, size = number of cells moving that way."""
    rows = []
    for depth in np.sort(list(depth_source.keys())):
        label = depth_source[depth]
        layer = _layer_in_region(data_dict[label], depth, region_extent)
 
        mean_dz = layer["delta_z"].mean(dim=["lat", "lon"], skipna=True).item()
        vel     = layer["delta_z"].values
 
        # Direction of the mean shift and number of cells shifting that way
        if mean_dz > eps:
            n, c = np.sum(vel > eps), "red"
        elif mean_dz < -eps:
            n, c = np.sum(vel < -eps), "blue"
        else:
            n, c = np.sum(np.abs(vel) <= eps), "gray"
 
        rows.append({"depth": depth, "mean": mean_dz, "n": n, "color": c, "group": label})
 
    stats = pd.DataFrame(rows).sort_values("depth")
    stats["size"] = np.clip((stats["n"] / stats["n"].max()) * 300, 10, 300)
 
    y_ord = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(14, 18))
 
    # Background band per biozone
    for g, bg in zip(profile_bounds, profile_bg_colors):
        dsub = stats[stats["group"] == g]
        if not dsub.empty:
            y_start = y_ord[stats.index.get_loc(dsub.index[0])]
            y_end   = y_ord[stats.index.get_loc(dsub.index[-1])]
            ax.axhspan(y_start, y_end, color=bg, alpha=0.4, zorder=0)
 
    ax.scatter(stats["mean"], y_ord, c=stats["color"], s=stats["size"],
               alpha=0.8, edgecolors="k", linewidth=0.5)
    ax.set_xscale("symlog", linthresh=3.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Vertical Shift (m)", fontsize=18)
    ax.set_ylabel("Depth (m)", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yticks(y_ord)
    ax.set_yticklabels([f"{d}" for d in stats["depth"].values])
    ax.tick_params(axis="both", labelsize=16)
    ax.axvline(0, color="k", linewidth=1.5, linestyle="-")
 
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="k",
               label=lab, markersize=12)
        for c, lab in [("red", "Upward"), ("blue", "Downward"), ("gray", "No movement")]
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=16)
 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    _finish(fig, save)


## MAIN ##
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)
 
    med_gdf    = gpd.read_file(med_shapefile).to_crs(map_crs.proj4_init)
    dft1_full  = pd.read_csv(os.path.join(data_dir, f"{prefix}_dft1.csv"), index_col=[0, 1, 2])
    geodf_full = gpd.read_file(package_file, layer="geodf").set_index(["depth", "lat", "lon"])
 
    # ---------------- Full water column ----------------
    data_full    = xr.open_dataset(velocity_file("full"))
    missing_full = pd.read_csv(no_analogue_file("full"))
    results_full, data_full = analyze_dataset(data_full, missing_full, dft1_full, mode="full")
 
    # ---------------- Surface layer ----------------
    label        = "surface"
    data_surf    = data_full.sel(depth=surface_depth)
    dft1_surf    = dft1_full.xs(surface_depth, level="depth")
    missing_surf = missing_full[missing_full["depth"] == surface_depth]
 
    results_surf, data_surf = analyze_dataset(data_surf, missing_surf, dft1_surf, mode="surface")
 
    movement_direction_rose(data_surf["xy_dir"], results_surf,
                            depth_label=f"{label} {surface_depth} m",
                            save=savepath(label, "dir_rose_plot"))
 
    plot_climate_map("Surface climate distance", data_surf["cc_dist"], med_gdf,
                     plot_type="single", depths=surface_depth, cmap="viridis",
                     save=savepath(label, "cc_distance"))
 
    for var, cfg in velocity_maps.items():
        plot_climate_map(cfg["title"], data_surf[var] / n_years, med_gdf,
                         plot_type="single", depths=surface_depth,
                         missing_coords_df=missing_surf,
                         cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"],
                         normalize=True, uom=cfg["uom"],
                         save=savepath(label, cfg["suffix"]))
 
    # ---------------- Seabed ----------------
    label = "bottom"
 
    # Deepest cell of each water column
    geodf_bottom = (geodf_full.reset_index()
                    .groupby(["lat", "lon"])["depth"].max()
                    .reset_index()
                    .set_index(["depth", "lat", "lon"]))
 
    data_bottom    = xr.open_dataset(velocity_file("bottom"))
    dft1_bottom    = select_bottom_points(dft1_full, geodf_bottom)
    missing_bottom = pd.read_csv(no_analogue_file("bottom"))
 
    results_bottom, data_bottom = analyze_dataset(data_bottom, missing_bottom, dft1_bottom, mode="bottom")
 
    movement_direction_rose(data_bottom["xy_dir"], results_bottom,
                            depth_label="Bottom layer",
                            save=savepath(label, "dir_rose_plot"))
 
    plot_climate_map("Climate velocity", data_bottom["ab_dist_fw"] / n_years, med_gdf,
                     plot_type="bottom", geodf_bottom=geodf_bottom,
                     missing_coords_df=missing_bottom,
                     cmap=climate_cmap, vmax=50, normalize=True, uom="km/year",
                     save=savepath(label, "vxy_map"))
 
    # ---------------- Biozones ----------------
    data_dict, missing_dict = {}, {}
 
    for label, (depth_min, depth_max) in depth_ranges.items():
        range_id = f"{depth_min}-{depth_max}"
        print(f"\n{'='*60}\nProcessing depth range: {label} ({range_id})\n{'='*60}\n")
 
        dft1    = filter_by_depth(dft1_full, depth_min, depth_max, level_name=dft1_full.index.names[0])
        data    = xr.open_dataset(velocity_file(range_id))
        missing = pd.read_csv(no_analogue_file(range_id))
 
        results, data_with_dir = analyze_dataset(data, missing, dft1, mode="range",
                                                 depth_range=(depth_min, depth_max))
 
        # Kept for the depth profiles below
        data_dict[label]    = data_with_dir
        missing_dict[label] = missing
 
        movement_direction_rose(data_with_dir["xy_dir"], results, depth_label=label,
                                save=savepath(label, "dir_rose_plot"))
 
        depth_dissimilarity_heatmaps(data_with_dir, depth_min, depth_max,
                                     save=savepath(label, "depth_dissimilarity"))
 
        for var, cfg in velocity_maps.items():
            plot_climate_map(cfg["title"], data_with_dir[var] / n_years, med_gdf,
                             plot_type="range", depths=(depth_min, depth_max),
                             legend=cfg["legend"], uom=cfg["uom"],
                             missing_coords_df=missing,
                             cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"],
                             normalize=True, save=savepath(label, cfg["suffix"]))
 
    # ---------------- Depth profiles ----------------
    depth_source = assign_depths_to_biozones(data_dict)
    plot_no_analogue_profile(data_dict, missing_dict, depth_source)
    plot_vertical_shift_profile(data_dict, depth_source)
 