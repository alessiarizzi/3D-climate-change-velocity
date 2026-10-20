"""
Step 6 — Analysis of 3D climate velocity by Mediterranean basin

Companion code to Rizzi et al. (2026)

What this script does
---------------------
Repeats the step-5 analysis separately for each Mediterranean basin (Provencal, Alboran-Algerian, Tyrrhenian, Sicily Strait, Adriatic, Ionian, Aegean, Levantine):
    - statistics, displacement types, analogue availability and directions, for the whole water column and for each biozone (saved as tables);
    - direction rose plots and maps of horizontal and vertical velocity, for each biozone;
    - depth profiles of no-analogue / null-velocity cells and of the mean vertical shift.

Each basin is the bounding box of its sub-basins (from basins_file).

The analysis functions and settings (threshold, n_years, depth ranges, colour scales, file names) are taken from step 5, so both steps always use the same methods. Keep 05_velocity_analysis.py in the same folder.


Inputs
------
- all inputs of step 5
- data/sub-basins_med.csv   sub-basin codes and bounding boxes (columns Code, lat_min, lat_max, lon_min, lon_max)


Outputs
-------
- basin_stats_full.csv, basin_stats_biozones.csv : statistics per basin
- PNG figures per basin

"""


## IMPORTS 
import os
import importlib
 
import numpy as np
import pandas as pd
import xarray as xr

# Step-5 functions and settings
va = importlib.import_module("05_velocity_analysis")



## CONFIGURATION ##
basins_file = "data/sub-basins_med.csv"
basin_dir   = os.path.join(va.data_dir, "basins")
 
# Basin names; sub-basin codes are grouped by their first two letters,
# except the DJ codes, split into Adriatic (DJ1-DJ3) and Ionian (other DJ)
basin_groups = {
    "DF":  "Provencal",
    "DS":  "Alboran-Algerian",
    "DT":  "Tyrrhenian",
    "DI":  "Sicily Strait",
    "DJa": "Adriatic",
    "DJb": "Ionian",
    "DH":  "Aegean",
    "DL":  "Levantine",
}
adriatic_codes = ["DJ1", "DJ2", "DJ3"]



## FUNCTIONS ##
def assign_basin(code):
    """Basin code of a sub-basin code."""
    if code.startswith("DJ"):
        return "DJa" if code in adriatic_codes else "DJb"
    return code[:2]
 
 
def basin_extents(regions_df):
    """Name and bounding box (lon_min, lon_max, lat_min, lat_max) of each basin."""
    extents = {}
    for basin_code, group in regions_df.groupby("basin_code"):
        extents[basin_code] = {
            "name": group["basin_name"].iloc[0],
            "region_extent": (group["lon_min"].min(), group["lon_max"].max(),
                              group["lat_min"].min(), group["lat_max"].max()),
        }
    return extents
 
 
def create_region_subsets(extents, data, missing_coords, dft1):
    """Subset the result dataset, no-analogue cells and present table to each basin's bounding box."""
    subsets = {}
    for basin_code, basin in extents.items():
        lon_min, lon_max, lat_min, lat_max = basin["region_extent"]
 
        dft1_reset = dft1.reset_index()
        sub_dft1   = dft1_reset[dft1_reset["lat"].between(lat_min, lat_max) &
                                dft1_reset["lon"].between(lon_min, lon_max)]
        index_cols = ["depth", "lat", "lon"] if "depth" in dft1_reset.columns else ["lat", "lon"]
 
        subsets[basin_code] = {
            **basin,
            "xarray":  data.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)),
            "missing": missing_coords[missing_coords["lat"].between(lat_min, lat_max) &
                                      missing_coords["lon"].between(lon_min, lon_max)],
            "dft1":    sub_dft1.set_index(index_cols),
        }
    return subsets
 
 
def scalar_results(results, **labels):
    """Keep the single-value results of analyze_dataset (plus quadrant counts) as one table row."""
    row = {**labels}
    row.update({k: v for k, v in results.items() if np.isscalar(v)})
    for qlab, qcount in zip(results["quadrant_labels"], results["quadrant_counts"]):
        row[f"quadrant_{qlab}"] = qcount
    return row



## MAIN ##
if __name__ == "__main__":
    os.makedirs(basin_dir, exist_ok=True)
 
    # Basins
    regions_df = pd.read_csv(basins_file)
    regions_df["basin_code"] = regions_df["Code"].apply(assign_basin)
    regions_df["basin_name"] = regions_df["basin_code"].map(basin_groups)
    extents = basin_extents(regions_df)
 
    med_gdf   = va.gpd.read_file(va.med_shapefile).to_crs(va.map_crs.proj4_init)
    dft1_full = pd.read_csv(os.path.join(va.data_dir, f"{va.prefix}_dft1.csv"), index_col=[0, 1, 2])
 
    # ---------------- Full water column ----------------
    data_full    = xr.open_dataset(va.velocity_file("full"))
    missing_full = pd.read_csv(va.no_analogue_file("full"))
 
    full_rows = []
    for basin_code, subset in create_region_subsets(extents, data_full, missing_full, dft1_full).items():
        print(f"\n=== Basin {basin_code}: {subset['name']} (FULL) ===")
        results, _ = va.analyze_dataset(subset["xarray"], subset["missing"], subset["dft1"], mode="full")
        full_rows.append(scalar_results(results, basin_code=basin_code, basin_name=subset["name"]))
 
    pd.DataFrame(full_rows).to_csv(os.path.join(basin_dir, "basin_stats_full.csv"), index=False)
 
    # ---------------- Biozones ----------------
    range_rows = []
    data_dict, missing_dict = {}, {}
 
    for label, (depth_min, depth_max) in va.depth_ranges.items():
        range_id    = f"{depth_min}-{depth_max}"
        depth_label = label.capitalize()
        print(f"\n{'='*60}\nProcessing depth range: {depth_label}\n{'='*60}\n")
 
        dft1    = va.filter_by_depth(dft1_full, depth_min, depth_max, level_name=dft1_full.index.names[0])
        data    = xr.open_dataset(va.velocity_file(range_id))
        missing = pd.read_csv(va.no_analogue_file(range_id))
 
        # Kept for the depth profiles below
        data_dict[label]    = data
        missing_dict[label] = missing
 
        for basin_code, subset in create_region_subsets(extents, data, missing, dft1).items():
            name = subset["name"]
            print(f"\n=== Basin {basin_code}: {name} ({depth_label}) ===")
 
            results, annotated = va.analyze_dataset(subset["xarray"], subset["missing"], subset["dft1"],
                                                    mode="range", depth_range=(depth_min, depth_max))
            range_rows.append(scalar_results(results, depth_label=depth_label,
                                             basin_code=basin_code, basin_name=name))
 
            if "xy_dir" in annotated:
                va.movement_direction_rose(annotated["xy_dir"], results, depth_label=depth_label,
                                           save=os.path.join(basin_dir, f"{name}_{range_id}_direction_rose.png"))
            else:
                print("xy_dir missing for this subset — skipping rose plot.")
 
            # Velocity maps zoomed on the basin
            va.plot_climate_map(f"Horizontal velocity ({depth_label})", annotated["delta_xy"] / va.n_years, med_gdf,
                                plot_type="range", depths=(depth_min, depth_max),
                                legend="Velocity XY", uom="km/year",
                                missing_coords_df=subset["missing"],
                                cmap=va.climate_cmap, normalize=True, vmax=50,
                                region_extent=subset["region_extent"],
                                save=os.path.join(basin_dir, f"{name}_{range_id}_vxy_map.png"))
 
            va.plot_climate_map(f"Vertical velocity ({depth_label})", annotated["delta_z"] / va.n_years, med_gdf,
                                plot_type="range", depths=(depth_min, depth_max),
                                legend="Velocity Z", uom="m/year",
                                missing_coords_df=subset["missing"],
                                cmap="coolwarm",
                                region_extent=subset["region_extent"],
                                save=os.path.join(basin_dir, f"{name}_{range_id}_vz_map.png"))
 
    pd.DataFrame(range_rows).to_csv(os.path.join(basin_dir, "basin_stats_biozones.csv"), index=False)
 
    # ---------------- Depth profiles per basin ----------------
    # Depth levels assigned to biozones as in step 5 (profile_bounds)
    depth_source = va.assign_depths_to_biozones(data_dict)
 
    for basin_code, basin in extents.items():
        name = basin["name"]
        print(f"\n=== Basin {basin_code}: {name} (depth profiles) ===")
 
        va.plot_no_analogue_profile(data_dict, missing_dict, depth_source,
                                    region_extent=basin["region_extent"],
                                    save=os.path.join(basin_dir, f"{name}_prop_noanalogue_nullvel.png"))
        va.plot_vertical_shift_profile(data_dict, depth_source,
                                       region_extent=basin["region_extent"],
                                       save=os.path.join(basin_dir, f"{name}_vertical_shift.png"))