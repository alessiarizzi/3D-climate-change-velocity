"""
Step 3 — Climate analogue and 3D climate velocity

Companion code to Rizzi et al. (2026)

What this script does
---------------------
1) Summarises the bioclimatic variables (step 2) with a PCA filtered on the present period, and projects both periods onto the same components. 
    Each cell is then described by its climate (3 principal components) and its position (x, y in km, EPSG:3035; depth z in km)
2) For each present cell, it searches the future cells with a similar climate (distance in PCA space below nn_threshold) and keeps the geographically closest one: the forward climate analogue.
    The distance to the analogue, split into horizontal and vertical components, is the displacement needed to track present climate conditions over the period. 
    Cells with no analogue within the threshold remain NaN. 
3) The search is run for the whole water column ("full"), for each depth range ("range", one per biozone) and for the seabed ("bottom", 2D search on the deepest cell of each water column)
4) Optionally (run_sensitivity), the depth-range runs are repeated for a set of thresholds, for the sensitivity analysis of step 4.

The search is done in chunks of present cells and saves progress after each chunk, so it can be stopped and restarted. 
Progress files depend on the inputs: delete progress_dir after changing the PCA, the periods, the depth ranges or chunk_size (!!)


Inputs (in data_dir)
------
- pre-pca_dft1.csv, pre-pca_dft2.csv : bioclimatic tables (step 2)
- {prefix}_t1_bio.nc                 : grid template for the outputs (step 2)
- 3d_package.gpkg, layer "geodf"     : cell positions in EPSG:3035 (step 2)


Outputs
-------
- {prefix}_dft1.csv, {prefix}_dft2.csv : tables with PCA components and coordinates
- {range}_dist_vel_{threshold}.nc; one file per run, with:
    ab_dist_fw       : distance to the analogue (km; 3D, or 2D for bottom)
    delta_xy         : horizontal distance to the analogue (km, over the period)
    delta_z          : vertical displacement, present - analogue depth (m, over the period; positive = analogue shallower)
    cc_dist          : local climate change - distance in PCA space between the present and future climate of the same cell
    analogue_x/_y/_z : analogue position (m, EPSG:3035; depth in m)
    
    delta_xy and delta_z are not written for the bottom run.
- {range}_no_analogue_{threshold}.csv             cells with no analogue  (columns depth, lat, lon)

{range} is "full", "bottom" or the depth range (e.g. "0-50").

"""



## IMPORTS
import os
 
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance



## CONFIGURATION ##
# Must match the prefix of steps 1 and 2
prefix = "3d_2006-2055" 

data_dir     = "data/outputs"
progress_dir = "data/outputs/progress_file"
package_file = os.path.join(data_dir, "3d_package.gpkg")


# Analogue search
# Maximum climate distance (in the standardised PCA space) for a future cell to count as an analogue of a present cell
nn_threshold = 0.1
 
# Number of present cells processed at a time (limits memory use).
chunk_size = 100
 
# Decimals used to match lat/lon when selecting bottom cells (float precision).
round_decimals = 1


# Runs
# Depth ranges (m, both bounds included), one per biozone. 
depth_ranges = [(0, 50), (30, 250), (200, 1250), (1000, 5500)]
 
run_full   = True   # whole water column
run_bottom = True   # seabed (2D search)
 

# Threshold sensitivity (used by step 4) 
# If True, the depth-range runs are repeated for each threshold below.
# Must match sensitivity_thresholds in step 4.
run_sensitivity        = False
sensitivity_thresholds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]


pc_columns = ["c0", "c1", "c2"]   # principal components kept



## FUNCTIONS ##
def run_pca(dft1, dft2, geodf_full):
    """Fit scaler and PCA on the present, project both periods, and add coordinates in km."""
    scaler = StandardScaler().fit(dft1)
    x1     = scaler.transform(dft1)
 
    pca = PCA(n_components=len(pc_columns))
    pca.fit(x1)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
 
    y1 = pca.transform(x1)
    y2 = pca.transform(scaler.transform(dft2))
 
    dft1[pc_columns] = y1
    dft2[pc_columns] = y2
 
    # Coordinates are assigned by position: geodf_full must list the cells in
    # the same order as the tables (both come from the same step-2 table)
    if len(geodf_full) != len(dft1):
        raise ValueError("geodf and bioclimatic tables have a different number of cells.")
 
    for df in (dft1, dft2):
        df["x"] = geodf_full.geometry.x.values / 1000                 # km
        df["y"] = geodf_full.geometry.y.values / 1000                 # km
        df["z"] = df.index.get_level_values("depth") / 1000           # km
 
    return dft1, dft2


def process_bioclimatic_data(dft1_full, dft2_full, geodf_full, mode="range",
                             depth_min=None, depth_max=None, threshold=nn_threshold):
    """Forward analogue search and 3D climate velocity for one run (full, depth range or bottom) and one threshold."""
    threshold_label = f"t{threshold}".replace(".", "")
 
    if mode == "full":
        range_label = "full"
    elif mode == "range":
        range_label = f"{depth_min}-{depth_max}"
    elif mode == "bottom":
        range_label = "bottom"
    else:
        raise ValueError("mode must be one of: 'full', 'range', 'bottom'")
 
    # --- Select the cells of this run ---
    if mode == "full":
        dft1  = dft1_full.copy()
        dft2  = dft2_full.copy()
        geodf = geodf_full.copy()
 
    elif mode == "range":
        depth1 = dft1_full.index.get_level_values(0)
        depth2 = dft2_full.index.get_level_values(0)
        depthg = geodf_full.index.get_level_values("depth")
 
        dft1  = dft1_full[(depth1 >= depth_min) & (depth1 <= depth_max)].copy()
        dft2  = dft2_full[(depth2 >= depth_min) & (depth2 <= depth_max)].copy()
        geodf = geodf_full[(depthg >= depth_min) & (depthg <= depth_max)].copy()
 
    elif mode == "bottom":
        # Deepest cell of each water column (lat, lon)
        deepest = (
            geodf_full.reset_index()
            .groupby(["lat", "lon"])["depth"].max()
            .reset_index()
            .set_index(["depth", "lat", "lon"])
        )
 
        # Match on rounded lat/lon to avoid float precision mismatches
        def _rounded(index):
            return pd.MultiIndex.from_tuples(
                [(d, round(la, round_decimals), round(lo, round_decimals)) for d, la, lo in index],
                names=index.names
            )
 
        bottom_idx = _rounded(deepest.index)
        dft1  = dft1_full[_rounded(dft1_full.index).isin(bottom_idx)].copy()
        dft2  = dft2_full[_rounded(dft2_full.index).isin(bottom_idx)].copy()
        geodf = geodf_full.loc[dft1.index].copy()
 
    # --- Progress files (allow restarting an interrupted run) ---
    progress_file_fw      = os.path.join(progress_dir, f"{range_label}_ab_dist_fw_{threshold_label}.npy")
    analogue_coords_file  = os.path.join(progress_dir, f"{range_label}_analogue_coords_{threshold_label}.npy")
    processed_chunks_file = os.path.join(progress_dir, f"{range_label}_processed_chunks_{threshold_label}.txt")
 
    ab_dist_fw = (np.load(progress_file_fw) if os.path.exists(progress_file_fw)
                  else np.full(len(dft1), np.nan))
    analogue_coords = (np.load(analogue_coords_file) if os.path.exists(analogue_coords_file)
                       else np.full((len(dft1), 3), np.nan))
 
    processed_chunks = []
    if os.path.exists(processed_chunks_file):
        with open(processed_chunks_file, "r") as f:
            processed_chunks = [int(line.strip()) for line in f]
 
    # Horizontal search for bottom, 3D search otherwise
    geo_columns = ["x", "y"] if mode == "bottom" else ["x", "y", "z"]
    xyz2        = dft2[["x", "y", "z"]].to_numpy()
 
    # --- Analogue search, one chunk of present cells at a time ---
    for start_idx in range(0, len(dft1), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dft1))
 
        if start_idx in processed_chunks:
            continue
 
        try:
            print(f"[{range_label}] Processing rows {start_idx} to {end_idx}...")
 
            # Climate distance (PCA space) and geographic distance (km)
            # between each present cell of the chunk and all future cells
            dmc_chunk = distance.cdist(dft1[pc_columns].iloc[start_idx:end_idx], dft2[pc_columns])
            dmg_chunk = distance.cdist(dft1[geo_columns].iloc[start_idx:end_idx], dft2[geo_columns])
 
            # Forward analogue: closest future cell among those with similar climate
            forward_min_dmg = np.where(dmc_chunk < threshold, dmg_chunk, np.nan)
            ab_dist_fw[start_idx:end_idx] = np.nanmin(forward_min_dmg, axis=1)
 
            # Analogue position (first closest cell in case of ties); NaN if no analogue
            has_analogue = ~np.isnan(ab_dist_fw[start_idx:end_idx])
            rows         = np.where(has_analogue)[0]
            nearest      = np.nanargmin(forward_min_dmg[rows], axis=1)
 
            analogue_coords[start_idx + rows] = xyz2[nearest]
            analogue_coords[start_idx + np.where(~has_analogue)[0]] = np.nan
 
            # Save progress
            with open(processed_chunks_file, "a") as f:
                f.write(f"{start_idx}\n")
            np.save(progress_file_fw, ab_dist_fw)
            np.save(analogue_coords_file, analogue_coords)
 
        except Exception as e:
            print(f"An error occurred while processing rows {start_idx} to {end_idx}: {e}")
 
    # --- Local climate change: distance in PCA space between the present and
    #     future climate of the same cell (rows of dft1 and dft2 are the same cells) ---
    cc_dist = np.linalg.norm(dft1[pc_columns].to_numpy() - dft2[pc_columns].to_numpy(), axis=1)
 
    # --- Cells with no analogue within the threshold ---
    no_analogue = dft1.index[np.isnan(ab_dist_fw)].to_frame(index=False)
    no_analogue.columns = ["depth", "lat", "lon"]
    no_analogue.to_csv(os.path.join(data_dir, f"{range_label}_no_analogue_{threshold_label}.csv"), index=False)
 
    # --- Write results on the model grid ---
    # The step-2 bioclimatic file is used as grid template
    bio = xr.open_dataset(os.path.join(data_dir, f"{prefix}_t1_bio.nc"))
    if mode == "range":
        bio = bio.sel(depth=slice(depth_min, depth_max)).copy()
 
    def set_var(name, val):
        geodf[name] = val
        bio[name] = geodf[name].to_xarray()
 
    set_var("ab_dist_fw", ab_dist_fw)
    set_var("cc_dist", cc_dist)
    set_var("analogue_x", analogue_coords[:, 0] * 1000)   # m
    set_var("analogue_y", analogue_coords[:, 1] * 1000)   # m
    set_var("analogue_z", analogue_coords[:, 2] * 1000)   # m
 
    out_vars = ["ab_dist_fw", "cc_dist", "analogue_x", "analogue_y", "analogue_z"]
 
    # Horizontal and vertical displacement (3D runs only)
    if mode in ["full", "range"]:
        delta_xy = np.sqrt(
            (dft1["x"].values - analogue_coords[:, 0]) ** 2 +
            (dft1["y"].values - analogue_coords[:, 1]) ** 2
        )
        delta_z = dft1["z"].values - analogue_coords[:, 2]
 
        set_var("delta_xy", delta_xy)      # km
        set_var("delta_z", delta_z * 1000)   # m
 
        out_vars = ["ab_dist_fw", "cc_dist", "delta_z", "delta_xy", "analogue_x", "analogue_y", "analogue_z"]
 
    bio[out_vars].to_netcdf(os.path.join(data_dir, f"{range_label}_dist_vel_{threshold_label}.nc"))
    


## MAIN ## 
if __name__ == "__main__":
    os.makedirs(progress_dir, exist_ok=True)
 
    geodf_full = gpd.read_file(package_file, layer="geodf").set_index(["depth", "lat", "lon"])
 
    dft1 = pd.read_csv(os.path.join(data_dir, "pre-pca_dft1.csv"), index_col=[0, 1, 2])
    dft2 = pd.read_csv(os.path.join(data_dir, "pre-pca_dft2.csv"), index_col=[0, 1, 2])
 
    if not dft1.index.equals(dft2.index):
        raise ValueError("Present and future tables cover different cells (dft1 and dft2 indexes differ).")
 
    # 1. PCA and coordinates
    dft1, dft2 = run_pca(dft1, dft2, geodf_full)
    dft1.to_csv(os.path.join(data_dir, f"{prefix}_dft1.csv"))
    dft2.to_csv(os.path.join(data_dir, f"{prefix}_dft2.csv"))
 
    # Read back the saved tables once, so every run uses exactly the saved values
    dft1_pca = pd.read_csv(os.path.join(data_dir, f"{prefix}_dft1.csv"), index_col=[0, 1, 2])
    dft2_pca = pd.read_csv(os.path.join(data_dir, f"{prefix}_dft2.csv"), index_col=[0, 1, 2])
 
    # 2. Analogue search and velocity
    if run_full:
        process_bioclimatic_data(dft1_pca, dft2_pca, geodf_full, mode="full")
 
    for depth_min, depth_max in depth_ranges:
        print(f"Running depth range {depth_min}-{depth_max} m...")
        process_bioclimatic_data(dft1_pca, dft2_pca, geodf_full, mode="range",
                                 depth_min=depth_min, depth_max=depth_max)
 
    if run_bottom:
        process_bioclimatic_data(dft1_pca, dft2_pca, geodf_full, mode="bottom")
 
    # 3. Optional: depth-range runs for each sensitivity threshold (step 4)
    if run_sensitivity:
        for threshold in sensitivity_thresholds:
            if threshold == nn_threshold:
                continue   # already computed above
            for depth_min, depth_max in depth_ranges:
                print(f"Sensitivity run: threshold {threshold}, depth range {depth_min}-{depth_max} m...")
                process_bioclimatic_data(dft1_pca, dft2_pca, geodf_full, mode="range",
                                         depth_min=depth_min, depth_max=depth_max,
                                         threshold=threshold)