"""
Part 3 — PCA and analogue-based climate velocity computation.

What this script does
---------------------
1) Loads pre-PCA bioclimatic tables for present (dft1) and future (dft2).
2) Fits a StandardScaler + PCA (3 components) on present conditions and applies
   the same transformation to future conditions.
3) Adds PCA scores (c0–c2) and geographic coordinates (x, y, z; km) to both
   tables and saves them.

4) Computes analogue-based climate velocity using a climate-space threshold,
   with three possible modes:

   (A) Full water column ("full")
       - uses all grid cells
       - climate-space distance: cc_dist (diagonal present↔future)
       - geographic distance in 3D (x,y,z): ab_dist_fw
       - analogue coordinates (x,y,z)
       - horizontal displacement (vel_xy) and vertical shift (vel_z)

   (B) Depth ranges ("range")
       - repeats the same 3D computation within predefined depth intervals
       - outputs one NetCDF per interval

   (C) Bottom points ("bottom")
       - restricts computation to the deepest grid cell at each (lat, lon)
       - geographic distance in 2D (x,y): ab_dist_fw (and optionally ab_dist_bw)
       - analogue coordinates (x,y,z)
       - (vel_xy/vel_z are not required for bottom outputs unless explicitly computed)

5) Writes NetCDF outputs for the selected mode(s), with intermediate progress
   saved to disk to allow restart after interruption.

Inputs
------
- data/outputs/pre-pca_dft1.csv
- data/outputs/pre-pca_dft2.csv
- data/outputs/<prefix>_t1_bio.nc
- data/outputs/3d_package.gpkg (layer='geodf', indexed by depth/lat/lon)
- Required Python packages: numpy, pandas, sklearn, scipy, geopandas, xarray

Outputs
-------
- PCA tables:
  - data/outputs/<prefix>_dft1.csv
  - data/outputs/<prefix>_dft2.csv

- Velocity/analogue outputs (depending on mode):
  - Full:    data/outputs/full_dist_vel_<threshold_label>.nc
  - Ranges:  data/outputs/<depth_min>-<depth_max>_dist_vel_<threshold_label>.nc
  - Bottom:  data/outputs/bottom_dist_vel_<threshold_label>.nc

- Progress files (per mode/range):
  - data/outputs/progress_file/<label>_*.npy
  - data/outputs/progress_file/<label>_processed_chunks_<threshold_label>.txt
"""



## Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
import geopandas as gpd
import xarray as xr


# ---- Settings ----
prefix = "3d_2006-2055"  # must match Part 1 and Part 2

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("data/outputs/progress_file", exist_ok=True)


# ---- Load GeoDataFrame (needed for x/y coordinates and later filtering) ----
geodf_full = (
    gpd.read_file("data/outputs/3d_package.gpkg", layer="geodf")
    .set_index(["depth", "lat", "lon"])
)

# ---- Load pre-PCA tables ----
dft1 = pd.read_csv("data/outputs/pre-pca_dft1.csv", index_col=[0, 1, 2])
dft2 = pd.read_csv("data/outputs/pre-pca_dft2.csv", index_col=[0, 1, 2])

# Safety check
if not dft1.index.equals(dft2.index):
    raise Exception("dft1 and dft2 indexes are different")


# ---- PCA on bioclimatic variables ----
# Fit scaler on present (dft1), then apply to both periods
scaler = StandardScaler().fit(dft1)
x1 = scaler.transform(dft1)

pca = PCA(n_components=3)
pca.fit(x1)
print(pca.explained_variance_ratio_)

# Transform present and future using the same PCA model
y1 = pca.transform(x1)
y2 = pca.transform(scaler.transform(dft2))

# Add PCA components
dft1[["c0", "c1", "c2"]] = y1
dft2[["c0", "c1", "c2"]] = y2

# Add geographic coordinates (km) and depth (km)
# NOTE: This assumes dft1/dft2 are ordered consistently with geodf_full index.
dft1["x"] = geodf_full.geometry.x.values / 1000
dft1["y"] = geodf_full.geometry.y.values / 1000
dft1["z"] = dft1.index.get_level_values("depth") / 1000

dft2["x"] = geodf_full.geometry.x.values / 1000 
dft2["y"] = geodf_full.geometry.y.values / 1000
dft2["z"] = dft2.index.get_level_values("depth") / 1000

# Save PCA tables
dft1.to_csv(f"data/outputs/{prefix}_dft1.csv")
dft2.to_csv(f"data/outputs/{prefix}_dft2.csv")


# ---- Function to calculate 3D climate velocity ----
def process_bioclimatic_data(mode="range",
                             depth_min=None, depth_max=None,
                             nn_threshold=0.1, chunk_size=100,
                             geodf_full=None, round_decimals=1):

    # Label / filenames
    threshold_label = f"t{nn_threshold}".replace(".", "")

    if mode == "full":
        range_label = "full"
    elif mode == "range":
        range_label = f"{depth_min}-{depth_max}"
    elif mode == "bottom":
        range_label = "bottom"
    else:
        raise ValueError("mode must be one of: 'full', 'range', 'bottom'")

    # Ensure progress directory exists
    os.makedirs("data/outputs/progress_file", exist_ok=True)

    # Load PCA data
    dft1_full = pd.read_csv(f"data/outputs/{prefix}_dft1.csv", index_col=[0, 1, 2])
    dft2_full = pd.read_csv(f"data/outputs/{prefix}_dft2.csv", index_col=[0, 1, 2])

    # Select points by mode
    if mode == "full":
        dft1 = dft1_full.copy()
        dft2 = dft2_full.copy()
        geodf = geodf_full.copy()
        
    elif mode == "range":
        dft1 = dft1_full[
            (dft1_full.index.get_level_values(0) >= depth_min) &
            (dft1_full.index.get_level_values(0) <= depth_max)
        ].copy()

        dft2 = dft2_full[
            (dft2_full.index.get_level_values(0) >= depth_min) &
            (dft2_full.index.get_level_values(0) <= depth_max)
        ].copy()

        geodf = geodf_full[
            (geodf_full.index.get_level_values("depth") >= depth_min) &
            (geodf_full.index.get_level_values("depth") <= depth_max)
        ].copy()

    elif mode == "bottom":
        # Deepest depth per (lat, lon) from geodf_full
        geodf_reset = geodf_full.reset_index()
        deepest_points = geodf_reset.groupby(["lat", "lon"])["depth"].max().reset_index()
        deepest_points = deepest_points.set_index(["depth", "lat", "lon"])

        # Match dft1/dft2 indices to deepest_points using rounded lat/lon (float precision issues)
        dft1_idx_round = pd.MultiIndex.from_tuples(
            [(d, round(la, round_decimals), round(lo, round_decimals)) for d, la, lo in dft1_full.index],
            names=dft1_full.index.names
        )
        
        dft2_idx_round = pd.MultiIndex.from_tuples(
            [(d, round(la, round_decimals), round(lo, round_decimals)) for d, la, lo in dft2_full.index],
            names=dft2_full.index.names
        )
        
        bottom_idx_round = pd.MultiIndex.from_tuples(
            [(d, round(la, round_decimals), round(lo, round_decimals)) for d, la, lo in deepest_points.index],
            names=deepest_points.index.names
        )

        # Use the subset of bottom points
        dft1 = dft1_full[dft1_idx_round.isin(bottom_idx_round)].copy()
        dft2 = dft2_full[dft2_idx_round.isin(bottom_idx_round)].copy()

        # Use a DataFrame with the same index ordering as dft1 for writing back to xarray
        geodf = geodf_full.loc[dft1.index].copy()        

    # Progress file paths
    progress_file_fw = f"data/outputs/progress_file/{range_label}_ab_dist_fw_{threshold_label}.npy"
    progress_file_dist = f"data/outputs/progress_file/{range_label}_cc_dist_{threshold_label}.npy"
    processed_chunks_file = f"data/outputs/progress_file/{range_label}_processed_chunks_{threshold_label}.txt"
    analogue_coords_file = f"data/outputs/progress_file/{range_label}_analogue_coords_{threshold_label}.npy"

    # Load progress if files exist, otherwise initialize arrays
    ab_dist_fw = np.load(progress_file_fw) if os.path.exists(progress_file_fw) else np.full(len(dft1), np.nan)
    cc_dist = np.load(progress_file_dist) if os.path.exists(progress_file_dist) else np.full(len(dft1), np.nan)
    analogue_coords = np.load(analogue_coords_file) if os.path.exists(analogue_coords_file) else np.full((len(dft1), 3), np.nan)

    processed_chunks = []
    if os.path.exists(processed_chunks_file):
        with open(processed_chunks_file, "r") as f:
            processed_chunks = [int(line.strip()) for line in f]

    # Chunk processing 
    for start_idx in range(0, len(dft1), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dft1))

        if start_idx in processed_chunks:
            continue

        try:
            print(f"[{range_label}] Processing rows {start_idx} to {end_idx}...")

            dmc_chunk = distance.cdist(
                dft1[["c0", "c1", "c2"]].iloc[start_idx:end_idx],
                dft2[["c0", "c1", "c2"]]
            )

            # geographic distances: 3D for full/range, 2D for bottom
            if mode == "bottom":
                dmg_chunk = distance.cdist(
                    dft1[["x", "y"]].iloc[start_idx:end_idx],
                    dft2[["x", "y"]]
                )
            else:
                dmg_chunk = distance.cdist(
                    dft1[["x", "y", "z"]].iloc[start_idx:end_idx],
                    dft2[["x", "y", "z"]]
                )

            # climate distance: diagonal present↔future (same grid cell order)
            cc_dist[start_idx:end_idx] = np.diagonal(dmc_chunk)

            # forward analogue distance (within climate threshold)
            mask = dmc_chunk < nn_threshold
            forward_min_dmg = np.where(mask, dmg_chunk, np.nan)
            ab_dist_fw[start_idx:end_idx] = np.nanmin(forward_min_dmg, axis=1)

            # analogue coordinates (x,y,z)
            for i in range(start_idx, end_idx):
                if np.isnan(ab_dist_fw[i]):
                    analogue_coords[i] = [np.nan, np.nan, np.nan]
                else:
                    valid_idx_fw = np.where(~np.isnan(forward_min_dmg[i - start_idx]))[0]
                    if len(valid_idx_fw) > 0:
                        min_idx_fw = valid_idx_fw[np.nanargmin(forward_min_dmg[i - start_idx][valid_idx_fw])]
                        analogue_coords[i] = dft2[["x", "y", "z"]].iloc[min_idx_fw].values

            # save progress
            with open(processed_chunks_file, "a") as f:
                f.write(f"{start_idx}\n")

            np.save(progress_file_fw, ab_dist_fw)
            np.save(analogue_coords_file, analogue_coords)

        except Exception as e:
            print(f"An error occurred while processing rows {start_idx} to {end_idx}: {e}")

    # Write results to NetCDF
    bio = xr.open_dataset(f"data/outputs/{prefix}_t1_bio.nc")  # It can be bio_t1 or bio_t2
    
    # For range, slice bio to reduce file size. For full/bottom keep full grid.
    if mode == "range":
        bio = bio.sel(depth=slice(depth_min, depth_max)).copy()

    # Helper to add a new variable to the GeoDataFrame and sync with NetCDF
    def set_var(name, val):
        geodf[name] = val
        bio[name] = geodf[name].to_xarray()

    # always saved: forward velocity + analogue coords
    set_var("ab_dist_fw", ab_dist_fw)
    set_var("analogue_x", analogue_coords[:, 0] * 1000)  # m
    set_var("analogue_y", analogue_coords[:, 1] * 1000)
    set_var("analogue_z", analogue_coords[:, 2] * 1000)
    
    # velocities only for full/range (3D)
    if mode in ["full", "range"]:
        vel_xy = np.sqrt(
            (dft1["x"].values - analogue_coords[:, 0])**2 +
            (dft1["y"].values - analogue_coords[:, 1])**2
        )
        vel_z = dft1["z"].values - analogue_coords[:, 2]

        set_var("vel_xy", vel_xy, geodf)          # km
        set_var("vel_z", vel_z * 1000, geodf)     # m

        out_vars = ["ab_dist_fw", "vel_z", "vel_xy",
                    "analogue_x", "analogue_y", "analogue_z"]
    else:
        out_vars = ["ab_dist_fw", "analogue_x", "analogue_y", "analogue_z"]

    
    bio[out_vars].to_netcdf(f"data/outputs/{range_label}_dist_vel_{threshold_label}.nc")


# ---- Run settings ----
threshold = 0.1
chunk_size = 100
round_decimals = 1

# Full water column
process_bioclimatic_data(
    mode="full",
    nn_threshold=threshold,
    chunk_size=chunk_size,
    geodf_full=geodf_full
)

# Depth ranges
depth_ranges = [(0, 50), (30, 250), (200, 1250), (1000, 5500)]

for (depth_min, depth_max) in depth_ranges:
    print(f"Running depth range {depth_min}-{depth_max} m...")
    process_bioclimatic_data(
        mode="range",
        depth_min=depth_min,
        depth_max=depth_max,
        nn_threshold=threshold,
        chunk_size=chunk_size,
        geodf_full=geodf_full
    )

# Bottom
process_bioclimatic_data(
    mode="bottom",
    nn_threshold=threshold,
    chunk_size=chunk_size,
    geodf_full=geodf_full
)
