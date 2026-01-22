"""
Part 4 — PCA interpretation and sensitivity analysis of thresholds.

What this script does
---------------------
(a) PCA interpretation
1) Loads pre-PCA bioclimatic tables for present (dft1) and future (dft2).
2) Runs PCA on the combined standardized dataset (9 biovariables) to interpret
   component structure and variance explained.
3) Produces:
   - cumulative explained variance + scree plot
   - correlation heatmaps between standardized variables and PC scores (PC1–PC3)
     by depth biozones.

(b) Sensitivity analysis of thresholds
4) For each depth biozone and for a range of analogue thresholds, reads the
   previously computed NetCDF outputs (Part 3) and computes:
   - median (and IQR) of forward analogue distance (ab_dist_fw)
   - proportion of no-analogue climates
5) Produces sensitivity plots (median distance + no-analogue proportion).

Inputs
------
- data/outputs/pre-pca_dft1.csv
- data/outputs/pre-pca_dft2.csv
- data/outputs/3d_package.gpkg (layer='geodf')
- data/outputs/<depth_min>-<depth_max>_dist_vel_<threshold_label>.nc (from Part 3)

Outputs
-------
- data/outputs/explain_var_scree_plots.png
- data/outputs/pca_corr_loadings.png
- data/outputs/<depth_min>-<depth_max>_sensitivity_analysis.png (one per depth range)
"""


## Import necessary libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Ensure output directory exists
os.makedirs("data/outputs", exist_ok=True)


## --- (a) PCA interpretation --- ##

# Load the bioclimatic tables (pre-PCA)
dft1 = pd.read_csv('data/outputs/pre-pca_dft1.csv', index_col=[0, 1, 2])
dft2 = pd.read_csv('data/outputs/pre-pca_dft2.csv', index_col=[0, 1, 2])

# Combine the two datasets (stack present + future)
df_combined = pd.concat([dft1, dft2], axis=0)

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_combined)

# Perform PCA on the 9 biovars
pca_interp = PCA(n_components=9)
pca_interp.fit(x_scaled)

# Variance explained by each component
explained_variance = pca_interp.explained_variance_ratio_
print("Explained Variance Ratio per Component:", explained_variance)

# Plot cumulative explained variance + scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(range(1, 10), np.cumsum(explained_variance), marker='o', linestyle='-')
axes[0].set_xlabel("Number of Principal Components", fontsize=16)
axes[0].set_ylabel("Cumulative Explained Variance", fontsize=16)
axes[0].set_title("Explained variance ratio per component", fontsize=18)
axes[0].axhline(y=0.95, color='g', linestyle='--', label="95% Variance Explained")
axes[0].axhline(y=0.99, color='r', linestyle='--', label="99% Variance Explained")
axes[0].legend(fontsize=16)
axes[0].tick_params(axis='both', labelsize=16)
axes[0].grid()

axes[1].plot(range(1, 10), explained_variance, marker='o', linestyle='--')
axes[1].set_xlabel("Principal Component", fontsize=16)
axes[1].set_ylabel("Explained Variance Ratio", fontsize=16)
axes[1].set_title("Scree Plot", fontsize=18)
axes[1].tick_params(axis='both', labelsize=16)
axes[1].grid()

plt.tight_layout()
plt.savefig('data/outputs/explain_var_scree_plots.png', dpi=300)
plt.show()
plt.close()


# PCA loadings (contribution of each variable to each component)
loadings = pd.DataFrame(
    pca_interp.components_,
    columns=df_combined.columns,
    index=[f"PC{i+1}" for i in range(9)]
)

# Add depth index as a column for filtering (depth is first index level)
df_combined['depth'] = [idx[0] for idx in df_combined.index]

# Transform the entire dataset using fitted PCA
X_pca = pca_interp.transform(x_scaled)
pc_df = pd.DataFrame(X_pca, index=df_combined.index, columns=[f'PC{i+1}' for i in range(9)])

# Recreate standardized data as a DataFrame (for correlations)
x_scaled_df = pd.DataFrame(
    x_scaled,
    index=df_combined.index,
    columns=df_combined.drop('depth', axis=1).columns
)

# Depth groups (matching your biozones)
depth_groups = {
    '0-40m': lambda d: d <= 50,
    '40-200m': lambda d: (d >= 30) & (d <= 250),
    '200-1000m': lambda d: (d >= 200) & (d <= 1250),
    '>1000m': lambda d: d > 1000
}

biozone_titles = {
    '0-40m': 'Euphotic zone',
    '40-200m': 'Mesophotic zone',
    '200-1000m': 'Mesopelagic zone',
    '>1000m': 'Bathyabyssopelagic zone'
}

loadings_by_depth = {}

# Correlation between standardized variables and PC scores by depth group
for group_name, condition in depth_groups.items():
    idx = df_combined['depth'].apply(condition)
    x_subset = x_scaled_df.loc[idx]
    pcs_subset = pc_df.loc[idx]

    loadings_group = pd.DataFrame({
        f'PC{i+1}': x_subset.corrwith(pcs_subset[f'PC{i+1}']) for i in range(3)
    })

    loadings_by_depth[group_name] = loadings_group


# Plot correlations vars-PCs
order = ['0-40m', '40-200m', '200-1000m', '>1000m']

fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
axes = axes.ravel()

heatmap_artist = None

for ax, group_name in zip(axes, order):
    loadings_grp = loadings_by_depth[group_name]

    hm = sns.heatmap(
        loadings_grp.T,
        ax=ax,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=False
    )

    if heatmap_artist is None:
        heatmap_artist = hm.collections[0]

    ax.set_title(biozone_titles[group_name])

cbar = fig.colorbar(
    heatmap_artist,
    ax=axes,
    orientation="horizontal",
    fraction=0.03,
    pad=0.06
)
cbar.set_label("Correlation (variable vs PC score)")
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

plt.savefig("data/outputs/pca_corr_loadings.png", dpi=300)
plt.show()
plt.close()


## ---- (b) Sensitivity analysis thresholds ---- ##

# Load GeoDataFrame
geodf_full = gpd.read_file('data/outputs/3d_package.gpkg', layer='geodf').set_index(['depth', 'lat', 'lon'])

# Define depth ranges you want to analyze
depth_ranges = [
    (0, 50),
    (30, 250),
    (200, 1250),
    (1000, 5500)
]
depth_range_labels = ['Euphotic', 'Mesophotic', 'Mesopelagic', 'Bathyabssopelagic']

# Define thresholds/bin widths
bin_widths = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]


# Function to run the analysis for a single depth range
def analyze_depth_range(depth_min, depth_max, label):

    range_prefix = f"{depth_min}-{depth_max}"

    # Filter geodf
    geodf = geodf_full[
        (geodf_full.index.get_level_values('depth') >= depth_min) &
        (geodf_full.index.get_level_values('depth') <= depth_max)
    ].copy()

    # Storage lists for results
    median_min_distances = []
    q1_min_distances = []
    q3_min_distances = []
    proportion_no_analog = []

    # Iterate over each threshold/bin width
    for threshold in bin_widths:
        threshold_label = f"t{threshold}".replace('.', '')
        file_path = f"data/outputs/{range_prefix}_dist_vel_{threshold_label}.nc"

        if os.path.exists(file_path):
            data = xr.open_dataset(file_path)
            ab_dist_fw = data['ab_dist_fw'].values

            # Compute stats
            median_min_distances.append(np.nanmedian(ab_dist_fw))
            q1_min_distances.append(np.nanpercentile(ab_dist_fw, 25))
            q3_min_distances.append(np.nanpercentile(ab_dist_fw, 75))

            # Proportion of no-analogue climates
            valid_coordinates = data.to_dataframe().dropna().reset_index()
            valid_coords_set = set(valid_coordinates[["depth", "lat", "lon"]].itertuples(index=False, name=None))
            geodf_index_set = set(geodf.index.to_list())

            missing_coords = geodf_index_set - valid_coords_set
            proportion_no_analog.append(len(missing_coords) / len(geodf_index_set))

            data.close()

        else:
            print(f"No file found for threshold {threshold} in range {range_prefix}")
            median_min_distances.append(np.nan)
            q1_min_distances.append(np.nan)
            q3_min_distances.append(np.nan)
            proportion_no_analog.append(np.nan)

    # Plot results for this depth range
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    label_fontsize = 18
    tick_fontsize = 16

    # Median distance plot
    axes[0].plot(bin_widths, median_min_distances, marker='o', linestyle='-', color='black')
    axes[0].fill_between(bin_widths, q1_min_distances, q3_min_distances, color='grey', alpha=0.2)
    axes[0].set_xlabel("Threshold", fontsize=label_fontsize)
    axes[0].set_ylabel("Median Distance to Climate Match (km)", fontsize=label_fontsize)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0].grid(True)

    # Proportion no-analog plot
    axes[1].plot(bin_widths, proportion_no_analog, marker='o', linestyle='-', color='black')
    axes[1].set_xlabel("Threshold", fontsize=label_fontsize)
    axes[1].set_ylabel("Proportion of No-Analog Climates", fontsize=label_fontsize)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].grid(True)

    fig.suptitle(f"{label}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"data/outputs/{range_prefix}_sensitivity_analysis.png", dpi=300)
    plt.show()
    plt.close()


# Run the loop for all depth ranges
for (depth_min, depth_max), label in zip(depth_ranges, depth_range_labels):
    analyze_depth_range(depth_min, depth_max, label)
