"""
Step 4 — Diagnostics: PCA interpretation and analogue threshold sensitivity. 

Companion code to Rizzi et al. (2026)

What this script does
---------------------
(a) PCA interpretation. Runs a PCA on the bioclimatic variables of both periods and plots:
    - the cumulative explained variance and the scree plot, to justify the number of components kept in step 3;
    - the correlation between each bioclimatic variable and the first three components, for each biozone, to interpret the components. 

(b) Threshold sensitivity. For each biozone, reads the step-3 outputs obtained with different analogue thresholds and plot, against the threshold:
    - the median (and interquartile range) of the distance to the analogue;
    - the proportion of cells with no analogue.
    It is used to justify the threshold chosen in step 3 (nn_threshold). 
    Each threshold in sensitivity_thresholds needs its own step-3 run; missing runs are reported and left blank in the plots. 


Inputs
------
- pre-pca_dft1.csv, pre-pca_dft2.csv  : bioclimatic tables (step 2)
- 3d_package.gpkg, layer "geodf"      : cell positions (step 2)
- {range}_dist_vel_{threshold}.nc     : step-3 outputs, one per threshold

Outputs
-------
- explain_var_scree_plots.png
- pca_corr_loadings.png
- {range}_sensitivity_analysis.png; one per biozone

"""


## IMPORTS
import os
 
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## CONFIGURATION
data_dir     = "data/outputs"
package_file = os.path.join(data_dir, "3d_package.gpkg")
 
# Number of components whose correlations with the variables are shown
# (the components kept in step 3)
n_components_kept = 3
 
# Depth ranges of step 3 (m), one per biozone
depth_ranges = {
    "Euphotic":           (0,    50),
    "Mesophotic":         (30,   250),
    "Mesopelagic":        (200,  1250),
    "Bathyabyssopelagic": (1000, 5500),
}
 
# Depth selection used for the PCA correlation plot, per biozone
biozone_depth_filters = {
    "Euphotic":           lambda d: d <= 50,
    "Mesophotic":         lambda d: (d >= 30) & (d <= 250),
    "Mesopelagic":        lambda d: (d >= 200) & (d <= 1250),
    "Bathyabyssopelagic": lambda d: d > 1000,
}
 
# Analogue thresholds tested (distance in the standardised PCA space)
sensitivity_thresholds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
 
show_figures = False   # True to display figures while running



## (a) PCA INTERPRETATION ##
def pca_interpretation():
    """Plot explained variance and variable-component correlations per biozone."""
    dft1 = pd.read_csv(os.path.join(data_dir, "pre-pca_dft1.csv"), index_col=[0, 1, 2])
    dft2 = pd.read_csv(os.path.join(data_dir, "pre-pca_dft2.csv"), index_col=[0, 1, 2])
 
    # PCA with all components on present and future cells together
    df_combined = pd.concat([dft1, dft2], axis=0)
    n_vars      = df_combined.shape[1]
    pc_names    = [f"PC{i+1}" for i in range(n_vars)]
 
    x_scaled    = StandardScaler().fit_transform(df_combined)
    pca_interp  = PCA(n_components=n_vars).fit(x_scaled)
 
    explained_variance = pca_interp.explained_variance_ratio_
    print("Explained variance ratio per component:", explained_variance)
 
    # --- Cumulative explained variance and scree plot ---
    components = range(1, n_vars + 1)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))
 
    axes[0].plot(components, np.cumsum(explained_variance), marker="o", linestyle="-")
    axes[0].axhline(y=0.95, color="g", linestyle="--", label="95% variance explained")
    axes[0].axhline(y=0.99, color="r", linestyle="--", label="99% variance explained")
    axes[0].set_xlabel("Number of principal components", fontsize=16)
    axes[0].set_ylabel("Cumulative explained variance", fontsize=16)
    axes[0].set_title("Cumulative explained variance", fontsize=18)
    axes[0].legend(fontsize=16)
 
    axes[1].plot(components, explained_variance, marker="o", linestyle="--")
    axes[1].set_xlabel("Principal component", fontsize=16)
    axes[1].set_ylabel("Explained variance ratio", fontsize=16)
    axes[1].set_title("Scree plot", fontsize=18)
 
    for ax in axes:
        ax.tick_params(axis="both", labelsize=16)
        ax.grid()
 
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "explain_var_scree_plots.png"), dpi=300)
    if show_figures:
        plt.show()
    plt.close()
 
    # --- Correlation between standardised variables and PC scores, per biozone ---
    x_scaled_df = pd.DataFrame(x_scaled, index=df_combined.index, columns=df_combined.columns)
    pc_df       = pd.DataFrame(pca_interp.transform(x_scaled), index=df_combined.index, columns=pc_names)
    depth       = pd.Series(df_combined.index.get_level_values(0), index=df_combined.index)
 
    correlations = {}
    for biozone, depth_filter in biozone_depth_filters.items():
        idx = depth.apply(depth_filter)
        correlations[biozone] = pd.DataFrame({
            pc: x_scaled_df.loc[idx].corrwith(pc_df.loc[idx, pc])
            for pc in pc_names[:n_components_kept]
        })
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
 
    for ax, (biozone, corr) in zip(axes.ravel(), correlations.items()):
        hm = sns.heatmap(
            corr.T, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, center=0,
            annot=True, fmt=".2f", linewidths=0.5, cbar=False
        )
        ax.set_title(f"{biozone} zone")
 
    cbar = fig.colorbar(hm.collections[0], ax=axes, orientation="horizontal", fraction=0.03, pad=0.06)
    cbar.set_label("Correlation (variable vs PC score)")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
 
    plt.savefig(os.path.join(data_dir, "pca_corr_loadings.png"), dpi=300)
    if show_figures:
        plt.show()
    plt.close()




## (b) THRESHOLD SENSITIVITY ##
def analyze_depth_range(geodf_full, depth_min, depth_max, label):
    """Plot analogue distance and proportion of no-analogue cells against the threshold, for one depth range."""
    range_prefix = f"{depth_min}-{depth_max}"
 
    depthg     = geodf_full.index.get_level_values("depth")
    geodf      = geodf_full[(depthg >= depth_min) & (depthg <= depth_max)]
    geodf_cells = set(geodf.index.to_list())
 
    median_dist, q1_dist, q3_dist, prop_no_analogue = [], [], [], []
 
    for threshold in sensitivity_thresholds:
        threshold_label = f"t{threshold}".replace(".", "")
        file_path       = os.path.join(data_dir, f"{range_prefix}_dist_vel_{threshold_label}.nc")
 
        if not os.path.exists(file_path):
            print(f"No file found for threshold {threshold} in range {range_prefix}")
            for values in (median_dist, q1_dist, q3_dist, prop_no_analogue):
                values.append(np.nan)
            continue
 
        data       = xr.open_dataset(file_path)
        ab_dist_fw = data["ab_dist_fw"].values
 
        # Distance to the analogue (km): median and interquartile range
        median_dist.append(np.nanmedian(ab_dist_fw))
        q1_dist.append(np.nanpercentile(ab_dist_fw, 25))
        q3_dist.append(np.nanpercentile(ab_dist_fw, 75))
 
        # No-analogue cells: ocean cells of the range without a valid result
        valid       = data.to_dataframe().dropna().reset_index()
        valid_cells = set(valid[["depth", "lat", "lon"]].itertuples(index=False, name=None))
        prop_no_analogue.append(len(geodf_cells - valid_cells) / len(geodf_cells))
 
        data.close()
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
 
    axes[0].plot(sensitivity_thresholds, median_dist, marker="o", linestyle="-", color="black")
    axes[0].fill_between(sensitivity_thresholds, q1_dist, q3_dist, color="grey", alpha=0.2)
    axes[0].set_ylabel("Median distance to climate analogue (km)", fontsize=18)
 
    axes[1].plot(sensitivity_thresholds, prop_no_analogue, marker="o", linestyle="-", color="black")
    axes[1].set_ylabel("Proportion of no-analogue cells", fontsize=18)
 
    for ax in axes:
        ax.set_xlabel("Threshold", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True)
 
    fig.suptitle(label, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"{range_prefix}_sensitivity_analysis.png"), dpi=300)
    if show_figures:
        plt.show()
    plt.close()


## MAIN ##
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)

    pca_interpretation()

    geodf_full = gpd.read_file(package_file, layer="geodf").set_index(["depth", "lat", "lon"])
    for label, (depth_min, depth_max) in depth_ranges.items():
        analyze_depth_range(geodf_full, depth_min, depth_max, label)