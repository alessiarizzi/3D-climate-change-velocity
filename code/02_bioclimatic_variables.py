"""
Step 2 — Bioclimatic variables for the present and future periods

Companion code to Rizzi et al. (2026)

What this script does
---------------------
From the daily 3D temperature of each period (output of step 1), it computes a set of bioclimatic variables for every grid cell (depth, lat, lon), averaged over the years of the period. These variables describe the thermal conditions of each cell and are used in step 3 to search for climate analogues. 

Variables (named after the WorldClim convention, computed on daily values):
    bio1   annual mean temperature
    bio2   mean monthly range (monthly max - monthly min, averaged per year)
    bio3   isothermality (bio2 / bio7 * 100)
    bio4   temperature seasonality (standard deviation of monthly means)
    bio5   maximum temperature
    bio6   minimum temperature
    bio7   annual temperature range (bio5 - bio6)
    bio10  mean temperature of the warmest quarter (3 consecutive months)
    bio11  mean temperature of the coldest quarter (3 consecutive months)


Inputs
------
- {prefix}_present_thetao.nc, {prefix}_future_thetao.nc


Outputs
-------
- {prefix}_t1_bio.nc, {prefix}_t2_bio.nc : bioclimatic variables as grids
- pre-pca_dft1.csv, pre-pca_dft2.csv     : same values as tables (one row per ocean cell), input of step 3

"""


## IMPORTS
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd


## CONFIGURATION ##
# Must match the prefix of step 1 ("3d_{first_year}-{last_year}")
prefix = "3d_2006-2055"
 
data_dir = "data/outputs"
package_file = os.path.join(data_dir, "3d_package.gpkg")


## FUNCTIONS ##
def bioclimatic(da):
    """Compute the bioclimatic variables of a daily temperature DataArray, averaged over its years."""
    ares = da.resample(time="1YS")   # yearly groups
    mres = da.resample(time="1MS")   # monthly groups
 
    # Quarters are 3-month moving means of monthly means; windows can span two
    # calendar years (e.g. Dec-Jan-Feb), and the first months of the series use
    # the months available so far
    quarter_means = mres.mean("time").rolling(time=3, min_periods=1).mean()
 
    ada = xr.Dataset({
        "bio1":  ares.mean(),
        "bio2":  (mres.max("time") - mres.min("time")).resample(time="1YS").mean(),
        "bio4":  mres.mean("time").resample(time="1YS").std(),
        "bio5":  ares.max("time"),
        "bio6":  ares.min("time"),
        "bio10": quarter_means.resample(time="1YS").max(),
        "bio11": quarter_means.resample(time="1YS").min(),
    })
 
    ada["bio7"] = ada.bio5 - ada.bio6
    ada["bio3"] = ada.bio2 / ada.bio7 * 100
 
    # Mean over the years of the period
    return ada.mean(dim="time")
 
 
def to_table(bio_ds):
    """Convert a bioclimatic Dataset to a table with one row per cell, keeping only cells with all variables."""
    df = bio_ds.to_dataframe().dropna()
    if "spatial_ref" in df.columns:
        df = df.drop(columns="spatial_ref")
    return df



## MAIN ##
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)
 
    # Present period
    present = xr.open_dataarray(os.path.join(data_dir, f"{prefix}_present_thetao.nc"))
    biot1   = bioclimatic(present)
    dft1    = to_table(biot1)
 
    biot1.to_netcdf(os.path.join(data_dir, f"{prefix}_t1_bio.nc"))
    dft1.to_csv(os.path.join(data_dir, "pre-pca_dft1.csv"))
 
    # Cell positions projected to EPSG:3035 (metres), used for distances in step 3
    cells = dft1.reset_index()
    geodf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(cells.lon, cells.lat),
        crs="epsg:4326",
        index=dft1.index
    ).to_crs(epsg=3035)
    geodf.to_file(package_file, layer="geodf", driver="GPKG")
 
    # Future period
    future = xr.open_dataarray(os.path.join(data_dir, f"{prefix}_future_thetao.nc"))
    biot2  = bioclimatic(future)
    dft2   = to_table(biot2)
 
    # Both periods must cover exactly the same cells for the analogue search
    if not dft1.index.equals(dft2.index):
        raise ValueError("Present and future tables cover different cells (dft1 and dft2 indexes differ).")
 
    biot2.to_netcdf(os.path.join(data_dir, f"{prefix}_t2_bio.nc"))
    dft2.to_csv(os.path.join(data_dir, "pre-pca_dft2.csv"))
