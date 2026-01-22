"""
Part 2 — Bioclimatic variable computation.

What this script does
---------------------
1) Computes bioclimatic variables from temperature datasets (dat1 = present, dat2 = future).
2) Saves bioclimatic variables as NetCDF and as tabular CSV for downstream PCA.

Inputs
------
- dat1: xarray DataArray of temperature (present period; output of Part 1)
- dat2: xarray DataArray of temperature (future period; output of Part 1)
- prefix: string used to name outputs (same as Part 1)
- Required Python packages: numpy, pandas, xarray, geopandas

Outputs
-------
- NetCDF bioclimatic variables:
  - data/outputs/<prefix>_t1_bio.nc
  - data/outputs/<prefix>_t2_bio.nc
- CSV tables (pre-PCA):
  - data/outputs/pre-pca_dft1.csv
  - data/outputs/pre-pca_dft2.csv

"""


## Import necessary libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd



# Load temperature datasets (output of Part 1) 

if 'dat1' not in globals() or 'dat2' not in globals():
    dat1 = xr.open_dataarray(f"data/outputs/{prefix}_present_thetao.nc")
    dat2 = xr.open_dataarray(f"data/outputs/{prefix}_future_thetao.nc")

# Ensure output directories exist
os.makedirs("data/outputs", exist_ok=True)
os.makedirs("data/outputs/biovars", exist_ok=True)


# Function to compute bioclimatic variables from temperature dataset
def bioclimatic(da):
    ares = da.resample(time='1YS')  # Annual resampling
    mres = da.resample(time='1MS')  # Monthly resampling

    ada = xr.Dataset({
        'bio1': ares.mean(),                                         # Annual Mean Temperature
        'bio2': (mres.max('time') - mres.min('time'))                 # Annual Mean Diurnal Range (monthly max-min, then annual mean)
                .resample(time='1YS').mean(),
        'bio4': mres.mean('time').resample(time='1YS').std(),         # Temperature Seasonality (SD of monthly means)
        'bio5': ares.max('time'),                                     # Maximum temperature
        'bio6': ares.min('time'),                                     # Minimum temperature
        'bio10': (mres.mean('time')                                   # Mean temperature of warmest quarter
                  .rolling(time=3, min_periods=1).mean()
                  .resample(time='1YS').max()),
        'bio11': (mres.mean('time')                                   # Mean temperature of coldest quarter
                  .rolling(time=3, min_periods=1).mean()
                  .resample(time='1YS').min())
    })

    ada['bio7'] = ada.bio5 - ada.bio6                                 # Annual temperature range
    ada['bio3'] = ada.bio2 / ada.bio7 * 100                           # Isothermality

    return ada.mean(dim="time")  # Mean over years within the selected period


# Compute and save bioclimatic variables 
# (dat1 = present)
biot1 = bioclimatic(dat1)

# Convert to DataFrame for PCA
dft1 = biot1.to_dataframe().dropna()
if "spatial_ref" in dft1.columns:
    dft1 = dft1.drop(columns="spatial_ref")

geodf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(dft1.reset_index().lon, dft1.reset_index().lat),
    crs="epsg:4326",
    index=dft1.index
).to_crs(epsg=3035)

biot1.to_netcdf(f"data/outputs/{prefix}_t1_bio.nc")
dft1.to_csv("data/outputs/pre-pca_dft1.csv")


# (dat2 = future)
biot2 = bioclimatic(dat2)

dft2 = biot2.to_dataframe().dropna()
if "spatial_ref" in dft2.columns:
    dft2 = dft2.drop(columns="spatial_ref")

# Safety check: same spatial index in both periods
if not dft1.index.equals(dft2.index):
    raise Exception("dft1 and dft2 indexes are different")

biot2.to_netcdf(f"data/outputs/{prefix}_t2_bio.nc")
dft2.to_csv("data/outputs/pre-pca_dft2.csv")
