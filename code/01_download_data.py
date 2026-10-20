"""
Step 1 — Download and prepare 3D sea water temperature projections

Companion code to Rizzi et al. (2026)

What this script does
---------------------
1) Downloads daily sea water potential temperature (thetao) over the whole water column from the Copernicus Climate Data Store (dataset "sis-marine-properties", POLCOMS-ERSEM model, RCP8.5), one month at a time, and saves each month as a NetCDF file
2) Merges the monthly files into two periods (present and future), masks invalid values and the Atlantic, and saves one NetCDF per period


Inputs
------
- a Copernicus CDS account with the API key configured (~/.cdsapirc)


Outputs (in output_dir)
-------
- {prefix}_present_thetao.nc : temperature for the present period
- {prefix}_future_thetao.nc  : temperature for the future period

These are the inputs of the climate velocity calculation (step 2).

"""


## IMPORTS
import os
import shutil

import numpy as np
import cdsapi
import xarray as xr
import rioxarray


## CONFIGURATION ##
# Time periods
# Present: first_year to split_year - 1; Future: split_year to last_year.
# Both periods have the same length (25 years), which step 5 uses as n_years
first_year = 2006
split_year = 2031
last_year  = 2055

present_years = range(first_year, split_year)       # 2006–2030
future_years  = range(split_year, last_year + 1)    # 2031–2055

# Prefix of the output files (e.g. "3d_2006-2055")
prefix = f"3d_{first_year}-{last_year}"


# CDS request
# Year and month are filled in for each request inside the download loop.
cds_dataset = "sis-marine-properties"
dsconfig = {
    "format":              "zip",
    "origin":              "polcoms_ersem",                     # model
    "vertical_resolution": "water_column",                      # all depth levels
    "time_aggregation":    "day",                               # daily values
    "variable":            "sea_water_potential_temperature",
    "experiment":          "rcp8_5",                            # climate scenario
    "area":                [45.8, -6.0, 30.2, 36.3],            # North, West, South, East (degrees)
    "year":                [],
    "month":               [],
}


# Atlantic mask
# Cells north of the line lat = 43 + lon * 2/5 are removed. This excludes the
# Atlantic part of the download box (Bay of Biscay) while keeping the
# Mediterranean.
atlantic_mask_intercept = 43
atlantic_mask_slope     = 2 / 5
 
# Values at or above this threshold are model fill values and are removed.
fill_value_threshold = 10000


# Paths
monthly_dir = "data/temp_arrays/"                                   # one NetCDF per month
zipfile     = "data/download/sea_water_potential_temperature.zip"   # temporary download
extractdir  = "data/download/extracted"                             # temporary extraction
output_dir  = "data/outputs/"                                       # period files



## FUNCTIONS ##
def ensure_directories():
    """Create the working and output directories if they do not exist."""
    for directory in [monthly_dir, os.path.dirname(zipfile), extractdir, output_dir]:
        os.makedirs(directory, exist_ok=True)
        
def clean_all(zipfile, extractdir):
    """Remove the temporary ZIP and extracted files from previous downloads."""
    if os.path.exists(zipfile):
        os.remove(zipfile)
 
    if os.path.exists(extractdir):
        for f in os.listdir(extractdir):
            fp = os.path.join(extractdir, f)
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)
 
        if len(os.listdir(extractdir)) == 0:
            os.rmdir(extractdir)

def monthly_file(year, month):
    """Path of the NetCDF file for one month."""
    return os.path.join(monthly_dir, f"thetao-{year}-{month:02}.nc")
 
 
def download_monthly_files(years):
    """Download one month at a time from the CDS and keep only thetao; existing months are skipped."""
    client = cdsapi.Client()
 
    for year in years:
        for month in range(1, 13):
            outfile = monthly_file(year, month)
 
            if os.path.exists(outfile):
                continue
 
            print("download", outfile)
 
            dsconfig["year"]  = [str(year)]
            dsconfig["month"] = [f"{month:02}"]
            client.retrieve(cds_dataset, dsconfig, zipfile)
 
            # Each ZIP is expected to contain a single NetCDF file
            shutil.unpack_archive(zipfile, extractdir)
            extractfile = os.path.join(extractdir, os.listdir(extractdir)[0])
 
            ds = xr.open_dataset(extractfile)
            ds.thetao.to_netcdf(outfile)
            ds.close()
 
            clean_all(zipfile, extractdir)

def get_temperature(years):
    """Merge the monthly files of a period, georeference them and mask fill values and the Atlantic."""
    files = [monthly_file(y, m) for y in years for m in range(1, 13)]
 
    da = (
        xr.open_mfdataset(files, engine="h5netcdf", mask_and_scale=True)
        .thetao
        .rio.write_crs(4326)
        .rio.write_transform()
        .rio.write_coordinate_system()
    )
 
    return (
        da.where(da < fill_value_threshold)
        .where(da.lat < (atlantic_mask_intercept + da.lon * atlantic_mask_slope))
        .rio.write_nodata(np.nan)
    )


## MAIN ##
if __name__ == "__main__":
    ensure_directories()
    clean_all(zipfile, extractdir)
 
    # 1. Download all months of both periods
    download_monthly_files(range(first_year, last_year + 1))
 
    # 2. Build and save one file per period
    get_temperature(present_years).to_netcdf(os.path.join(output_dir, f"{prefix}_present_thetao.nc"))
    get_temperature(future_years).to_netcdf(os.path.join(output_dir, f"{prefix}_future_thetao.nc"))