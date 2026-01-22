"""
Part 1 — Download and prepare Copernicus CDS marine temperature data.

What this script does
---------------------
1) Downloads monthly Copernicus CDS data ('sis-marine-properties') for the period 2006–2055.
2) Extracts sea water potential temperature (thetao) and stores one NetCDF per month.
3) Loads the monthly NetCDF files and builds two aggregated xarray DataArrays:
   - Present period (2006–2030)
   - Future period  (2031–2055)
4) Saves the aggregated datasets to NetCDF for use in subsequent analyses.

Inputs
------
- Copernicus CDS API access configured on your machine (cdsapi + ~/.cdsapirc).
- Dataset: 'sis-marine-properties' (origin: polcoms_ersem),
  variable: sea_water_potential_temperature.
- Required Python packages: cdsapi, xarray, rioxarray, numpy.

Outputs
-------
- Monthly files:  data/temp_arrays/sst-YYYY-MM.nc
- Aggregated files:
  - data/outputs/<prefix>_present_thetao.nc
  - data/outputs/<prefix>_future_thetao.nc
"""


## Import necessary libraries
import os
import shutil

import numpy as np
import cdsapi
import xarray as xr
import rioxarray



# Create the CDS API Client
c = cdsapi.Client()

# Data request configuration (year and month are updated inside the loop)
dsconfig = {
    'format': 'zip',                                   # Download in ZIP format
    'origin': 'polcoms_ersem',                         # Data source within the CDS
    'vertical_resolution': 'water_column',             # Vertical resolution of data
    'time_aggregation': 'day',                         # Temporal aggregation of data
    'variable': 'sea_water_potential_temperature',     # Climate variable to retrieve 
    'year': [],
    'month': [],
    'experiment': 'rcp8_5',                            # Climate scenario
    'area': [45.8, -6.0, 30.2, 36.3],                  # Geographical area (North, West, South, East) - this exclude the Atlantic
}

# Paths for saving data
sstdir = 'data/temp_arrays/'                                    # Monthly NetCDF output directory
zipfile = 'data/download/sea_water_potential_temperature.zip'   # Temporary ZIP file path
extractdir = 'data/download/extracted'                          # Temporary extraction directory
output_dir = 'data/outputs/'                                    # Outputs directory

# Create necessary directories
def ensure_directories():
    directories = [
        sstdir,
        os.path.dirname(zipfile),
        extractdir,
        output_dir
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

ensure_directories()


# Remove temporary files from previous runs (zip + extracted files)
def clean_all(zipfile, extractdir):
    if os.path.exists(zipfile):
        os.remove(zipfile)

    if os.path.exists(extractdir):
        for f in os.listdir(extractdir):
            fp = os.path.join(extractdir, f)
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)

        # Remove extraction directory if empty
        if len(os.listdir(extractdir)) == 0:
            os.rmdir(extractdir)

clean_all(zipfile, extractdir)



# Download and save monthly temperature files
for year in range(2006, 2056):
    _year = str(year)

    for month in range(1, 13):
        _month = str(month).zfill(2)
        sstfile = os.path.join(sstdir, f'sst-{_year}-{_month}.nc')

        # Skip if file already exists
        if os.path.exists(sstfile):
            continue

        print("download", sstfile)

        # Update year and month in configuration
        dsconfig['month'] = [_month]
        dsconfig['year'] = [_year]

        # Download data from Copernicus CDS API
        c.retrieve(
            'sis-marine-properties',
            dsconfig,
            zipfile
        )

        # Unpack ZIP file
        shutil.unpack_archive(zipfile, extractdir)

        # Get extracted file (assumes one main file per zip)
        extractfile = os.path.join(extractdir, os.listdir(extractdir)[0])

        # Process data and save only thetao
        ds = xr.open_dataset(extractfile)
        ds.thetao.to_netcdf(sstfile)
        ds.close()

        # Clean up temporary files after processing
        clean_all(zipfile, extractdir)


# Generate file paths for climatic data
def get_sstfiles(years, months=range(1, 13)):
    return [os.path.join(sstdir, f"sst-{y}-{m:02}.nc") for y in years for m in months]


# Load and process climatic data from multiple files
def get_sst(years, months=range(1, 13), depths=slice(None)):
    _files = get_sstfiles(years, months)

    da = (
        xr.open_mfdataset(_files, engine='h5netcdf', mask_and_scale=True)
        .thetao
        .isel(depth=depths)
        .rio.write_crs(4326)
        .rio.write_transform()
        .rio.write_coordinate_system()
    )

    return (
        da.where(da < 10000)                             # Remove invalid values
        .where(da.lat < (43 + da.lon * 2 / 5))           # Heuristic mask to exclude Atlantic 
        .rio.write_nodata(np.nan)
    )


# Build and save present and future datasets
prefix = '3d_2006-2055'  # !!! change prefix if needed !!!

dat1 = get_sst(range(2006, 2031))  # Present: 2006–2030
dat2 = get_sst(range(2031, 2056))  # Future: 2031–2055

dat1.to_netcdf(os.path.join(output_dir, f"{prefix}_present_thetao.nc"))
dat2.to_netcdf(os.path.join(output_dir, f"{prefix}_future_thetao.nc"))