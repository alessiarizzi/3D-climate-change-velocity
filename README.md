# Analogue-based 3D climate velocity

This repository contains the code used to compute analogue-based 3D climate velocity in marine environments, with the Mediterranean Sea used as a case study.

## Scripts (run in order)

1. **Part 1 — Download and prepare temperature data**
   - Downloads Copernicus CDS marine temperature (thetao) and aggregates present (2006–2030) and future (2031–2055).

2. **Part 2 — Compute bioclimatic variables**
   - Computes bioclimatic variables from present/future temperature datasets and exports NetCDF + CSV tables for PCA.

3. **Part 3 — PCA and analogue-based climate velocity**
   - Performs PCA on bioclimatic variables and computes analogue-based climate velocity.
   - This script is run for a range of climate-space threshold values to generate intermediate outputs.

4. **Part 4 — PCA interpretation and threshold selection**
   - Uses the outputs generated in Part 3 to produce diagnostic plots.
   - These plots are used to select the number of principal components and the climate-space threshold adopted in the final analysis.
     
After parameter selection, Part 3 is rerun using the chosen configuration to produce the final results used in the manuscript.

## Data availability
Input temperature data are obtained from the Copernicus Climate Data Store (CDS) and are not redistributed in this repository.
