# Comet-tail
This repository provides access to software for working with astronomical images


# Solar Magnetic Field Visualization and 13P/Olbers Tools

This repository contains Python scripts for visualizing the Sun’s radial magnetic field (Br) using MHDweb data and analyzing sky coordinates in astronomical images of comet 13P/Olbers.

## Repository contents

### `Visualization Br for several date.py`

Downloads and visualizes Br data across a configurable date range. It can create an interactive HTML animation and display the data in spherical coordinates or as longitude, polar angle, and radius.

Configure the date range, time step, model, solution type, longitude sector, and output directory at the beginning of the script. The selected model must provide data for the requested dates.
python -m pip install playwright
python -m playwright install chromium
python -m pip install Pillow

### `Visualization Br for specific date.py`

Creates an interactive visualization for a single date. The script’s settings control the model, solution type, radial range, angular sector, color scale, and plot format.
python -m pip install playwright
python -m playwright install chromium
python -m pip install Pillow
### `Automatic load Dec Ra from AstrometryNet.py`

Opens the image referenced in the latest complete Astrometry.net log entry. Click a point in the image to calculate its right ascension (RA) and declination (Dec). When a local WCS FITS solution is available, the script uses it; otherwise, it estimates coordinates from the image center, size, and orientation.

The script can compare the selected position with MPC and JPL Horizons observation catalogs. It records measurements in `measured_positions.csv` and appends details to the Astrometry.net log.

## Requirements

Use Python 3.

### Magnetic field visualization

Install the required packages:

```bash
python -m pip install numpy plotly pyhdf astroquery astropy sunpy
