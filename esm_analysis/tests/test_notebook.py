"""Test the notebook examples."""


import pytest


def test_import():
    """Test importing all external libraries."""
    from getpass import getuser # Libaray to copy things
    from pathlib import Path # Object oriented libary to deal with paths
    from tempfile import NamedTemporaryFile, TemporaryDirectory # Creating temporary Files/Dirs
    from subprocess import run, PIPE
    from cartopy import crs as ccrs # Cartography library
    import dask # Distributed data libary
    from distributed import Client, progress, wait # Libaray to orchestrate distributed resources
    from hurry.filesize import size as filesize # Get human readable file sizes
    from matplotlib import pyplot as plt # Standard Plotting library
    from metpy import calc as metcalc # Calculate atmospheric variables
    from metpy.units import units as metunits # Easy to use meteorological units
    import numpy as np # Standard array library
    import pandas as pd # Libary to work with labeled data frames and time series
    import seaborn as sns # Makes plots more beautiful
    import xarray as xr # Libary to work with labeled n-dimensional data and dask


