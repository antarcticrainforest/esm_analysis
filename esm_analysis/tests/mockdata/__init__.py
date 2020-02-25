"""Definition of mock datasets."""
import datetime

from netCDF4 import Dataset as nc
import numpy as np
import pandas as pd


def create_grid(fname):
    """Create a grid file describing the lon-lat layout."""
    griddes = """
gridtype = lonlat
xsize    =   64
ysize    =   4
xfirst   = -180.0  ! start longitude (fixed)
xinc     =   5.625    ! 360 / xsize
yfirst   =   -1  ! start latitude (fixed)
yinc     =    1    !  20 / ysize"""

    with open(str(fname), 'w') as f:
        f.write(griddes)


def get_weights(filename, ncells=512):
    """Generate a weight file."""
    with nc(str(filename), 'w') as f:
        f.createDimension('ncells', ncells)
        f.createDimension('nv', 3)
        f.createDimension('bnds', 2)
        f.createDimension('time', 1)
        f.createDimension('level', 1)
        f.createVariable('time', 'i', ('time', ))
        f.createVariable('clon', 'f', ('ncells', ))
        f.createVariable('clat', 'f', ('ncells', ))
        f.createVariable('level', 'f', ('level', ))
        f.createVariable('clon_bnds', 'f', ('ncells', 'nv'))
        f.createVariable('clat_bnds', 'f', ('ncells', 'nv'))
        f.createVariable('o3', 'f', ('time', 'level', 'ncells'))
        f.variables['time'].units = 'months since 1850-1-1 00:00:00'
        f.variables['time'].long_name = 'Time'
        f.variables['time'].standard_name = 'Time'
        f.variables['time'].calendar = 'standard'
        f.variables['time'].axis = 'T'
        f.variables['time'][:] = 0
        f.variables['level'][:] = 0.5
        f.variables['level'].standard_name = "air_pressure"
        f.variables['level'].long_name = 'Pressue'
        f.variables['level'].units = 'Pa'
        f.variables['level'].positive = 'down'
        f.variables['level'].axis = 'Z'

        f.variables['o3'][:] = 0.0
        f.variables['o3'].standard_name = 'mole_fraction_of_ozone_in_air'
        f.variables['o3'].long_name = 'O3'
        f.variables['o3'].untis = 'ppmv'
        f.variables['o3'].CDI_grid_type = "unstructured"
        f.variables['o3'].coordinates = 'clat clon'
        clat = np.array([-0.26179939,  1.04719755, -0.26179939,  1.04719755,
                         -0.26179939,  1.04719755, -0.26179939, -2.0943951])

        clon = np.array([4.90873852e-02, -9.81747704e-02,  4.90873852e-02,
                         0.00000000e+00,  4.90873852e-02, -9.81747704e-02,
                         4.90873852e-02,  9.81747704e-02])
        f.variables['clon'][:] = np.tile(clon, int(ncells/clon.shape[0]))
        f.variables['clat'][:] = np.tile(clat, int(ncells/clat.shape[0]))
        f.variables['clon'].units = 'radian'
        f.variables['clat'].units = 'radian'
        f.variables['clon'].standard_name = 'longitude'
        f.variables['clon'].long_name = 'center longitude'
        f.variables['clon'].bounds = 'clon_bnds'
        f.variables['clat'].standard_name = 'latitude'
        f.variables['clat'].long_name = 'center latitude'
        f.variables['clat'].bounds = 'clat_bnds'

        f.uuidOfHGrid = "6717c462-ad67-11e9-a8ed-fb16578acc5b"
        f.Conventions = "CF-1.6"


def write_file(filename, variables, ntimesteps, firststep=None, dt='10min'):
    """Create a newfile with a given list of variables."""
    firststep = firststep or datetime.datetime.now()
    timesteps = pd.date_range(start=firststep, periods=ntimesteps,
                              freq=dt)
    ref_time = pd.Timestamp(1970, 1, 1, 0, 0, 0)
    times = (pd.DatetimeIndex(timesteps) - ref_time).total_seconds()

    with nc(filename, 'w') as f:
        f.createDimension('ncells', 512)
        f.createDimension('time', len(times))
        f.createVariable('time', 'i', ('time',))
        f.variables['time'][:] = np.array(times).astype('i')
        f.variables['time'].units = 'Seconds since 1970-01-01 00:00:00'
        f.variables['time'].long_name = 'Time'
        f.variables['time'].axis = 'T'
        f.variables['time'].CDI_grid_type = 'unstructured'
        f.variables['time'].number_of_grid_in_reference = 1
        for varn in variables:
            f.createVariable(varn, 'f', ('time', 'ncells'))
            f.variables[varn][:] = gaussian(len(f.dimensions['ncells']))
            f.variables[varn].CDI_grid_type = 'unstructured'
            f.variables[varn].number_of_grid_in_reference = 1
        f.number_of_grid_used = 42
        f.uuidOfVGrid = "9f6795b0-aad5-338d-cce3-52b56b9a1d40"
        f.uuidOfHGrid = "6717c462-ad67-11e9-a8ed-fb16578acc5b"


def gaussian(size):
    """Create a 1D gaussian."""
    mu = size/2.
    sigma = size * 0.15 / 2.335
    return np.exp(-(np.arange(size)-mu)**2 / (2*sigma**2))
