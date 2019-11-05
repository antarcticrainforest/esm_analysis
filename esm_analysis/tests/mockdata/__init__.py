import datetime

import h5netcdf
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

    with open(str(fname), 'w' ) as f:
        f.write(griddes)

def get_weights(filename, ncells=512):
    with h5netcdf.File(str(filename), 'w') as f:
        f.dimensions = dict(
            ncells=ncells,
            nv=3,
            bnds=2,
            time=1,
            level=1)
        f.create_variable('time', ('time', ), float)
        f.create_variable('clon', ('ncells', ), float)
        f.create_variable('clat', ('ncells', ), float)
        f.create_variable('level', ('level', ), float)
        f.create_variable('clon_bnds', ('ncells', 'nv'), float)
        f.create_variable('clat_bnds', ('ncells', 'nv'), float)
        f.create_variable('o3', ('time', 'level', 'ncells'), float)
        f['time'].attrs['units'] = 'months since 1850-1-1 00:00:00'
        f['time'].attrs['long_name'] = 'Time'
        f['time'].attrs['standard_name'] = 'Time'
        f['time'].attrs['calendar'] = 'standard'
        f['time'].attrs['axis'] = 'T'
        f['time'][:] = 0.
        f['level'][:] = 0.5
        f['level'].attrs['standard_name'] = "air_pressure"
        f['level'].attrs['long_name']  ='Pressue'
        f['level'].attrs['units'] = 'Pa'
        f['level'].attrs['positive'] = 'down'
        f['level'].attrs['axis'] = 'Z'

        f['o3'][:] = 0.0
        f['o3'].attrs['standard_name'] = 'mole_fraction_of_ozone_in_air'
        f['o3'].attrs['long_name'] = 'O3'
        f['o3'].attrs['untis'] = 'ppmv'
        f['o3'].attrs['CDI_grid_type'] = "unstructured"
        f['o3'].attrs['coordinates'] = 'clat clon'
        clat = np.array([ -0.26179939,  1.04719755, -0.26179939,  1.04719755,
                         -0.26179939,  1.04719755, -0.26179939, -2.0943951])

        clon = np.array([ 4.90873852e-02, -9.81747704e-02,  4.90873852e-02,
                         0.00000000e+00,  4.90873852e-02, -9.81747704e-02,
                         4.90873852e-02,  9.81747704e-02])
        f['clon'][:] = np.tile(clon, int(ncells/clon.shape[0]))
        f['clat'][:] = np.tile(clat, int(ncells/clat.shape[0]))
        f['clon'].attrs['units'] = 'radian'
        f['clat'].attrs['units'] = 'radian'
        f['clon'].attrs['standard_name'] = 'longitude'
        f['clon'].attrs['long_name'] = 'center longitude'
        f['clon'].attrs['bounds'] = 'clon_bnds'
        f['clat'].attrs['standard_name'] = 'latitude'
        f['clat'].attrs['long_name'] = 'center latitude'
        f['clat'].attrs['bounds'] = 'clat_bnds'

        f.attrs['uuidOfHGrid'] = "6717c462-ad67-11e9-a8ed-fb16578acc5b"
        f.attrs['Conventions'] = "CF-1.6"

def write_file(filename, variables, ntimesteps, firststep=None, dt='10min'):
    """Create a newfile with a given list of variables."""

    firststep = firststep or datetime.datetime.now()
    timesteps = pd.date_range(start=firststep, periods=ntimesteps, 
            freq=dt)
    ref_time = pd.Timestamp(1970,1,1,0,0,0)
    times = (pd.DatetimeIndex(timesteps) - ref_time).total_seconds()

    with h5netcdf.File(filename, 'w') as f:
        f.dimensions = dict(
                ncells=512)
        f.dimensions['time'] = len(times)

        time = f.create_variable('time', ('time',), float)
        f['time'][:] = times.astype('i')
        f['time'].attrs['units'] = 'Seconds since 1970-01-01 00:00:00'
        f['time'].attrs['long_name'] = 'Time'
        f['time'].attrs['axis'] = 'T'
        f['time'].attrs['CDI_grid_type'] = 'unstructured'
        f['time'].attrs['number_of_grid_in_reference'] = 1
        for varn in variables:
            var = f.create_variable(varn, ('time', 'ncells'), float)
            f[varn][:] = gaussian(f.dimensions['ncells'])
            f[varn].attrs['CDI_grid_type'] = 'unstructured'
            f[varn].attrs['number_of_grid_in_reference'] = 1
        f.attrs['number_of_grid_used'] = 42
        f.attrs['uuidOfVGrid'] = "9f6795b0-aad5-338d-cce3-52b56b9a1d40"
        f.attrs['uuidOfHGrid'] = "6717c462-ad67-11e9-a8ed-fb16578acc5b"

def gaussian(size):
    mu = size/2.
    sigma = size * 0.15 / 2.335
    return  np.exp(-(np.arange(size)-mu)**2/ (2*sigma**2))

