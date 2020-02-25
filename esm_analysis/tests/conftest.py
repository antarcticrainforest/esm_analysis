"""Define mock objects for testing."""
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pytest

from .mockdata import (create_grid, get_weights, write_file)


def write_ncfiles(path, suffix='.nc'):
    """Create moke netcdf files with 10 days of data."""
    import pandas as pd
    dates = pd.date_range(datetime.date.today(), periods=10, freq='1D')
    for d in dates:
        fname = 'test_{}Z{}'.format(d.strftime("%Y%m%d"), suffix)
        write_file(Path(path) / fname, ('t_2m', 'pres_sfc'), 24,
                   firststep=d, dt='1H')


@pytest.fixture(scope='function')
def mock_slurm(monkeypatch):
    """Set environment variables to call mock slurm commands."""
    import os
    this_dir = os.path.abspath(os.path.dirname(__file__))
    PATH = os.path.join(this_dir, 'mockcmds')
    monkeypatch.setenv("JOB_ID", "19628442")
    monkeypatch.setenv("STATUS", 'PD')
    monkeypatch.setenv("PATH", PATH, prepend=os.pathsep)


@pytest.fixture(scope='session')
def mock_workdir():
    """Create a temp-dir."""
    with TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope='function')
def mock_client():
    """Create a distributed client."""
    from dask.distributed import Client
    yield Client()


@pytest.fixture(scope='session')
def mock_timedir():
    """Create a directory that contains netcdf files."""
    with TemporaryDirectory() as td:
        write_ncfiles(td)
        with open(str(Path(td) / 'NAMELIST_test_testing.nml'), 'w') as f:
            f.write("""
&config
    input = 'wind.nc'
    steps = 432
    layout = 8, 16
    visc = 1.0e-4
    use_biharmonic = .false.
/
""")
        yield td


@pytest.fixture(scope='module')
def mock_vardir():
    """Create a directory that contains nc files of different variables."""
    today = datetime.date.today()
    with TemporaryDirectory() as td:
        for v in ('t_2m', 'pres_sfc', 'rain_gsp_rate'):
            fname = 'test_{}_{}Z.nc'.format(v, today.strftime("%Y%m%d"))
            write_file(Path(td) / fname, (v, ), 144,
                       firststep=today, dt='10min')
        yield td


@pytest.fixture(scope='session')
def mockgrid():
    """Create a mock grid file for regrdding."""
    with NamedTemporaryFile(suffix='.grid') as tf:
        create_grid(tf.name)
        yield tf.name


@pytest.fixture(scope='session')
def mock_grb_dir():
    """Pretend to create grib files."""
    with TemporaryDirectory() as td:
        write_ncfiles(td, suffix='.grb')
        yield td


@pytest.fixture(scope='session')
def mockweights():
    """Create weight files."""
    with NamedTemporaryFile(suffix='.nc') as tf:
        get_weights(tf.name)
        yield tf.name


@pytest.fixture(scope='session')
def mock_run(mockgrid, mock_timedir, mockweights, esm_analysis):
    """Create a mock RunDirectory object."""
    run = esm_analysis.RunDirectory.gen_weights(mockgrid,
                                                mock_timedir,
                                                prefix='test',
                                                model_type='DWD',
                                                infile=mockweights)
    yield run


@pytest.fixture(scope='session')
def mock_dataset(mock_timedir):
    """Read a mock dataset with xarray."""
    import xarray as xr
    yield xr.open_mfdataset(str(Path(mock_timedir) / '*.nc'),
                            combine='by_coords')


@pytest.fixture(scope='session')
def mock_tmpdir():
    """Create a another temp dir."""
    with TemporaryDirectory() as td:
        yield td


@pytest.fixture(scope='session')
def esm_analysis():
    """Manipulate the cachedir of esm_analysis."""
    import esm_analysis
    with TemporaryDirectory() as cache_dir:
        esm_analysis.Reader._cache_dir = Path(cache_dir)
        esm_analysis.cacheing._cache_dir = Path(cache_dir)
        yield esm_analysis


@pytest.fixture
def spec_hum():
    """Define target specific humidity."""
    yield 7.8526e-3


@pytest.fixture
def mixing_r():
    """Define target mixing ratio."""
    yield 7.9148e-3


@pytest.fixture
def temp_c():
    """Define target temperature."""
    yield 25.


@pytest.fixture
def rh():
    """Define target relative humidity."""
    yield 40


@pytest.fixture
def pres():
    """Define target pressure."""
    yield 1013.25


def model_config(config=None):
    """Create a model configuration."""
    conf = '''
    title = "This could ba a descriptive title."

    {}
    '''.format(config or '')
    print(conf)
    return conf


@pytest.fixture
def model_setup_with_config():
    """Create a model setup config file."""
    conf_str = '''
    [config]
        mpt01 = "Some fancy model setup"
        mpt02 = "An even fancier model setup"
    '''
    with NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as f:
            f.write(model_config(conf_str))
        yield tf.name


@pytest.fixture
def model_setup_without_config():
    """Create a model setup without configuration."""
    with NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as f:
            f.write(model_config())
        yield tf.name
