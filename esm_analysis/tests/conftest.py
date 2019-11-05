import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pytest

from .mockdata import (create_grid, get_weights, write_file)

def write_ncfiles(path, suffix='.nc'):
    import pandas as pd
    dates = pd.date_range(datetime.date.today(), periods=10, freq='1D')
    for d in dates:
        fname = 'test_{}Z{}'.format(d.strftime("%Y%m%d"), suffix)
        write_file(Path(path) / fname, ('t_2m', 'pres_sfc'), 24, firststep=d, dt='1H')

@pytest.fixture(scope='function')
def mock_client():
    from dask.distributed import Client
    yield Client()

@pytest.fixture(scope='session')
def mock_timedir():
    with TemporaryDirectory() as td:
        write_ncfiles(td)
        yield td

@pytest.fixture(scope='module')
def mock_vardir():
    today = datetime.date.today()
    vars = ('t_2m', 'pres_sfc', 'rain_gsp_rate')
    with TemporaryDirectory() as td:
        for v in vars:
            fname = 'test_{}_{}Z.nc'.format(v, today.strftime("%Y%m%d"))
            write_file(Path(td) / fname, (v, ), 144, firststep=today, dt='10min')
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
    with NamedTemporaryFile(suffix='.nc') as tf:
        get_weights(tf.name)
        yield tf.name

@pytest.fixture(scope='session')
def mock_run(mockgrid, mock_timedir, mockweights, esm_analysis):
    run = esm_analysis.RunDirectory.gen_weights(mockgrid, mock_timedir, prefix='test', model_type='DWD',
            infile=mockweights)
    yield run

@pytest.fixture(scope='session')
def mock_dataset(mock_timedir):
    import xarray as xr
    yield xr.open_mfdataset(str(Path(mock_timedir) / '*.nc'), combine='by_coords')

@pytest.fixture(scope='session')
def mock_tmpdir():
    with TemporaryDirectory() as td:
        yield td

@pytest.fixture(scope='session')
def esm_analysis():
    import esm_analysis
    with TemporaryDirectory() as cache_dir:
        esm_analysis.cacheing._cache_dir = esm_analysis.Reader._cache_dir = Path(cache_dir)
        yield esm_analysis

@pytest.fixture
def spec_hum():
    yield 7.8526e-3


@pytest.fixture
def mixing_r():
    yield 7.9148e-3


@pytest.fixture
def temp_c():
    yield 25.

@pytest.fixture
def rh():
    yield 40

@pytest.fixture
def pres():
    yield 1013.25

def model_config(config=None):
    conf = '''
    title = "This could ba a descriptive title."

    {}
    '''.format(config or '')
    print(conf)
    return conf

@pytest.fixture
def model_setup_with_config():
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
    with NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as f:
            f.write(model_config())
        yield tf.name
