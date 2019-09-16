import datetime
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import pytest

from .mockdata import (create_grid, get_weights, write_file)

@pytest.fixture(scope='session')
def mockrun_time():
    import pandas as pd
    dates = pd.date_range(datetime.date.today(), periods=10, freq='1D')
    with TemporaryDirectory() as td:
        for d in dates:
            fname = 'test_{}Z.nc'.format(d.strftime("%Y%m%d"))
            write_file(Path(td) / fname, ('t_2m', 'pres_sfc'), 24, firststep=d, dt='1H')
        yield td

@pytest.fixture(scope='module')
def mockrun_var():
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
 
    with NamedTemporaryFile() as tf:
        create_grid(tf.name)
        yield tf.name


@pytest.fixture(scope='session')
def mockweights():
    with NamedTemporaryFile() as tf:
        get_weights(tf.name)
        yield tf.name

@pytest.fixture(scope='session')
def mock_run(mockgrid, mockrun_time, mockweights):
    from esm_analysis import RunDirectory
    run = RunDirectory.gen_weights(mockgrid, mockrun_time, 'test', model_type='DWD',
            infile=mockweights, overwrite=True)
    yield run

@pytest.fixture(scope='session')
def spec_hum():
    yield 7.8526e-3


@pytest.fixture(scope='session')
def mixing_r():
    yield 7.9148e-3


@pytest.fixture(scope='session')
def temp_c():
    yield 25.

@pytest.fixture(scope='session')
def rh():
    yield 40

@pytest.fixture(scope='session')
def pres():
    yield 1013.25
