from tempfile import (NamedTemporaryFile, TemporaryDirectory)

from netCDF4 import Dataset as nc
import pytest
from testpath import assert_isfile

from .mockdata import (create_grid, get_weights, write_file)

def test_clear_cache_dir(esm_analysis):

    _cache_dir =  esm_analysis.cacheing._cache_dir
    from pathlib import Path
    esm_analysis.clear_cache_dir()
    assert len([f for f in _cache_dir.rglob('*.*')]) == 0

def test_write_file():
    import datetime
    from pathlib import Path

    import pandas as pd
    import numpy as np
    dates = pd.date_range(datetime.date.today(), periods=10, freq='1D')
    with TemporaryDirectory() as td:
        for d in dates:
            fname = 'test_{}Z.nc'.format(d.strftime("%Y%m%d"))
            write_file(Path(td) / fname, ('t_2m', 'pres_sfc'), 24, firststep=d, dt='1H')

        ncfiles = [nc(str(p)) for p in Path(td).rglob('*.nc')]
        assert len(ncfiles) == 10
        vars = sorted([list(f.variables.keys()) for f in ncfiles])
        assert len(np.unique(vars)) == 3
        shapes = np.array([f.variables['t_2m'].shape for f in ncfiles])
        assert tuple(np.unique(shapes)) == (24, 512)
        [f.close() for f in ncfiles]

def test_get_weigths():

    with NamedTemporaryFile() as tf:
        get_weights(tf.name)

        with nc(tf.name) as f:
            dims = dict(ncells=512, nv=3, bnds=2, time=1, level=1)
            assert sorted(f.dimensions.keys()) == sorted(dims.keys())
            for dim, size in dims.items():
                assert len(f.dimensions[dim]) == size
            assert ('o3' in f.variables.keys()) == True
