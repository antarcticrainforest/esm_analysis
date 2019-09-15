from pathlib import Path

import pytest
import numpy as np
from testpath import assert_isfile
from unittest import mock

from esm_analysis import RunDirectory


def test_load_data(mockrun_time):
    run = RunDirectory(mockrun_time, 'test', model_type='DWD')
    assert len(run.files) == 10
    assert run.is_remapped == False
    assert run.weightfile == None
    run.load_data()
    assert (run.variables['tas'] in run.dataset.keys()) == True
    assert run.dataset[run.variables['tas']].shape == (240, 512)
    assert run.name_list['picklefile'] != None

def test_gen_weights(mockrun_var, mockgrid, mockweights):

    run = RunDirectory.gen_weights(mockgrid, mockrun_var, 'test',
            model_type='DWD', infile=mockweights, overwrite=True)
    assert run.name_list['picklefile'] == None
    # Test for essential keys to be pressent
    assert sorted(run.name_list.keys()) == ['gridfile', 'output', 'picklefile', 'remap', 'run_dir', 'weightfile']
    assert run.gridfile == mockgrid
    assert run.weightfile == str(Path(mockrun_var) / 'remapweights.nc')
    run.load_data('*t_2m*.nc')
    assert len(run.dataset.variables.keys()) == 2

def test_is_remapped(mock_run):
    assert mock_run.is_remapped == False

def test_remap(mock_run, mockgrid, mockrun_time):
    mock_run.remap(mockgrid)
    assert mock_run.run_dir == str(Path(mockrun_time) / 'remap_grid')

def test_dataset(mock_run):
    assert mock_run.dataset == {}
    mock_run.load_data()
    assert (mock_run.dataset['lon'].shape[0], mock_run.dataset['lat'].shape[0]) == (64, 4)
    assert len(mock_run.dataset.variables.keys()) == 2 + 2 + 1 #2 variables + lon/lat + time

def test_apply_function(mock_run):
    apply_func = lambda dset, varn: dset[varn].min().values
    max_vals = mock_run.apply_function(apply_func, (mock_run.dataset, mock_run.dataset),
                                       args=('t_2m', ), client='dask')
    assert len(max_vals) == 2
    assert np.allclose(max_vals, 0) == True
