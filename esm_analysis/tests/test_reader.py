from pathlib import Path

import pytest
import numpy as np
from testpath import assert_isfile
from unittest import mock

from esm_analysis import (Config, icon2datetime, lookup, RunDirectory)

def test_load_empty_data(mock_tmpdir, mockgrid, mockweights):
    with pytest.raises(FileNotFoundError):
        with RunDirectory.gen_weights(mockgrid, mock_tmpdir, prefix='test',
                model_type='DWD', infile=mockweights, overwrite=False):
            pass

def test_load_data(mock_timedir):
    with RunDirectory(mock_timedir, prefix='test', model_type='DWD') as run:
        # At first not data should be loaded only information gathered
        run.empty_cache()
        assert run.name_list['picklefile'] == None
        assert run.dataset == {}
        assert len(run.files) == 10
        assert run.is_remapped == False
        assert run.weightfile == None
        # Now load the data
        run.load_data()
        assert (run.variables['tas'] in run.dataset.keys()) == True
        assert run.dataset[run.variables['tas']].shape == (240, 512)
        assert run.name_list['picklefile'] != None
        run.empty_cache()
        assert run.name_list['picklefile'] == None


def test_gen_weights(mock_vardir, mockgrid, mockweights):
    with RunDirectory.gen_weights(mockgrid, mock_vardir, prefix='test',
            infile=mockweights, overwrite=True) as run:
        # At first not data should be loaded only information gathered
        assert run.dataset == {}
        assert run.name_list['picklefile'] == None
        # Test for essential keys to be pressent
        assert sorted(run.name_list.keys()) == sorted(['gridfile',
                                                       'output',
                                                       'picklefile',
                                                       'remap',
                                                       'run_dir',
                                                       'weightfile',
                                                       'json_file'])
        assert run.gridfile == mockgrid
        assert run.weightfile == str(Path(mock_vardir) / 'remapweights.nc')
        run.load_data('*t_2m*.nc')
        assert len(run.dataset.variables.keys()) == 2


def test_is_remapped(mock_run):
    assert mock_run.is_remapped == False

def test_weighted_remap(mock_run, mockgrid, mock_timedir):
    mock_run.load_data()
    mock_run.remap(mockgrid)
    assert mock_run.is_remapped == True
    # Loading the a second time without the overwrite kwargs should have no effect
    mock_run.load_data()
    assert mock_run.dataset['t_2m'].shape == (240, 512)

def test_other_remap(mock_timedir, mockgrid, mock_tmpdir):
    from pathlib import Path
    with RunDirectory(Path(mock_timedir) / 'remap_grid') as run:
        # Try giving a list for loading the data
        assert run.is_remapped == True
        run.load_data(run.files)
        run.weightfile = None
        # Now try to remap the data into another folder
        # First try a wrong method name
        with pytest.raises(NotImplementedError):
            run.remap(mockgrid, method='remapconr')

        # Now try testing without a valid weightfile
        with pytest.raises(ValueError):
            run.remap(mockgrid)
        print(run.files[0])
        # Try remapping with an invalid file
        with pytest.raises(FileNotFoundError):
            run.remap(mockgrid, method='remapbil', out_dir=mock_tmpdir, files='*.Z.nc')
        # Finally try a valid remapping
        run.remap(mockgrid, method='remapbil', out_dir=mock_tmpdir, files='*Z.nc')
        # Update of the data directory sould only be done after reloading the dataset
        assert run.run_dir == mock_tmpdir
        files = [p for p in Path(mock_tmpdir).rglob('*Z.nc')]
        assert len(files) == 10

def test_reload(mock_tmpdir, mockgrid):
    with RunDirectory(mock_tmpdir) as run:
        assert run.run_dir == mock_tmpdir
        # reload the dataset (do not use the picklefile)
        run.load_data(['*.nc'], overwrite=True)
        run.remap(mockgrid, method='remapcon', files=run.files[0])
        assert len(run.files) == 1


def test_dataset(mock_run, mock_timedir):
    mock_run.load_data(overwrite=True)
    assert (mock_run.dataset['lon'].shape[0], mock_run.dataset['lat'].shape[0]) == (64, 4)
    assert mock_run.run_dir == str(Path(mock_timedir) / 'remap_grid')
    assert len(mock_run.dataset.keys()) ==   1 + 1 #1 variables  + time

def test_apply_function(mock_run):
    apply_func = lambda dset, varn: dset[varn].min().values
    max_vals = mock_run.apply_function(apply_func, (mock_run.dataset, mock_run.dataset),
                                       args=('t_2m', ))
    assert len(max_vals) == 2
    assert np.allclose(max_vals, 0) == True
    with pytest.raises(KeyError):
        failed_out = mock_run.apply_function(apply_func, (mock_run.dataset, mock_run.dataset),
                                         args=('blabla', ))

def test_lookup():
    from esm_analysis.Reader import MPI, CMORPH
    echam_setup = lookup('MPI')
    assert type(echam_setup) == MPI
    assert list(echam_setup.values()) == list(echam_setup.keys())

    cmorph_setup = lookup('CMORPH')
    assert cmorph_setup['pr'] == 'precip'
    # Smoke test
    assert cmorph_setup['ta'] == 'ta'

    with pytest.raises(KeyError):
        lookup('BALBLA')

def test_icon2datetime():

    timesteps = [20091027.5, 20101027.5] # 27th October 2009/2010 12 Noon
    ts_conv = icon2datetime(timesteps)
    assert (ts_conv[0].year, ts_conv[1].month, ts_conv[0].day) == (2009, 10, 27)
    assert ts_conv[1].hour == ts_conv[0].hour == 12

    ts_conv_int = icon2datetime(timesteps[0])
    assert ts_conv_int == ts_conv[0]


def test_config(model_setup_with_config, model_setup_without_config):
    import pandas
    conf_with = Config(model_setup_with_config)
    assert type(conf_with.setup) == pandas.DataFrame

    conf_without = Config(model_setup_without_config)
    assert type(conf_without.setup) == type(conf_without.content) == dict



