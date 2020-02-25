"""Test the reader module."""
from pathlib import Path

import pytest
import numpy as np

from esm_analysis import (Config, icon2datetime, lookup,
                          RunDirectory, progress_bar)


def test_icon2datetime(mock_dataset):
    """Test icon2datetime."""
    import pandas as pd
    pd_conf = icon2datetime(mock_dataset.time)
    assert isinstance(pd_conf, pd.DatetimeIndex)
    timesteps = [20091027.5, 20101027.5]  # 27th October 2009/2010 12 Noon
    ts = icon2datetime(timesteps)
    assert (ts[0].year, ts[1].month, ts[0].day) == (2009, 10, 27)
    assert ts[1].hour == ts[0].hour == 12

    ts_conv_int = icon2datetime(timesteps[0])
    assert ts_conv_int == ts[0]
    assert isinstance(ts_conv_int, pd.Timestamp) is True


def test_load_empty_data(mock_tmpdir, mockgrid, mockweights):
    """Test loading a dataset with no data."""
    with pytest.raises(FileNotFoundError):
        with RunDirectory.gen_weights(mockgrid, mock_tmpdir, prefix='test',
                                      model_type='DWD',
                                      infile=mockweights, overwrite=False):
            pass


def test_load_data(mock_timedir):
    """Test loading an ordinary dataset."""
    from esm_analysis.Reader import GenericModel
    with RunDirectory(mock_timedir, prefix='test', model_type='DWD') as run:
        # At first not data should be loaded only information gathered
        assert len(run.files) == 10
        assert run.weightfile is None
        assert run.name_list['config']['input'] == 'wind.nc'
        # Now load the data
        run.dataset = run.load_data()
        assert (run.variables['tas'] in run.dataset.keys()) is True
        assert run.dataset[run.variables['tas']].shape == (240, 512)
    with RunDirectory(mock_timedir, f90name_list=('blabla')) as run:
        assert type(run.variables) == GenericModel
        assert run.variables['pr'] == 'pr'
        assert run.variables.pr == 'pr'


def test_gen_weights(mock_vardir, mockgrid, mockweights, mock_client):
    """Test the weight generation."""
    with RunDirectory.gen_weights(mockgrid, mock_vardir, prefix='test',
                                  infile=mockweights, overwrite=True,
                                  client=mock_client) as run:
        # At first not data should be loaded only information gathered
        # Test for essential keys to be pressent
        assert sorted(run.name_list.keys()) == sorted(['gridfile',
                                                       'output',
                                                       'run_dir',
                                                       'weightfile',
                                                       'json_file'])
        assert run.gridfile == mockgrid
        assert run.weightfile == str(Path(mock_vardir) / 'remapweights.nc')
        run.dataset = run.load_data('*t_2m*.nc')
        assert len(run.dataset.variables.keys()) == 2


def test_weighted_remap(mock_run, mockgrid, mock_timedir, mockweights,
                        mock_tmpdir):
    """Test weighted remapping."""
    dataset = mock_run.load_data()
    out_files = mock_run.remap(mockgrid,
                               out_dir=mock_tmpdir,
                               grid_file=mockweights)

    # Loading the a second time without the overwrite kwargs
    # should have no effect
    remap_dataset = mock_run.load_data(out_files)
    assert remap_dataset['t_2m'].shape == (240, 4, 64)
    mock_run.dataset = remap_dataset
    single_remap = mock_run.remap(mockgrid, mock_run.files[0],
                                  out_dir=mock_tmpdir,
                                  grid_file=mockweights)
    single_dset = mock_run.load_data(single_remap)
    assert single_dset['t_2m'].shape == (24, 4, 64)


def test_other_remap(mock_timedir, mockgrid, mock_tmpdir, mock_client):
    """Testing non weighted rmapping."""
    from pathlib import Path
    import xarray as xr
    with RunDirectory(Path(mock_tmpdir), client=mock_client) as run:
        # Try giving a list for loading the data
        dset = run.load_data(run.files)
        run.weightfile = None
        # Now try to remap the data into another folder
        # First try a wrong method name
        with pytest.raises(NotImplementedError):
            run.remap(mockgrid, method='remapconr')

        # Now try testing without a valid weightfile
        with pytest.raises(ValueError):
            run.remap(mockgrid)
        # Try remapping with an invalid file
        with pytest.raises(FileNotFoundError):
            run.remap(mockgrid, '*.Z.nc', mock_tmpdir, method='remapbil')
        # Finally try a valid remapping
        remap_dset = run.remap(mockgrid, dset, method='remapbil')
        # Update of the data directory sould only be done after
        # reloading the dataset
        assert isinstance(remap_dset, xr.Dataset) is True
        remap_dset = run.remap(mockgrid, dset['t_2m'], method='remapbil')
        assert isinstance(remap_dset, xr.DataArray) is True


def test_client(mock_run, mock_client):
    """Test the dask distributed client."""
    mock_run.restart_client()
    assert mock_run.status == 'running'
    mock_run.close_client()
    assert mock_run.status == 'closed'
    mock_run.dask_client = mock_client


def test_progress_bar(mock_client):
    """Testing the progress bar."""
    future = mock_client.submit(lambda: 2*6)
    progress_bar(future)
    assert future.result() == 12


def test_apply_function(mock_run, mock_client):
    """Test the parallel apply function."""
    max_vals = mock_run.apply_function(lambda d, v: d[v].min().values,
                                       (mock_run.dataset, mock_run.dataset),
                                       args=('t_2m', ), client=mock_client)
    assert len(max_vals) == 2
    assert np.allclose(max_vals, 0) is True
    with pytest.raises(KeyError):
        failed_out = mock_run.apply_function(lambda d, v: d[v].min().values,
                                             (mock_run.dataset,
                                              mock_run.dataset),
                                             args=('blabla', ))


def test_lookup():
    """Testing the variable tranlation."""
    from esm_analysis.Reader import MPI, CMORPH
    echam_setup = lookup('MPI')
    assert isinstance(echam_setup,  MPI) is True
    assert list(echam_setup.values()) == list(echam_setup.keys())

    cmorph_setup = lookup('CMORPH')
    assert isinstance(cmorph_setup, CMORPH) is True
    assert cmorph_setup['pr'] == 'precip'
    # Smoke test
    assert cmorph_setup['ta'] == 'ta'

    with pytest.raises(KeyError):
        lookup('BALBLA')


def test_config(model_setup_with_config, model_setup_without_config):
    """Test reading the model configuration."""
    import pandas
    conf_with = Config(model_setup_with_config)
    conf_without = Config(model_setup_without_config)
    assert isinstance(conf_without.setup, pandas.DataFrame) is True
    assert isinstance(conf_with.setup, pandas.DataFrame) is True
    assert len(conf_without.setup) == 0
    assert conf_with.__repr__() == conf_with._table.__repr__()
    assert isinstance(conf_with._repr_html_(), str) is True
