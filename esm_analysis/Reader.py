"""Collection to read datasets."""
import datetime
import hashlib
import json
import os
import os.path as op
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

try:
    from cdo import Cdo
    cdo = Cdo()
except FileNotFoundError:
    cdo = {}
from dask.distributed import (as_completed, Client)
from distributed.diagnostics.progressbar import (futures_of, is_kernel)

import f90nml
import numpy as np
import pandas as pd
import toml
import tqdm
import xarray as xr


def progress_bar(*futures, **kwargs):
    """
    Connect dask futures to tqdm progressbar.

    The probress_bar method gives you the ability to get some feedback while
    processing data.

    ::

        from dask.distributed import Client
        dask_client = Client()
        futures = dask_client.map(lambda x: x*2, [0, 2, 4, 6])
        progress_bar(futures)
        Progress: 100%|████████████| 4.00/4.00 [00:00<00:00, 487it/s]
        results = dask_client.gather(results)

    Parameters
    ----------

    futures: collection
        collections of (dask, concurrent) futures

    notebook: bool, optional (default: False)
        whether or not to display a progress bar optimized for
        jupyter notebooks

    bar_title: str, optional (default: Progress)
        Title of the progress bar

    kwargs:
        Additional keyword arguments passed to the tqdm object

    """
    notebook = kwargs.pop('notebook', None)
    bar_title = kwargs.pop('label', 'Progress')
    futures = futures_of(futures)
    kwargs.setdefault('total', len(futures))
    kwargs.setdefault('unit', 'it')
    kwargs.setdefault('unit_scale', True)
    kwargs.setdefault('leave', True)
    kwargs.setdefault('desc', '{}: '.format(bar_title))
    if notebook is None:
        notebook = is_kernel()  # often but not always correct assumption

    progress = tqdm.tqdm_notebook if notebook else tqdm.tqdm
    _ = list(progress(as_completed(futures), **kwargs))


class _BaseVariables(dict):

    """Base Class to define Variable Name."""

    _base_variables = {'lon', 'lat', 'time', 'ps', 'psl', 'cosmu0', 'rsdt',
                       'rsut', 'rsutcs', 'rlut', 'rlutcs', 'rsds',
                       'rsdscs', 'rlds', 'rldscs', 'rsus', 'rsuscs',
                       'rlus', 'ts', 'sic', 'sit', 'albedo', 'clt',
                       'prlr', 'prls', 'prcr', 'prcs', 'pr', 'prw', 'cllvi',
                       'clivi', 'qgvi', 'qrvi', 'qsvi', 'hfls', 'hfss',
                       'evspsbl', 'tauu', 'tauv', 'sfcwind', 'uas', 'vas',
                       'tas', 'dew2', 'ptp', 'height', 'height_bnds',
                       'pfull', 'zg', 'rho', 'ta', 'ua', 'va', 'wap', 'hus',
                       'clw', 'cli', 'qg', 'qs', 'cl', 'cli_2', 'qr',
                       'tend_ta', 'tend_qhus', 'tend_ta_dyn', 'tend_qhus_dyn',
                       'tend_ta_phy', 'tend_qhus_phy', 'tend_ta_rlw',
                       'tend_ta_rsw', 'tend_ta_mig', 'tend_qhus_mig',
                       'tend_qclw_mig', 'tend_qcli_mig', 'tend_qqr_mig',
                       'tend_qqs_mig', 'tend_qqg_mig', 'tend_ddt_tend_t',
                       'tend_ddt_tend_qv', 'tend_ddt_tend_qc',
                       'tend_ddt_tend_qi', 'tend_ddt_tend_qr',
                       'tend_ddt_tend_qs', 'tend_ddt_tend_qg',
                       'tend_ta_vdf', 'tend_qhus_vdf',
                       'height', 'comsmu0', 'qi'}
    _translate = zip(_base_variables, _base_variables)

    def __init__(self):
        """The base class is based on ECHAM / MPI-ICON variable name."""
        for var1, var2 in self._translate:
            self.__setattr__(var2, var1)
            self[var2] = var1

    def __getattr__(self, attr):
        return self.get(attr, attr)

    def __getitem__(self, attr):
        return self.__getattr__(attr)


class DWD(_BaseVariables):

    """Variable name Class for DWD version of ICON."""

    #             DWD-Name  , Base-Name
    _translate = (
                    ('pres_sfc', 'ps'),
                    ('pres_msl', 'psl'),
                    ('rain_gsp_rate', 'pr'),
                    ('v', 'va'),
                    ('qv', 'hus'),
                    ('temp', 'ta'),
                    ('u', 'ua'),
                    ('rain_con_rate', 'prcr'),
                    ('snow_con_rate', 'prcs'),
                    ('z_mc', 'zg'),
                    ('snow_gsp_rate', 'prls'),
                    ('shfl_s', 'hfss'),
                    ('lhfl_s', 'hfls'),
                    ('omega', 'wap'),
                    ('sp_10m', 'sfcwind'),
                    ('t_2m', 'tas'),
                    ('tqv_dia', 'prw'),
                    ('pres', 'pfull'),
                    ('tqc_dia', 'cllvi'),
                    ('clct', 'clt'),
                    ('qc', 'clw'),
                    ('qi', 'cli'),
                    ('pres', 'pfull'),
                    ('tqi_dia', 'clivi')
                 )

    def __init__(self):
        super().__init__()


class CMORPH(_BaseVariables):

    """Variable name Class for CMORPH."""

    #             CMPORPH-Name  , Base-Name
    _translate = (('precip', 'pr'),)

    def __init__(self):
        super().__init__()


class MPI(_BaseVariables):

    """Variable name Class for ECHAM / MPI version of ICON."""

    def __init__(self):
        super().__init__()


class GenericModel(dict):

    """Default dummy class - No lookup takes place."""

    def __getattr__(self, attr):
        return self.get(attr, attr)

    def __getitem__(self, attr):
        return self.__getattr__(attr)


ECHAM = MPI


def lookup(setup):
    """
    Create a variable translator.

    This methods creats a variable translator object based on a given
    input setup.

    Parameters
    ----------

    setup : str
        The name of the input model

    Returns
    -------
        Translator Object : esm_analysis.Reader._BaseVariables

    """
    if setup is None:
        return GenericModel()
    try:
        LookupObj = getattr(sys.modules[__name__], setup)
    except AttributeError:
        raise KeyError('Model output type not found')
    return LookupObj()


__all__ = ('RunDirectory', 'lookup', 'Config', 'cdo',
           'icon2datetime', 'progress_bar')


def icon2datetime(icon_dates):
    """
    Convert datetime objects in icon format to python datetime objects.

    ::

        time = icon2datetime([20011201.5])

    Parameters
    ----------

    icon_dates: collection
        Collection of icon date dests

    Returns
    -------

        dates:  pd.DatetimeIndex
    """
    try:
        icon_dates = icon_dates.values
    except AttributeError:
        pass

    try:
        icon_dates = icon_dates[:]
    except TypeError:
        icon_dates = np.array([icon_dates])

    def _convert(icon_date):
        frac_day, date = np.modf(icon_date)
        frac_day *= 60**2 * 24
        date = str(int(date))
        date_str = datetime.datetime.strptime(date, '%Y%m%d')
        td = datetime.timedelta(seconds=int(frac_day.round(0)))
        return date_str + td

    conv = np.vectorize(_convert)
    try:
        out = conv(icon_dates)
    except TypeError:
        out = icon_dates
    if len(out) == 1:
        return pd.DatetimeIndex(out)[0]
    return pd.DatetimeIndex(out)


class Config:

    """Configuration Object to save model setups."""

    def __init__(self, toml_config_file):
        """
        Load a configuration.

        ::
            model_setup = Config('model_setup.toml')

        Read a toml configuration file and save the config for access into
        a pandas dataframe. The configuration in the toml file must be saved
        under the 'config' key.

        Parameters
        ----------

        toml_config_file : str
                          File name of the toml configuration file.
                          The configuration must be saved under the 'config'
                          key in the toml file.
        Returns
        -------

            Data Frame of containing model stups: pandas.core.frame.DataFrame

        """
        self._config = toml.load(toml_config_file)
        try:
            self._table = pd.DataFrame(self._config['config'].values(),
                                       index=self._config['config'].keys(),
                                       columns=['Description'])
        except KeyError:
            self._table = pd.DataFrame({})

    def __repr__(self):
        """Return a pandas dataframe."""
        return self._table.__repr__()

    def _repr_html_(self):
        return self._table.style._repr_html_()

    @property
    def setup(self):
        """Get the model setup."""
        return self._table


_cache_dir = (Path('~')/'.cache'/'esm_analysis').expanduser()
_cache_dir.mkdir(parents=True, exist_ok=True)


class RunDirectory:

    """Open data in experiment folder."""

    weightfile = None
    griddes = None

    def __enter__(self):
        """
        Create enter method.

        The enter method just returns the object it self. It is used
        to work along the with __exit__ method that closes a distributed
        worker.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the distributed client befor exiting."""
        self.close_client()

    def __init__(self,
                 run_dir, *,
                 prefix=None,
                 model_type=None,
                 overwrite=False,
                 f90name_list=None,
                 filetype='nc',
                 client=None):
        """
        Create an RunDirecotry object from a given input directory.

        ::

            run = RunDirectory('/work/mh0066/precip-project/3-hourly/CMORPH')

        The RunDirectory object gathers all nesseccary information on the
        data that is stored in the run directory. Once loaded the most
        important meta data will be stored in the run directory for faster
        access the second time.

        Parameters
        ----------
        run_dir: str
            Name of the directory where the data that should be read is stored.

        prefix: str, optional (default: None)
            filname prefix
        model_type: str, optional (default: None)
            model name/ observation porduct that created the data. This will
            be used to generate a variable lookup table. This can be useful
            for loading various model datasets and comparing them while only
            accessing the data with one set of variable names. By default
            no lookupt table will be generated.
        overwrite: bool, optional (default : False)
            If true the meta data will be generated again even if it has been
            stored to disk already.
        f90name_list: str, optional (default: None)
            Filename to an optional f90 namelist with additional information
            about the data
        filetype: str, optional (default: nc)
            Input data file format
        client: dask.distributed cleint, optional (default: None)
            Configuration that is used the create a dask client which recieves
            tasks for multiproccessing. By default (None) a local client will
            be started.

        """
        if isinstance(client, Client):
            self.dask_client = client
        else:
            self.dask_client = Client(client)
        self.prefix = prefix or ''
        self.variables = lookup(model_type)
        run_dir = op.abspath(str(run_dir))
        nml_file = f90name_list or 'NAMELIST_{}*'.format(prefix)
        info_file = self._hash_file(run_dir)
        if overwrite or not info_file.is_file():
            self.name_list = {}
            for nml_file in Path(run_dir).rglob(nml_file):
                self.name_list = {**self.name_list,
                                  **f90nml.read(str(run_dir / nml_file))}
            self.name_list['output'] = self._get_files(run_dir, filetype)
            self.name_list['weightfile'] = None
            self.name_list['gridfile'] = self.griddes
            self.name_list['run_dir'] = op.abspath(str(run_dir))
            self._dump_json(run_dir)
        else:
            with open(str(info_file), 'r') as f:
                self.name_list = json.load(f)

    @staticmethod
    def _hash_file(run_dir):
        run_dir = op.expanduser(str(run_dir))
        hash_obj = hashlib.md5(op.abspath(run_dir).encode())
        hash_str = str(hash_obj.hexdigest())
        return _cache_dir / Path('run_info_{}.json'.format(hash_str))

    @staticmethod
    def _get_files(run_dir, extensions):
        """Get all netcdf filenames."""
        ext_str = ''.join(['[{}{}]'.format(l.lower(), l.upper())
                          for l in extensions])
        pat = re.compile('^(?!.*restart|.*remap).*{}'.format(ext_str))
        glob_pad = '*.{}'.format(ext_str)
        result = sorted([f.as_posix() for f in Path(run_dir).rglob(glob_pad)
                        if re.match(pat, f.as_posix())])
        return result

    @staticmethod
    def _remap(infile, out_dir=None, griddes=None, weightfile=None,
               method=None, gridfile=None, options=None):
        options = options or '-f nc4'
        if isinstance(infile, (str, Path)):
            infile = Path(infile)
            out_file = str(Path(out_dir) / infile.with_suffix('.nc').name)
        else:
            out_file = None
        with NamedTemporaryFile(dir=out_dir, suffix='.nc') as tf_in:

            if method == 'weighted':
                cdo_str = str(griddes)+','+str(weightfile)
                remap_func = getattr(cdo, 'remap')
            else:
                cdo_str = str(griddes)
                remap_func = getattr(cdo, method)
            if gridfile is not None:
                cdo_str += ' -setgrid,'+str(gridfile)

            if isinstance(infile, xr.DataArray):
                _ = xr.Dataset(
                        data_vars={infile.name: infile}
                        ).to_netcdf(tf_in.name)
                kwargs = dict(returnXArray=infile.name)
                infile = Path(tf_in.name)
            elif isinstance(infile, xr.Dataset):
                _ = infile.to_netcdf(tf_in.name)
                infile = Path(tf_in.name)
                kwargs = dict(returnXDataset=True)
            else:
                kwargs = dict(output=str(out_file), options=options)

            out = remap_func('{} {}'.format(str(cdo_str), str(infile)),
                             **kwargs)
            try:
                return out.compute()
            except AttributeError:
                return out

    @property
    def run_dir(self):
        """Get the name of the experiment path."""
        return Path(self.name_list['run_dir'])

    @property
    def files(self):
        """Return all files that have been opened."""
        return pd.Series(self.name_list['output'])

    @staticmethod
    def apply_function(mappable,
                       collection, *,
                       args=None,
                       client=None,
                       **kwargs):
        """
        Apply function to given collection.

        ::

            result = run.apply_function(lambda d, v: d[v].sum(dim='time'),
                                        run.dataset, args=('temp',))

        Parameters
        ----------

        mappable: method
            method that is applied

        collection: collection
            collection that is distributed in a thread pool

        args:
            additional arguments passed into the method

        client: dask distributed client (default: None)
            worker scheduler client that submits the jobs. If None is given
            a new client is started

        progress: bool (default: True)
            display tqdm progress bar

        **kwargs: optional
            additional keyword arguments controlling the progress bar parameter

        Returns
        -------

            combined output of the thread-pool processes: collection
        """
        client = client or Client()
        args = args or ()
        if isinstance(collection, (xr.DataArray, xr.Dataset)):
            tasks = [(client.scatter(collection), *args)]
        else:
            tasks = [(client.scatter(entry), *args) for entry in collection]
        futures = [client.submit(mappable, *task) for task in tasks]
        progress = kwargs.pop('progress', True)
        if progress is True:
            progress_bar(futures, **kwargs)
        output = client.gather(futures)
        if len(output) == 1:  # Possibly only one job was submitted
            return output[0]
        return output

    def close_client(self):
        """Close the opened dask client."""
        self.dask_client.close()

    def restart_client(self):
        """Restart the opened dask client."""
        self.dask_client.restart()

    @property
    def status(self):
        """Query the status of the dask client."""
        return self.dask_client.status

    def remap(self,
              grid_description,
              inp=None,
              out_dir=None, *,
              method='weighted',
              weightfile=None,
              options='-f nc4',
              grid_file=None):
        """
        Regrid to a different input grid.

        ::

            run.remap('echam_griddes.txt', method='remapbil')

        Parameters
        ----------

        grid_description: str
                          Path to file containing the output grid description
        inp: (collection of) str, xarray.Dataset, xarray.DataArray
                Filenames that are to be remapped.
        out_dir: str (default: None)
                  Directory name for the output
        weight_file: str (default: None)
                     Path to file containing grid weights
        method: str (default: weighted)
                 Remap method that is applyied to the data, can be either
                 weighted (default), bil, con, laf, nn. If weighted is chosen
                 this class should have been instanciated either with a given
                 weightfile or using the gen_weights methods.
        weightfile: str (default: None)
                     File containing the weights for the distance weighted
                     remapping.
        grid_file: str (default: None)
                  file containing the source grid describtion
        options: str (default: -f nc4)
                 additional file options that are passed to cdo

        Returns
        -------

            Collection of output: (str, xarray.DataArray, xarray.Dataset)

        """
        out_dir = out_dir or TemporaryDirectory().name
        Path(out_dir).absolute().mkdir(exist_ok=True, parents=True)
        impl_methods = ('weighted', 'remapbil', 'remapcon', 'remaplaf',
                        'remapnn')
        weightfile = weightfile or self.weightfile
        if method not in impl_methods:
            raise NotImplementedError('Method not available.'
                                      ' Currently implemented'
                                      ' methods are:'
                                      'weighted, remapbil, '
                                      'remapcon, remaplaf, remapnn')
        if weightfile is None and method == 'weighted':
            raise ValueError('No weightfile was given, either choose different'
                             ' remapping method or instanciated the Reader'
                             ' object by providing a weightfile or generate '
                             'a weightfile by calling the gen_weights methods')

        args = (Path(out_dir), grid_description, weightfile, method,
                grid_file, options)
        run_dir = self.name_list['run_dir']
        if inp is None:
            inp = self.files
        elif isinstance(inp, (str, Path)):
            if not Path(inp).is_file():
                inp = sorted([f for f in Path(run_dir).rglob(inp)])
            else:
                inp = (inp, )
        if len(inp) == 0:
            raise FileNotFoundError('No files for remapping found')
        return self.apply_function(self._remap, inp,
                                   args=args,
                                   client=self.dask_client,
                                   label='Remapping')

    def _dump_json(self, run_dir):
        run_dir = op.abspath(str(run_dir))
        info_file = self._hash_file(run_dir)
        name_list = self.name_list
        name_list['run_dir'] = run_dir
        name_list['json_file'] = str(info_file.absolute())
        with open(str(info_file), 'w') as f:
            json.dump(name_list, f, sort_keys=True, indent=4)

    @classmethod
    def gen_weights(cls,
                    griddes,
                    run_dir, *,
                    prefix=None,
                    model_type='ECHAM',
                    infile=None,
                    overwrite=False,
                    client=None):
        """
        Create grid weigths from grid description and instanciate class.

        ::

            run = RunDirectory.gen_weights('echam_grid.txt',
                            '/work/mh0066/precip-project/3-hourly/CMORPH/',
                            infile='griddes.nc')

        Parameters
        ----------

        griddess: str
            filename containing the desired output grid information
        run_dir: str
            path to the experiment directory
        prefix: str
            filename prefix
        model_type: str
            Model/Product name of the dataset to be read
        infile: str
            Path to input file. By default the method looks for appropriate
            inputfiles
        overwrite: bool, optional (default: False)
            should an existing weight file be overwritten

        Returns
        -------

            RunDirectory: RunDirectory object

        """
        try:
            out_file = [f for f in Path(run_dir).absolute().rglob('*2d*.nc')][0]
        except IndexError:
            try:
                out_file = [f for f in Path(run_dir).absolute().rglob('*.nc')][0]
            except IndexError:
                raise FileNotFoundError('Run Directory is empty')

        def get_input(rundir, inp_file):
            for file in (inp_file,
                         op.join(rundir, 'o3_icon_DOM01.nc'),
                         op.join(rundir,      'bc_ozone.nc')):
                if op.isfile(str(file)):
                    return inp_file

        input_file = get_input(run_dir, infile)
        weight_file = op.abspath(op.join(run_dir, 'remapweights.nc'))
        if overwrite or not os.path.isfile(weight_file):
            cmd = '{} -setgrid,{} {}'.format(op.abspath(griddes),
                                             input_file,
                                             out_file)
            weight_file = cdo.gendis(cmd, output=weight_file)
        cls.gridfile = griddes
        cls.weightfile = op.abspath(weight_file)
        return cls(run_dir, prefix=prefix,
                   model_type=model_type,
                   overwrite=overwrite,
                   client=client)

    def load_data(self, filenames=None,
                  **kwargs):
        """
        Open a multifile dataset using xrarray open_mfdataset.

        ::

           dset = run.load_data('*2008*.nc')

        Parameters
        ----------

        filenames: collection/str
            collection of filenames, filename or glob pattern for filenames
            that should be read. Default behavior is reading all dataset files

        **kwargs: optional
            Additional keyword arguments passed to xarray's open_mfdataset

        Returns
        -------

            Xarray (multi-file) dataset: xarray.Dataset

        """
        filenames = self._get_files_from_glob_pattern(filenames) or self.files
        kwargs.setdefault('parallel',  True)
        kwargs.setdefault('combine', 'by_coords')
        return xr.open_mfdataset(filenames, **kwargs)

    def _get_files_from_glob_pattern(self, filenames):
        """Construct filename to read."""
        if isinstance(filenames, (str, Path)):
            ncfiles = [filenames, ]
        elif filenames is None:
            return None
        else:
            ncfiles = list(filenames)
        read_files = []
        for in_file in ncfiles:
            if op.isfile(in_file):
                read_files.append(str(in_file))
            else:
                read_files += [str(f) for f in self.run_dir.rglob(str(in_file))]
        return sorted(read_files)
