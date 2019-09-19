
import datetime
from glob import glob
import inspect
import json
import multiprocessing as mp
import os
import os.path as op
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile
import warnings

from cdo import Cdo
import cloudpickle
import dask
from dask.distributed import (Client, progress, utils)
import f90nml
import numpy as np
import pandas as pd
import toml
from xarray import open_mfdataset

class _BaseVariables(dict):
   """Base Class to define Variable Name."""

   _base_variables = {'lon', 'lat', 'time', 'ps', 'psl', 'cosmu0', 'rsdt', 'rsut',
                'rsutcs', 'rlut', 'rlutcs', 'rsds', 'rsdscs', 'rlds', 'rldscs',
                'rsus', 'rsuscs', 'rlus', 'ts', 'sic', 'sit', 'albedo', 'clt',
                'prlr', 'prls', 'prcr', 'prcs', 'pr', 'prw', 'cllvi', 'clivi',
                'qgvi', 'qrvi', 'qsvi', 'hfls', 'hfss', 'evspsbl', 'tauu',
                'tauv', 'sfcwind', 'uas', 'vas', 'tas', 'dew2', 'ptp',
                'height', 'height_bnds', 'pfull', 'zg', 'rho', 'ta', 'ua',
                'va', 'wap', 'hus', 'clw', 'cli', 'qg', 'qs', 'cl', 'cli_2', 'qr',
                'tend_ta', 'tend_qhus', 'tend_ta_dyn', 'tend_qhus_dyn',
                'tend_ta_phy', 'tend_qhus_phy', 'tend_ta_rlw', 'tend_ta_rsw',
                'tend_ta_mig', 'tend_qhus_mig', 'tend_qclw_mig', 'tend_qcli_mig',
                'tend_qqr_mig', 'tend_qqs_mig', 'tend_qqg_mig', 'tend_ddt_tend_t',
                'tend_ddt_tend_qv', 'tend_ddt_tend_qc','tend_ddt_tend_qi',
                'tend_ddt_tend_qr', 'tend_ddt_tend_qs', 'tend_ddt_tend_qg',
                'tend_ta_vdf', 'tend_qhus_vdf', 'height', 'comsmu0', 'qi'}
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
   _translate = (('pres_sfc', 'ps'),
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
                ('tqv_dia','prw'),
                ('pres', 'pfull'),
                ('tqc_dia','cllvi'),
                ('clct', 'clt'),
                ('qc', 'clw'),
                ('qi', 'cli'),
                ('pres', 'pfull'),
                ('tqi_dia','clivi'))

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



cdo = Cdo()
ECHAM = MPI

def lookup(setup):
   if setup is None:
       return GenericModel()
   try:
      LookupObj = getattr(sys.modules[__name__], setup)
   except AttributeError:
      raise KeyError('Model output type not found')
   return LookupObj()

__all__ = ['RunDirectory', 'lookup', 'Config', 'cdo', 'icon2datetime']


def icon2datetime(icon_dates, start=None):
    """Convert datetime objects in icon format to python datetime objects.

    Parameters
    ==========
    icon_dates : (collection)
    Collection of icon date dests

    Returns
    =======
    collection of datetime objects
    """

    def convert(icon_date):
        frac_day, date = np.modf(icon_date)
        frac_day *= 60**2 * 24
        return datetime.datetime.strptime(str(int(date)), '%Y%m%d')\
                + datetime.timedelta(seconds=int(frac_day.round(0)))
    conv = np.vectorize(convert)
    return conv(icon_dates)


class Config:
   def __init__(self, toml_config_file):

      self._config = toml.load(toml_config_file)
      try:
         self._table = pd.DataFrame(self._config['config'].values(),
                                    index=self._config['config'].keys(),
                                    columns=['Description'])
      except KeyError:
         self._table = {}

   def __repr__(self):
        a = self._table.__repr__()
        return a

   def _repr_html_(self):
        try:
           a = self._table.style
        except AttributeError:
           return
        return a._repr_html_()

   @property
   def setup(self):
      return self._table

   @property
   def content(self):
      return self._config


class RunDirectory:

    weightfile = None
    griddes = None
    _open_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self,
                 run_dir, *,
                 prefix=None,
                 model_type=None,
                 overwrite=False,
                 f90name_list=None,
                 client=None):
        '''Create an RunDirecotry object from a given input directory.

        The RunDirectory object gathers all nesseccary information on the
        data that is stored in the run directory. Once loaded the most
        important meta data will be stored in the run directory for faster
        access the second time.

        Parameters:
        ===========
        run_dir : str
            Name of the directory where the data that should be read is stored.

        prefix : str (default: None)
            filname prefix
        model_type : str (default: None)
            model name/ observation porduct that created the data. This will
            be used to generate a variable lookup table. If None is given
            (default) then no lookup table will be generated. This can be useful
            for loading various model datasets and comparing them while only
            accessing the data with one set of variable names.
        overwrite : bool (default : False)
            If true the meta data will be generated again even if it has been
            stored to disk already.
        f90name_list : str (default : None)
            Filename to an optional f90 namelist with additional information
            about the data
        client : dask.distributed cleint (default : None)
            Configuration that is used the create a dask client which recieves
            tasks for multiproccessing. By default (None) a local client will
            be started.
        '''

        self.dask_client = Client(client)
        self.prefix = prefix or ''
        self.variables = lookup(model_type)
        info_file = op.join(str(run_dir), '.run_info.json')
        nml_file = f90name_list or 'NAMELIST_{}*'.format(prefix)
        if overwrite or not op.isfile(info_file):
            self.name_list = {}
            for nml_file in Path(run_dir).rglob(nml_file):
                try:
                    self.name_list = {**self.name_list,
                                      **f90nml.read(str(run_dir/ nml_file))}
                except (FileNotFoundError, IsADirectoryError):
                    pass
            self.name_list['output'] = self._get_files(run_dir)
            self.name_list['weightfile'] = None
            self.name_list['gridfile'] = self.griddes
            self.name_list['run_dir'] = op.abspath(run_dir)
            self.name_list['remap'] = False
            self.name_list['picklefile'] = None
            self._dump_json()
        else:
            with open(str(info_file), 'r') as f:
                self.name_list = json.load(f)
        # Sanity check
        if self.name_list['remap']:
           self.name_list['output'] = self._get_files(run_dir, remap=True)
           self.name_list['run_dir'] = run_dir
           self._dump_json()
        self._dataset = {}

    @staticmethod
    def _get_files(run_dir, remap=False):
       """Get all netcdf filenames."""
       if remap:
          pat = re.compile('^(?!.*restart).*[nN][cC]')
       else:
          pat = re.compile('^(?!.*restart|.*remap).*[nN][cC]')
       glob_pad = '*.[nN][cC]'
       result = sorted([f.as_posix() for f in Path(run_dir).rglob(glob_pad) \
                             if re.match(pat, f.as_posix())])
       return result

    @staticmethod
    def _remap(infile, out_dir=None, griddes=None, weightfile=None, method=None):
        out_file = op.join(out_dir, op.basename(infile))
        if method == 'weighted':
           cdo_str = str(griddes)+','+str(weightfile)
           return cdo.remap('{} {}'.format(cdo_str, str(infile)),
                                           output=str(out_file))
        else:
           remap_func = getattr(cdo, method)
           return remap_func('{} {}'.format(griddes, str(infile)), output=str(out_file))

    @property
    def run_dir(self):
        return self.name_list['run_dir']

    @property
    def files(self):
        return pd.Series(self.name_list['output'])

    def apply_function(self,
                       mappable,
                       collection, *,
                       args=None,
                       n_workers=None,
                       bar_title=None,
                       backend='dask',
                       bar_kwargs={}):
       """Apply function to given collection.

       Parameters:
       ==========
       mappable : method
       method that is applied

       collection : collection
       collection that is distributed in a thread pool

       args : additional arguments passed into the method
       n_workers : int
       Number of parallel proccess that are applied

       bar_title : str
       Title of the  progress bar

       bar_kwargs : dict
       dict controlling the progress bar parameter

       Returns: list / int
       combined output of the thread-pool processes or if processes failed 
       an integer of 257 status
       """
       n_workers = n_workers or mp.cpu_count()
       args = args or ()
       bar_title = bar_title or 'Progress'
       tasks = [(entry, *args) for entry in collection]
       bar_kwargs.setdefault('unit', 'it')
       bar_kwargs.setdefault('unit_scale', True)
       bar_kwargs.setdefault('leave', True)
       bar_kwargs.setdefault('desc', '{}: '.format(bar_title))
       n_workers  = min(n_workers, len(tasks))
       with dask.config.set(get=self.dask_client.get):
            futures = [self.dask_client.submit(mappable, *task) for task in tasks]
            progress(futures)
       status = 0
       output = []
       for future in futures:
          try:
             output.append(future.result())
          except Exception as e:
             print('There was an error: {}'.format(e))
             status = 1
       if status != 0:
           output = 257
       return output

    def close(self):
       """Close the opened dask client."""
       try:
           self.dask_client.close()
       except AttributeError:
           pass


    def remap(self,
              grid_description, *,
              n_workers=None,
              out_dir=None,
              files=None,
              method='weighted',
              weightfile=None
              ):
        """Regrid to a different input grid.

        Parameters:
        ===========
        grid_description : str
                          Path to file containing the output grid description

        n_workers : int (default: num of cpu's)
                    Number of parallel processes that remap input files

        weight_file : str (default: None)
                     Path to file containing grid weights

        out_dir : str (default: lonlat)
                  Directory name for the output
        files : str, collection (default : all datafiles)
                Filenames that are to be remapped.
        method : str (default : weighted)
                 Remap method that is applyied to the data, can be either
                 weighted (default), bil, con, laf. Not if weighted is chosen
                 this class should have been instanciated either with a given
                 weightfile or using the gen_weights methods.
        weightfile : str (default : None)
                     File containing the weights for the distance weighted
                     remapping.

        """
        n_workers = n_workers or mp.cpu_count()
        out_dir = out_dir or Path(self.run_dir) / 'remap_grid'
        Path(out_dir).absolute().mkdir(exist_ok=True)
        impl_methods = ('weighted', 'remapbil','remapcon', 'remaplaf')
        weightfile = weightfile or self.weightfile
        if method not in impl_methods:
           raise NotImplementedError('Method not available. Currently implemented'
                                     ' methods are weighted, remapbil, remapcon, reamplaf')
        if weightfile is None and method == 'weighted':
           raise ValueError('No weightfile was given, either choose different'
                            ' remapping method or instanciated the Reader object'
                            ' by providing a weightfile or generate a weightfile'
                            ' by calling the gen_weights methods')

        args = (str(out_dir), str(grid_description), str(weightfile), method)
        run_dir = self.name_list['run_dir']
        if files is None:
            files = self.files
        elif isinstance(files, (str, Path)):
           if not Path(files).is_file():
               files = sorted([f.as_posix() for f in Path(run_dir).rglob(files)])
           else:
               files = (files, )
        if len(files) == 0:
            raise FileNotFoundError('No files for remapping found')
        grid_files = self.apply_function(self._remap, files,
                                         args=args,
                                         n_workers=n_workers,
                                         bar_title='Remapping')

        if not isinstance(grid_files, int):
            self.name_list['output'] = sorted(grid_files)
            self.name_list['remap'] = True
            self._dump_json(out_dir)

    def _dump_json(self, run_dir=None):
        run_dir = run_dir or Path(self.run_dir)
        info_file = Path(run_dir) / '.run_info.json'
        name_list = self.name_list
        name_list['run_dir'] = str(run_dir)
        with open(str(info_file), 'w') as f:
            json.dump(name_list, f, sort_keys=True, indent=4)

    @classmethod
    def gen_weights(cls,
                    griddes,
                    run_dir,* ,
                    prefix=None,
                    model_type='ECHAM',
                    infile=None,
                    overwrite=False,
                    client=None):
        """Create grid weigths from given grid description and instanciate class.

        Parameters:
        ===========
        griddess : str
            filename containing the desired output grid information
        run_dir : str
            path to the experiment directory
        prefix  : str
            filename prefix
        model_type : str
            Model/Product name of the dataset to be read
        infile : str
            Path to input file. By default the method looks for appropriate
            inputfiles
        overwrite : bool (default: False)
            should an existing weight file be overwritten

        Retruns:
        ========
        RunDirectory object
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
            weight_file = cdo.gendis('{} -setgrid,{} {}'.format(op.abspath(griddes),
                                                            input_file,
                                                            out_file),
                                     output=weight_file)
        cls.gridfile = griddes
        cls.weightfile = op.abspath(weight_file)
        return cls(run_dir, prefix=prefix,
                   model_type=model_type,
                   overwrite=overwrite,
                   client=client)

    def load_data(self, filenames=None,
                  overwrite=False,
                  **kwargs):
       """Open a multifile dataset.

        Parameters:
        ==========
        filenames : collection/str
        collection of filenames, filename or glob pattern for filenames
        that should be read. Default behavior is reading all dataset files

        overwrite : bool (default : False)
        For faster access datasets are serialized and stored in pickle files
        after beeing loaded, this drastically speeds up read datasets
        consecutive times. If overwrite is set True the data will be read
        from netcdf-files regardless.

        **kwargs : Additional keyword arguments passed to xarray's open_mfdataset
       """

       if overwrite or self.name_list['picklefile'] is None:
          self._load_data(filenames, kwargs)
       try:
          with open(str(self.name_list['picklefile']), 'rb') as f:
             self._dataset = cloudpickle.load(f)
       except :
          self._load_data(filenames, kwargs)


    @property
    def is_remapped(self):
        return self.name_list['remap']

    def _load_data(self, filenames, kwargs):
        """Load datasets for given filenames."""

        if filenames is None:
           read_files = self.files
        else:
          read_files = self._get_files_from_glob_pattern(filenames)

        self._open_file = read_files

        kwargs.setdefault('parallel',  True)
        kwargs.setdefault('combine', 'by_coords')
        with dask.config.set(get=self.dask_client.get):
            self._dataset = open_mfdataset(read_files, **kwargs)

        self._pickle_dataset()

    def _pickle_dataset(self):
       """Dump a pickle of a open dataset."""
       _ = [f.unlink() for f in Path(self.run_dir).rglob('.*.pkl')]
       with NamedTemporaryFile(dir=self.run_dir,
                               suffix='.pkl',
                               prefix='.', delete=False) as tmpfile:
              cloudpickle.dump(self.dataset, tmpfile, protocol=4)
              self.name_list['picklefile'] = tmpfile.name
       self._dump_json()

    def _get_files_from_glob_pattern(self, filenames):
        """Construct filename to read."""
        if isinstance(filenames, str):
            ncfiles = [filenames,]
        else:
           ncfiles = list(filenames)
        read_files = []

        for in_file in ncfiles:
          read_files += glob(op.join(self.run_dir, op.basename(in_file)))
        return sorted(read_files)

    @property
    def dataset(self):
       return self._dataset
