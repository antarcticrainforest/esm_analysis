Reading data files
==================

You will normally access data from a model experiement, which is stored in a
directory containing netcdf files. You can load the **meta-data** which is
associated with such an experiement by calling :func: `RunDirectory``.

.. module:: esm_analysis
.. autofunction:: RunDirectory

Applying :func: ``RunDirectory`` only loads meta data, that is which model,
who many data files are present and any other meta-data that is important for
the experiment.

.. class:: RunDirectory
    ..attribute: run_dir

    The name of the directory that has been loaded

    ..attribute: files

    Apply a given function to the dataset via the dask scheduling client

    ..automethod: close

    Close the dask client


Loading the Data
----------------
Creating an instance of the ``RunDirecotry`` object won't load any data. To get
access to the netcdf data the :func: ``load_data`` method has to be apply

.. class:: RunDirectory

    ..automethod: load_data

    Open a multi-file dataset

    ..attribute: dataset

    xarray dataset that contains the model data

    ..automethod: remap

    Regrid the loaded dataset to a different input grid

    All netcdf data files that have been loaded

    ..automethod: apply_function

    Apply a given function to the loaded dataset.
