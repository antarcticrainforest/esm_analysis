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

    .attribute: dask_client

    This attributed gives access to the dask distrbuted client that is created
    upon loading the RunDirecotry. The client is used to apply functions, load 
    and regrid data in parallel.

    ..attribute: run_dir

    The name of the directory that has been loaded

    ..attribute: files

    Apply a given function to the dataset via the dask scheduling client

    ..automethod: close

    Close the dask client

    ..attribute: is_remapped

    Retruns true or fals whether or not the dataset has been remapped by the the
    remapped method

    ..automethod: remap


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

    Regrid the files or a subset of it. The user can chose the output directory
    and the remapping method. Currently distance weighted, nearest neighbor,
    bilinear, conservative and largest area fraction are implemented


    ..automethod: apply_function

    Apply a given input function to a collection of input streams. The function
    will be applied in parallel using the dask distributed client.

    ..automethod: gen_weights

    Create weights file that can be later used for distance weighted regridding.



