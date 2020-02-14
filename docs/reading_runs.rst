Reading data files
==================

You will normally access data from a model experiement, which is stored in a
directory containing netcdf files. You can load the **meta-data** which is
associated with such an experiement by calling :func:`RunDirectory`.

.. module:: esm_analysis

.. autofunction:: RunDirectory

Applying :func:`RunDirectory` only loads meta data, that is which model,
who many data files are present and any other meta-data that is important for
the experiment.

.. class:: RunDirectory

    .. attribute: dask_client

        This attributed gives access to the dask distrbuted client that is created
        upon loading the RunDirecotry. The client is used to apply functions, load 
        and regrid data in parallel.

    .. attribute:: run_dir

        The name of the directory that has been loaded

    .. attribute: files

        Apply a given function to the dataset via the dask scheduling client

    .. automethod:: close_client

    .. automethod:: restart_client


    .. attribute: status

        Returns the status of the associated dask worker client

    .. automethod:: remap


Loading the Data
----------------
Creating an instance of the :func:`RunDirecotry` object won't load any data. To get
access to the netcdf data the :func:`load_data` method has to be apply

.. class:: RunDirectory

    .. automethod:: load_data

    .. attribute:: dataset

        xarray dataset that contains the model data

    .. automethod:: remap

    .. automethod:: apply_function

    .. automethod:: gen_weights
