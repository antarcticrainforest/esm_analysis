Creating a cluster for distributed processing
=============================================

`esm_analysis` supports creating HPC style clusters for distributed data
processing using dask-mpi. At the moment only clusters created by the slurm
workload manager are supported.

.. module:: esm_analysis

.. autoclass:: MPICluster

    .. automethod:: load

    .. automethod:: slurm

    .. attribute:: job_script

        A representation of the job script that was submitted

    .. attribute:: submit_time

        `datetime.datetime` ojbect representing the time the job script was submitted

    .. attribute:: workdir

        The working directory that was used to submit the job to the cluster

    .. attribute:: job_id

        The Id of the submitted job script



