.. esm-analysis documentation master file, created by
   sphinx-quickstart on Wed Sep 25 14:59:56 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Tools for Analysing Earth System Model Data
==================================================

**esm-analysis** is a Python 3 library for accessingand working with output
data from various *Earth System Models*. It has been developed to process 
output data from global storm resolving simulations at the
`Max-Planck-Institute for Meteorology <https://www.mpimet.mpg.de/en/communication/news/news/neue-veroeffentlichung-dyamond-klimamodelle-der-naechsten-generation/?tx_news_pi1%5Bcontroller%5D=News&tx_news_pi1%5Baction%5D=detail&cHash=204d5f33e497caaf8f194b44dda0d0f8>`_.

.. note::
    The source code is available via `GitHub <https://github.com/antarcticrainforest/esm_analysis>`_.

Installation
------------

The crrent master branch can be installed from the GitHub repository::

    pip install git+https://github.com/antarcticrainforest/esm_analysis.git

Contents
--------

.. toctree::
   :maxdepth: 2

   reading_runs
   calculator

.. toctree::
    :caption: Examples:

    demo

.. toctree::
    :caption: Development
    :maxdepth: 1

    changelog

.. seealso::

    `Xarray <http://xarray.pydata.org/en/stable/>`_
    `Dask <https://docs.dask.org/en/latest/>`_
    `Dask-Jobqueue <https://jobqueue.dask.org/en/latest/configuration-setup.html>`_
    `Dask-Cleint <https://distributed.dask.org/en/latest/client.html>`_



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
