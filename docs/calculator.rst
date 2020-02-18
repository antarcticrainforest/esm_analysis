Useful methods
==============
 .. currentmodule:: esm_analysis

The ``progress_bar`` method gives you the ability to get some feedback while
processing data. It brings together the functionality of ``tqdm`` and ``dask-distributed``.

 .. autofunction:: progress_bar

 .. autofunction:: icon2datetime




Variable Calculation
====================

.. currentmodule:: esm_analysis


The ``Calculator`` sub module offers some methods to calculated common variables
like relative humidity.

.. class:: Calculator

 .. automethod:: calc_rh

 .. automethod:: calc_sathum

 .. automethod:: calc_satpres


