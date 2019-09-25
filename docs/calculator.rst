Variable Calculation
====================

.. currentmodule:: esm_analysis


The ``Calculator`` sub module offers some methods to calculated common variables
like relative humidity.

.. autoclass:: Calculator

 .. automethod:: calc_rh

 Calculate relative humidity from given specific humidity, temperature and air pressure.

 .. automethod:: calc_sathum

 Calculate saturation humidity from given temperature and air pressrue

 .. automethod:: calc_satpres

 Calculate saturation pressure at a given temperature


