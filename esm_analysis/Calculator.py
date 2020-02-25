"""Collection of methods to calculated atmospheric variables."""

import numpy as np

__all__ = ('calc_rh', 'calc_sathum', 'calc_satpres')


def calc_rh(q, temp, pres, temp_unit='K', pres_unit='hPa'):
    """
    Calculate Realtive Humidity.

    Parameters
    ----------

    q: float, nd-array
        Specific humidity that is taken to calculate the relative humidity

    temp: float, nd-array
        Temperature that is taken to calculate the relative humidity

    pres: float, nd-array
        Pressure that is taken to calculate the relative humidity

    temp_unit: str, optional (default: K)
        Temperature unit (C: Celsius, K: Kelvin)

    pres_unit: str, optional (default: hPa)
        Pressure unit (hPa: ha Pascal, Pa: Pascal)


    Returns
    -------

        Relative Humidity in percent: float/nd-array

    """
    qs = calc_sathum(temp, pres, temp_unit=temp_unit,
                     pres_unit=pres_unit)
    return 100 * q / qs


def calc_sathum(temp, pres, temp_unit='K', pres_unit='hPa'):
    """
    Calculate Saturation Humidity.

    Parameters
    ----------

    temp: float, nd-array
        Temperature that is taken to calculate the sat. humidity

    pres: float, nd-array
        Pressure that is taken to calculate the sat. humidity

    temp_unit: str, optional (default: K)
        Temperature unit (C: Celsius, K: Kelvin)

    pres_unit: str, optional (default: hPa)
        Pressure unit (hPa: ha Pascal, Pa: Pascal)

    Returns
    -------

        Saturation Humidity: float/nd-array

    """
    es = calc_satpres(temp, unit=temp_unit)
    if pres_unit.lower().startswith('p'):
        es *= 100
    r_v = 0.622 * es / (pres - es)
    return r_v / (1 + r_v)


def calc_satpres(temp, unit='K'):
    """
    Calculate saturation presure.

    Parameters
    ----------

    temp: float, nd-array
        Temperature that is taken to calculate the saturation pressure

    unit: str, optional (default: K)
        Temperature unit (C: Celsius, K: Kelvin)

    Returns
    -------

        Saturation Pressure in hPa: float/nd-array

    """
    if unit.lower().startswith('k'):
        add = 273.15
    else:
        add = 0
    return 6.112 * np.exp((17.62 * (temp-add)) / ((temp-add) + 243.12))
