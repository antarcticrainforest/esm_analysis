import numpy as np

__all__ = ['calc_rh', 'calc_sathum', 'calc_satpres']

def calc_rh(q, temp, pres, temp_unit='K', pres_unit='hPa', percentage=True, mixing_r=False):
    """Calculate Realtive Humidity.

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

    percentage: bool, optional (default: True)
        Return RH in percent

    mixing_r: bool, optional (default: False)
        humidit is mixing ratio instead of specific humidity

    Returns
    -------

        Relative Humidity: float/nd-array
    """
    qs = calc_sathum(temp, pres, temp_unit=temp_unit, pres_unit=pres_unit, mixing_r=mixing_r)
    if percentage is True:
        return 100 * q / qs
    else:
        return q / qs

def calc_sathum(temp, pres, temp_unit='K', pres_unit='hPa', mixing_r=False):
    """Calculate Saturation Humidity.

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

    mixing_r: bool, optional (default: False)
        humidit is mixing ratio instead of specific humidity

    Returns
    -------

        Saturation Humidity: float/nd-array

    """
    es = calc_satpres(temp, unit=temp_unit)
    if pres_unit.lower().startswith('p'):
        es *= 100
    r_v = 0.622 * es / ( pres - es )
    if mixing_r:
        return r_v
    else:
        return r_v / (1 + r_v)

def calc_satpres(temp, unit='K'):
    """Calculate saturation presure.

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
    return 6.112 * np.exp(( 17.62 * ( temp-add ) )/ ( (temp-add) + 243.12))
