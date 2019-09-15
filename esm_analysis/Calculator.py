from .calc import (calc_rh)

__all__ = ['rel_humidity']

def rel_humidity(dataset, varname, tstep, **kwargs):
    """Calculate Relative Humidity"""
    try:
        temp = dataset['temp'][tstep, ..., 1, :]
        pres = dataset['pres'][tstep, ..., 1, :]
        q = dataset['qv'][tstep, ..., 1, :]
    except KeyError:
        temp = dataset['ta'][tstep, ..., 1, :]
        pres = dataset['pfull'][tstep, ..., 1, :]
        q = dataset['hus'][tstep, ..., 1, :]
    if kwargs.get('d1', False):
        l = temp.shape[-1]//2
        temp = temp[..., l]
        pres = pres[..., l]
        q = q[..., l]
    return calc_rh(q, temp, pres)
