
import numpy as np
import pytest

def test_calc_rh(rh, spec_hum, mixing_r, temp_c, pres):

    from esm_analysis.calc import calc_rh
    rel_hum = calc_rh(spec_hum, temp_c+273.15, pres, percentage=False)
    assert np.allclose(rel_hum, rh/100., rtol=1e-04) == True

    rel_hum = calc_rh(spec_hum, temp_c, pres, temp_unit='C', percentage=True)
    assert np.allclose(rel_hum,  rh, rtol=1e-04) == True

    rel_hum = calc_rh(spec_hum, temp_c, pres*100, temp_unit='C', pres_unit='Pa')
    assert np.allclose(rel_hum, rh, rtol=1e-04) == True

def test_pres(temp_c):

    from esm_analysis.calc import calc_satpres
    es = calc_satpres(temp_c, unit='C')
    assert np.allclose(es, 31.6, rtol=1e-03)

    es_k = calc_satpres(temp_c+273.15)
    assert es_k == es

def test_sathum(rh, spec_hum, temp_c, pres):

    from esm_analysis.calc import calc_sathum

    qs = 100 * spec_hum / rh

    qs1 = calc_sathum(temp_c+273.15, pres)
    assert np.allclose(qs, qs1, rtol=1e-04) == True

    qs2 = calc_sathum(temp_c, pres, temp_unit='C')
    assert np.allclose(qs, qs2, rtol=1e-04) == True

    qs3 = calc_sathum(temp_c, pres*100., temp_unit='C', pres_unit='Pa')
    assert np.allclose(qs, qs3, rtol=1e-04) == True

