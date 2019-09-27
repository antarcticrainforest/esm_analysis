
import numpy as np
import pytest

def test_calc_rh(rh, spec_hum, mixing_r, temp_c, pres, esm_analysis):

    rel_hum = esm_analysis.calc_rh(spec_hum, temp_c+273.15, pres, percentage=False)
    np.testing.assert_allclose(rel_hum, rh/100., rtol=1e-04)

    rel_hum = esm_analysis.calc_rh(spec_hum, temp_c, pres, temp_unit='C', percentage=True)
    np.testing.assert_allclose(rel_hum,  rh, rtol=1e-04)

    rel_hum = esm_analysis.calc_rh(spec_hum, temp_c, pres*100, temp_unit='C', pres_unit='Pa')
    np.testing.assert_allclose(rel_hum, rh, rtol=1e-04)

def test_pres(temp_c, esm_analysis):

    es = esm_analysis.calc_satpres(temp_c, unit='C')
    np.testing.assert_allclose(es, 31.6, rtol=1e-03)

    es_k = esm_analysis.calc_satpres(temp_c+273.15)
    np.testing.assert_allclose(es_k, es)

def test_sathum(rh, spec_hum, temp_c, pres, esm_analysis):

    qs = 100 * spec_hum / rh

    qs1 = esm_analysis.calc_sathum(temp_c+273.15, pres)
    np.testing.assert_allclose(qs, qs1, rtol=1e-04)

    qs2 = esm_analysis.calc_sathum(temp_c, pres, temp_unit='C')
    np.testing.assert_allclose(qs, qs2, rtol=1e-04)

    qs3 = esm_analysis.calc_sathum(temp_c, pres*100., temp_unit='C', pres_unit='Pa')
    np.testing.assert_allclose(qs, qs3, rtol=1e-04)

