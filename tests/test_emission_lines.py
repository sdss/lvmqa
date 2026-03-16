"""Tests for lvmqa.metrics.emission_lines."""

import numpy as np
import pytest
from astropy.table import Table

from lvmqa.metrics.emission_lines import (
    emission_line_error_comparison,
    emission_line_rms_over_flux,
    ha_snr,
    weakest_emission_line,
)


def test_ha_snr_shape(sframe_hdul):
    snr = ha_snr(sframe_hdul)
    assert snr.ndim == 1
    assert len(snr) > 0


def test_ha_snr_with_redshift(sframe_hdul):
    snr = ha_snr(sframe_hdul, redshift=0.01)
    assert snr.ndim == 1


def test_weakest_emission_line_basic(dap_table):
    result = weakest_emission_line(dap_table)
    assert "min_flux" in result
    assert "n_detections" in result
    assert result["n_detections"] >= 0


def test_weakest_emission_line_missing_cols():
    t = Table({"A": [1.0, 2.0]})
    result = weakest_emission_line(t)
    assert result["n_detections"] == 0
    assert np.isnan(result["min_flux"])


def test_emission_line_rms_over_flux(dap_table):
    result = emission_line_rms_over_flux(dap_table)
    assert "rms_over_flux" in result
    assert isinstance(result["rms_over_flux"], np.ndarray)


def test_emission_line_rms_over_flux_missing_cols():
    t = Table({"X": [1.0]})
    result = emission_line_rms_over_flux(t)
    assert np.isnan(result["rms_over_flux"]).all()


def test_emission_line_error_comparison_shape(sframe_hdul, dap_table):
    result = emission_line_error_comparison(sframe_hdul, dap_table)
    n = len(dap_table)
    assert result["reported_err"].shape == (n,)
    assert result["residual_std"].shape == (n,)
    assert result["err_ratio"].shape == (n,)


def test_emission_line_error_comparison_missing_cols(sframe_hdul):
    t = Table({"A": [1.0]})
    result = emission_line_error_comparison(sframe_hdul, t)
    assert np.all(np.isnan(result["reported_err"]))
