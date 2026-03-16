"""Tests for lvmqa.metrics.calibration."""

import numpy as np
import pytest

from lvmqa.metrics.calibration import spectral_resolution, wavelength_calibration_accuracy


def test_wavelength_calibration_keys(cframe_hdul):
    result = wavelength_calibration_accuracy(cframe_hdul)
    for key in ("line_wave", "delta_lambda", "mean_delta_lambda", "std_delta_lambda"):
        assert key in result


def test_wavelength_calibration_shapes(cframe_hdul):
    result = wavelength_calibration_accuracy(cframe_hdul)
    n_lines = result["line_wave"].size
    if n_lines:
        assert result["delta_lambda"].shape[0] == n_lines
        assert result["mean_delta_lambda"].shape == (n_lines,)


def test_wavelength_calibration_delta_near_zero(cframe_hdul):
    """For a line we injected at the correct position, the offset should be small."""
    result = wavelength_calibration_accuracy(cframe_hdul, sky_lines=[5577.3])
    if result["mean_delta_lambda"].size:
        delta = result["mean_delta_lambda"][0]
        if np.isfinite(delta):
            assert abs(delta) < 5.0  # allow up to 5 Å shift


def test_spectral_resolution_keys(cframe_hdul):
    result = spectral_resolution(cframe_hdul)
    for key in ("line_wave", "fwhm", "resolving_power", "mean_fwhm", "mean_R"):
        assert key in result


def test_spectral_resolution_positive_fwhm(cframe_hdul):
    result = spectral_resolution(cframe_hdul, sky_lines=[5577.3])
    fwhm = result["fwhm"]
    valid = fwhm[np.isfinite(fwhm)]
    if valid.size:
        assert np.all(valid > 0)
