"""Tests for lvmqa.metrics.sky."""

import numpy as np
import pytest

from lvmqa.metrics.sky import (
    sky_brightness,
    sky_continuum_residuals,
    sky_extinction,
    sky_line_residuals,
)


def test_sky_brightness_with_sky_ext(cframe_hdul):
    result = sky_brightness(cframe_hdul)
    assert "sky_median" in result
    assert "sky_std" in result
    assert np.isfinite(result["sky_median"])


def test_sky_brightness_without_sky_ext(sframe_hdul):
    """Should still work using the full FLUX array."""
    result = sky_brightness(sframe_hdul)
    assert np.isfinite(result["sky_median"])


def test_sky_line_residuals_keys(cframe_hdul, sframe_hdul):
    result = sky_line_residuals(cframe_hdul, sframe_hdul)
    for key in ("line_wave", "flux_before", "rms_after", "rms_over_flux"):
        assert key in result


def test_sky_line_residuals_shapes(cframe_hdul, sframe_hdul):
    result = sky_line_residuals(cframe_hdul, sframe_hdul)
    n = result["line_wave"].size
    assert result["flux_before"].shape == (n,)
    assert result["rms_after"].shape == (n,)
    assert result["rms_over_flux"].shape == (n,)


def test_sky_continuum_residuals_keys(cframe_hdul, sframe_hdul):
    result = sky_continuum_residuals(cframe_hdul, sframe_hdul)
    for key in ("cont_flux_before", "cont_rms_after", "cont_rms_over_flux"):
        assert key in result


def test_sky_extinction_returns_dict(cframe_hdul, drpall_row):
    result = sky_extinction(cframe_hdul, drpall_row)
    assert "flux_level" in result
    assert "extinction_delta_mag" in result


def test_sky_extinction_missing_col(cframe_hdul, drpall_row):
    result = sky_extinction(cframe_hdul, drpall_row, reference_mag_col="NONEXISTENT")
    assert np.isnan(result["extinction_delta_mag"])
