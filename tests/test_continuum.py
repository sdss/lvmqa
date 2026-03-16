"""Tests for lvmqa.metrics.continuum."""

import numpy as np
import pytest

from lvmqa.metrics.continuum import arm_color_jumps, continuum_snr, weakest_continuum


def test_arm_color_jumps_shape(cframe_hdul):
    result = arm_color_jumps(cframe_hdul)
    assert isinstance(result, dict)
    assert "blue_red" in result
    assert "red_nir" in result
    # blue-red window is within our synthetic wavelength range
    arr = result["blue_red"]
    assert isinstance(arr, np.ndarray)


def test_arm_color_jumps_ratio_near_one(cframe_hdul):
    """For a flat continuum, the flux ratio should be near 1."""
    result = arm_color_jumps(cframe_hdul)
    valid = result["blue_red"][np.isfinite(result["blue_red"])]
    if valid.size:
        # Allow wide tolerance because we added a sky line
        assert np.nanmedian(valid) == pytest.approx(1.0, abs=0.5)


def test_continuum_snr_shape(sframe_hdul):
    snr = continuum_snr(sframe_hdul)
    assert snr.ndim == 1
    assert len(snr) > 0
    assert np.all(snr[np.isfinite(snr)] > 0)


def test_continuum_snr_custom_window(sframe_hdul):
    snr = continuum_snr(sframe_hdul, wave_min=4000.0, wave_max=4200.0)
    assert snr.ndim == 1


def test_weakest_continuum_returns_scalar(sframe_hdul):
    val = weakest_continuum(sframe_hdul)
    assert np.isfinite(val)


def test_weakest_continuum_percentile(sframe_hdul):
    p5 = weakest_continuum(sframe_hdul, percentile=5.0)
    p50 = weakest_continuum(sframe_hdul, percentile=50.0)
    assert p5 <= p50
