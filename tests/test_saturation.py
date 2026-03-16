"""Tests for lvmqa.metrics.saturation."""

import numpy as np
import pytest

from lvmqa.metrics.saturation import detect_saturated_pixels


def test_no_saturation(cframe_hdul):
    """Standard CFrame has no saturated pixels in bitmask."""
    result = detect_saturated_pixels(cframe_hdul)
    assert result["n_saturated"] == 0
    assert result["frac_saturated"] == 0.0
    assert result["method"] == "bitmask"


def test_bitmask_saturation(cframe_with_saturation):
    result = detect_saturated_pixels(cframe_with_saturation)
    assert result["n_saturated"] == 10
    assert result["n_fibers_affected"] == 1
    assert result["fiber_sat_counts"][0] == 10
    assert result["method"] == "bitmask"


def test_flux_threshold_saturation():
    """Use flux threshold when no MASK extension is present."""
    import numpy as np
    from astropy.io import fits as _fits

    n_fibers, n_wave = 5, 100
    flux = np.ones((n_fibers, n_wave), dtype=np.float32) * 1e-16
    wave = np.linspace(5000, 6000, n_wave).astype(np.float32)

    # No MASK extension
    hdul = _fits.HDUList([
        _fits.PrimaryHDU(),
        _fits.ImageHDU(data=flux, name="FLUX"),
        _fits.ImageHDU(data=wave, name="WAVE"),
    ])
    result = detect_saturated_pixels(hdul, sat_flux_threshold=0.5e-16)
    assert result["method"] == "flux_threshold"
    # All flux values are 1e-16 > 0.5e-16 → all saturated
    assert result["n_saturated"] == n_fibers * n_wave


def test_no_mask_no_threshold():
    """When no MASK and no threshold, return zeros."""
    import numpy as np
    from astropy.io import fits as _fits

    n_fibers, n_wave = 5, 50
    flux = np.ones((n_fibers, n_wave), dtype=np.float32)
    hdul = _fits.HDUList([
        _fits.PrimaryHDU(),
        _fits.ImageHDU(data=flux, name="FLUX"),
    ])
    result = detect_saturated_pixels(hdul, sat_flux_threshold=None)
    assert result["n_saturated"] == 0
    assert result["method"] == "none"
