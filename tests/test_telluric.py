"""Tests for lvmqa.metrics.telluric."""

import numpy as np
import pytest

from lvmqa.metrics.telluric import telluric_ew


def test_telluric_ew_keys(sframe_hdul):
    result = telluric_ew(sframe_hdul)
    assert isinstance(result, dict)
    # Check the expected band labels are present
    for band in ("O2_B", "H2O_1", "O2_A", "H2O_2"):
        assert band in result


def test_telluric_ew_structure(sframe_hdul):
    result = telluric_ew(sframe_hdul)
    for band, vals in result.items():
        assert "ew_median" in vals
        assert "ew_std" in vals
        assert "ew_per_fiber" in vals


def test_telluric_ew_band_outside_range():
    """Bands outside the wavelength range should return NaN."""
    import numpy as np
    from astropy.io import fits

    # Very short wavelength range that doesn't cover telluric bands
    n_fibers, n_wave = 5, 100
    wave = np.linspace(3600, 4000, n_wave)
    flux = np.ones((n_fibers, n_wave), dtype=np.float32)
    ivar = np.ones_like(flux)

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=flux, name="FLUX"),
        fits.ImageHDU(data=ivar, name="IVAR"),
        fits.ImageHDU(data=wave.astype(np.float32), name="WAVE"),
    ])
    result = telluric_ew(hdul)
    # All bands should give NaN since they're outside 3600–4000 Å
    for band, vals in result.items():
        assert np.isnan(vals["ew_median"])
