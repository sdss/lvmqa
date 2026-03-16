"""Tests for lvmqa.utils."""

import numpy as np
import pytest
from astropy.io import fits

from lvmqa.utils import (
    get_wavelength_array,
    median_flux_in_window,
    snr_per_fiber,
)


def test_get_wavelength_array_from_wave_ext(sframe_hdul):
    wave = get_wavelength_array(sframe_hdul)
    assert wave.ndim == 1
    assert len(wave) > 0
    # Should be monotonically increasing
    assert np.all(np.diff(wave) > 0)


def test_get_wavelength_array_from_wcs():
    """Fallback to WCS keywords when no WAVE extension is present."""
    hdr = fits.Header()
    hdr["NAXIS1"] = 100
    hdr["CRVAL1"] = 4000.0
    hdr["CDELT1"] = 1.2
    hdr["CRPIX1"] = 1.0
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    wave = get_wavelength_array(hdul)
    assert len(wave) == 100
    assert abs(wave[0] - 4000.0) < 1e-6
    assert abs(wave[1] - wave[0] - 1.2) < 1e-6


def test_median_flux_in_window(wave):
    n_fibers, n_wave = 10, len(wave)
    rng = np.random.default_rng(0)
    flux = rng.uniform(1, 2, (n_fibers, n_wave))
    result = median_flux_in_window(flux, wave, 5000.0, 5200.0)
    assert result.shape == (n_fibers,)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)


def test_median_flux_in_window_out_of_range(wave):
    flux = np.ones((5, len(wave)))
    result = median_flux_in_window(flux, wave, 1000.0, 1100.0)
    assert np.all(np.isnan(result))


def test_snr_per_fiber(wave):
    n_fibers, n_wave = 8, len(wave)
    rng = np.random.default_rng(1)
    flux = rng.uniform(1e-16, 2e-16, (n_fibers, n_wave))
    ivar = np.full((n_fibers, n_wave), 1.0 / (1e-18) ** 2)
    snr = snr_per_fiber(flux, ivar, wave, 5000.0, 5200.0)
    assert snr.shape == (n_fibers,)
    assert np.all(snr > 0)
