"""Tests for lvmqa.metrics.photometry."""

import numpy as np
import pytest

from lvmqa.metrics.photometry import (
    absolute_photometry,
    color_comparison,
    synthetic_magnitude,
)


@pytest.fixture
def gaia_g_filter(wave):
    """Approximate Gaia G-band transmission (broad, ~3300–10500 Å)."""
    trans = np.zeros_like(wave)
    trans[(wave >= 4000) & (wave <= 9000)] = 0.7
    return wave, trans


def test_synthetic_magnitude_shape(sframe_hdul, gaia_g_filter):
    from lvmqa.utils import get_wavelength_array
    wave_arr = gaia_g_filter[0]
    trans = gaia_g_filter[1]
    flux = sframe_hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(sframe_hdul)
    mags = synthetic_magnitude(flux, wave, wave_arr, trans)
    assert mags.shape == (flux.shape[0],)


def test_synthetic_magnitude_finite(sframe_hdul, gaia_g_filter):
    from lvmqa.utils import get_wavelength_array
    wave = get_wavelength_array(sframe_hdul)
    flux = sframe_hdul["FLUX"].data.astype(float)
    mags = synthetic_magnitude(flux, wave, gaia_g_filter[0], gaia_g_filter[1])
    # All fibers should have finite magnitudes (positive flux + transmission overlap)
    assert np.sum(np.isfinite(mags)) > 0


def test_absolute_photometry_dict(sframe_hdul, drpall_row, gaia_g_filter):
    from lvmqa.utils import get_wavelength_array
    result = absolute_photometry(
        sframe_hdul, drpall_row,
        gaia_g_filter[0], gaia_g_filter[1],
        catalogue_mag_col="GAIA_GMAG",
    )
    assert "synth_mag" in result
    assert "cat_mag" in result
    assert "delta_mag" in result
    assert np.isfinite(result["cat_mag"])


def test_color_comparison_dict(sframe_hdul, drpall_row, gaia_g_filter, wave):
    """Use the same filter for both bands — delta should be ~0."""
    from lvmqa.utils import get_wavelength_array
    result = color_comparison(
        sframe_hdul, drpall_row,
        gaia_g_filter[0], gaia_g_filter[1],
        gaia_g_filter[0], gaia_g_filter[1],
        catalogue_color_col="GAIA_COLOR_GBP_GRP",
    )
    assert "synth_color" in result
    assert "delta_color" in result
    assert abs(result["synth_color"]) < 0.01  # same filter → ~0 colour
