"""
Continuum QA metrics (requirements 1, 5, 7).

1.  Color/jumps in the continuum between spectrograph arms (use CFrames).
5.  Signal-to-noise ratio of the continuum (use SFrames).
7.  Weakest detected continuum level (use SFrames and DAP outputs).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from lvmqa.utils import get_wavelength_array, median_flux_in_window, snr_per_fiber

__all__ = [
    "arm_color_jumps",
    "continuum_snr",
    "weakest_continuum",
]

# Approximate boundary wavelengths between the LVM spectrograph arms (Å).
# Blue arm: ~3600–5900 Å, Red arm: ~5900–7700 Å, NIR arm: ~7700–9900 Å.
_ARM_WINDOWS = {
    "blue": (3600.0, 5900.0),
    "red": (5900.0, 7700.0),
    "nir": (7700.0, 9900.0),
}

# Narrow windows bracketing each arm boundary used to measure colour jumps.
_JUMP_WINDOWS = {
    "blue_red": ((5800.0, 5880.0), (5920.0, 6000.0)),
    "red_nir": ((7650.0, 7730.0), (7770.0, 7850.0)),
}


def arm_color_jumps(hdul: fits.HDUList) -> dict[str, np.ndarray]:
    """Measure the continuum flux ratio across arm boundaries (req. 1).

    For each boundary the median flux in a 80 Å window just *blue-ward* of
    the gap is divided by the median flux just *red-ward*.  A value of 1
    indicates a perfectly smooth join.

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file.  Must contain ``FLUX`` and ``WAVE``
        extensions (or WCS keywords).

    Returns
    -------
    dict
        ``"blue_red"`` and ``"red_nir"`` keys, each mapping to a 1-D array
        of per-fiber flux ratios.
    """
    flux = hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(hdul)

    result: dict[str, np.ndarray] = {}
    for label, (win_blue, win_red) in _JUMP_WINDOWS.items():
        med_blue = median_flux_in_window(flux, wave, *win_blue)
        med_red = median_flux_in_window(flux, wave, *win_red)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(med_red != 0, med_blue / med_red, np.nan)
        result[label] = ratio
    return result


def continuum_snr(
    hdul: fits.HDUList,
    wave_min: float = 5400.0,
    wave_max: float = 5600.0,
) -> np.ndarray:
    """Compute the per-fiber continuum S/N in a fiducial green window (req. 5).

    The default 5400–5600 Å window sits in the middle of the blue arm and is
    free of strong emission lines.

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file with ``FLUX`` and ``IVAR`` extensions.
    wave_min, wave_max:
        Wavelength window in Ångström.

    Returns
    -------
    numpy.ndarray
        Per-fiber mean S/N.
    """
    flux = hdul["FLUX"].data.astype(float)
    ivar = hdul["IVAR"].data.astype(float)
    wave = get_wavelength_array(hdul)
    return snr_per_fiber(flux, ivar, wave, wave_min, wave_max)


def weakest_continuum(
    hdul: fits.HDUList,
    percentile: float = 5.0,
    wave_min: float = 5400.0,
    wave_max: float = 5600.0,
) -> float:
    """Return the *percentile* continuum flux level across all fibers (req. 7).

    This represents the faintest detectable continuum in a given exposure.

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file.
    percentile:
        Percentile of the per-fiber median flux distribution to report
        (default 5th percentile → faintest fibers).
    wave_min, wave_max:
        Wavelength window in Ångström.

    Returns
    -------
    float
        Flux level at the requested percentile.
    """
    flux = hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(hdul)
    med_flux = median_flux_in_window(flux, wave, wave_min, wave_max)
    valid = med_flux[np.isfinite(med_flux)]
    if valid.size == 0:
        return np.nan
    return float(np.nanpercentile(valid, percentile))
