"""
Utility helpers shared across all QA modules.

Functions here handle opening the four main file types used by lvmqa:
  * SFrame.fits  – sky-subtracted row-stacked spectra (RSS)
  * CFrame.fits  – non-sky-subtracted row-stacked spectra
  * DAP outputs  – FITS tables with emission-line / stellar-population fits
  * drpall.fits  – summary table of all exposures
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = [
    "load_rss",
    "load_drpall",
    "load_dap_table",
    "get_wavelength_array",
    "median_flux_in_window",
    "snr_per_fiber",
]

PathLike = Union[str, Path]


def load_rss(path: PathLike) -> fits.HDUList:
    """Open an RSS FITS file (SFrame or CFrame) and return the HDUList.

    Parameters
    ----------
    path:
        Filesystem path to the FITS file.

    Returns
    -------
    fits.HDUList
        The open HDUList.  The caller is responsible for closing it or using
        a context manager.
    """
    return fits.open(path, memmap=True)


def load_drpall(path: PathLike) -> Table:
    """Load the drpall summary FITS table.

    Parameters
    ----------
    path:
        Filesystem path to the drpall FITS file.

    Returns
    -------
    astropy.table.Table
    """
    return Table.read(path)


def load_dap_table(path: PathLike, extname: str = 1) -> Table:
    """Load a DAP output FITS table extension.

    Parameters
    ----------
    path:
        Filesystem path to the DAP FITS file.
    extname:
        Extension name or index to read (default ``1``).

    Returns
    -------
    astropy.table.Table
    """
    return Table.read(path, hdu=extname)


def get_wavelength_array(hdul: fits.HDUList, extname: str = "WAVE") -> np.ndarray:
    """Extract the wavelength array from an RSS HDUList.

    The wavelength axis is taken from the ``WAVE`` extension when present;
    otherwise it is reconstructed from the WCS keywords in the primary header.

    Parameters
    ----------
    hdul:
        Open HDUList for an SFrame or CFrame file.
    extname:
        Name of the wavelength extension (default ``"WAVE"``).

    Returns
    -------
    numpy.ndarray
        1-D array of wavelengths in Ångström.
    """
    if extname in hdul:
        return hdul[extname].data.astype(float).ravel()

    # Fall back to WCS reconstruction from primary header
    hdr = hdul[0].header
    naxis = hdr.get("NAXIS1", hdr.get("NAXIS2", 1))
    crval = hdr.get("CRVAL1", 0.0)
    cdelt = hdr.get("CDELT1", hdr.get("CD1_1", 1.0))
    crpix = hdr.get("CRPIX1", 1.0)
    pix = np.arange(naxis)
    return crval + (pix - (crpix - 1)) * cdelt


def median_flux_in_window(
    flux: np.ndarray,
    wave: np.ndarray,
    wave_min: float,
    wave_max: float,
) -> np.ndarray:
    """Compute the median flux per fiber inside a wavelength window.

    Parameters
    ----------
    flux:
        2-D flux array with shape ``(n_fibers, n_wave)``.
    wave:
        1-D wavelength array with length ``n_wave``.
    wave_min, wave_max:
        Wavelength range (inclusive) in Ångström.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n_fibers``.
    """
    mask = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(mask):
        return np.full(flux.shape[0], np.nan)
    return np.nanmedian(flux[:, mask], axis=1)


def snr_per_fiber(
    flux: np.ndarray,
    ivar: np.ndarray,
    wave: np.ndarray,
    wave_min: float,
    wave_max: float,
) -> np.ndarray:
    """Compute the mean S/N per fiber in a wavelength window.

    S/N is computed pixel-by-pixel as ``flux * sqrt(ivar)`` and averaged
    over the selected wavelength range.

    Parameters
    ----------
    flux:
        2-D flux array ``(n_fibers, n_wave)``.
    ivar:
        2-D inverse-variance array ``(n_fibers, n_wave)``.
    wave:
        1-D wavelength array with length ``n_wave``.
    wave_min, wave_max:
        Wavelength range in Ångström.

    Returns
    -------
    numpy.ndarray
        1-D array of mean S/N values, one per fiber.
    """
    mask = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(mask):
        return np.full(flux.shape[0], np.nan)
    snr = flux[:, mask] * np.sqrt(np.clip(ivar[:, mask], 0, None))
    return np.nanmean(snr, axis=1)
