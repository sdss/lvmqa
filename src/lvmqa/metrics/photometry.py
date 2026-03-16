"""
Photometry QA metrics (requirements 2, 3).

2.  Color comparisons with existing datasets (PanSTARRS / Gaia) using drpall
    and SFrames.
3.  Absolute photometry comparison with Gaia using drpall and SFrames.

The synthetic broadband magnitudes are derived by convolving the RSS spectra
with standard filter transmission curves.  The comparison with catalogue values
is then performed using the pre-computed magnitudes stored in the drpall table.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Row, Table

from lvmqa.utils import get_wavelength_array

__all__ = [
    "synthetic_magnitude",
    "color_comparison",
    "absolute_photometry",
]

# Speed of light in Å/s (used for converting f_λ to AB magnitudes)
_C_AA_S = 2.99792458e18


def synthetic_magnitude(
    flux: np.ndarray,
    wave: np.ndarray,
    filter_wave: np.ndarray,
    filter_trans: np.ndarray,
) -> np.ndarray:
    """Compute a synthetic AB magnitude for each fiber via filter convolution.

    The integral is evaluated with the trapezoidal rule:

        m_AB = -2.5 * log10( ∫ f_λ T dλ / ∫ T/λ² dλ ) - 48.6

    where *T* is the filter transmission and *f_λ* is in units of
    erg s⁻¹ cm⁻² Å⁻¹.

    Parameters
    ----------
    flux:
        2-D flux array ``(n_fibers, n_wave)`` in erg s⁻¹ cm⁻² Å⁻¹.
    wave:
        1-D wavelength array in Å with length ``n_wave``.
    filter_wave:
        Wavelength grid of the filter transmission curve (Å).
    filter_trans:
        Filter transmission values (0–1) at ``filter_wave``.

    Returns
    -------
    numpy.ndarray
        AB magnitudes per fiber.  Fibers with zero or negative integrated
        flux are returned as ``NaN``.
    """
    # Interpolate filter onto the spectrum wavelength grid
    trans = np.interp(wave, filter_wave, filter_trans, left=0.0, right=0.0)

    numerator = np.trapezoid(flux * trans[np.newaxis, :], wave, axis=1)
    # Reference integral: ∫ T / λ² dλ  (converts f_λ → f_ν AB zeropoint)
    ref_integral = np.trapezoid(trans / wave**2, wave)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (numerator > 0) & (ref_integral > 0),
            numerator / ref_integral,
            np.nan,
        )
        mag = -2.5 * np.log10(ratio) - 48.6
    return mag


def color_comparison(
    hdul: fits.HDUList,
    drpall_row: Row,
    filter1_wave: np.ndarray,
    filter1_trans: np.ndarray,
    filter2_wave: np.ndarray,
    filter2_trans: np.ndarray,
    catalogue_color_col: str,
) -> dict[str, float]:
    """Compare synthetic colour with catalogue colour (req. 2).

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file.
    drpall_row:
        Row from the drpall table for this exposure.
    filter1_wave, filter1_trans:
        Transmission curve for the *bluer* filter.
    filter2_wave, filter2_trans:
        Transmission curve for the *redder* filter.
    catalogue_color_col:
        Column name in *drpall_row* that holds the catalogue colour.

    Returns
    -------
    dict with keys ``"synth_color"``, ``"cat_color"``, ``"delta_color"``,
    and ``"n_fibers"`` (number of fibers with valid synthetic magnitudes).
    """
    flux = hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(hdul)

    mag1 = synthetic_magnitude(flux, wave, filter1_wave, filter1_trans)
    mag2 = synthetic_magnitude(flux, wave, filter2_wave, filter2_trans)
    synth_color = np.nanmedian(mag1 - mag2)

    cat_color = float(drpall_row[catalogue_color_col]) if catalogue_color_col in drpall_row.colnames else np.nan

    return {
        "synth_color": float(synth_color),
        "cat_color": float(cat_color),
        "delta_color": float(synth_color - cat_color),
        "n_fibers": int(np.sum(np.isfinite(mag1) & np.isfinite(mag2))),
    }


def absolute_photometry(
    hdul: fits.HDUList,
    drpall_row: Row,
    filter_wave: np.ndarray,
    filter_trans: np.ndarray,
    catalogue_mag_col: str,
) -> dict[str, float]:
    """Compare median synthetic magnitude with a Gaia catalogue value (req. 3).

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file.
    drpall_row:
        Row from the drpall table for this exposure.
    filter_wave, filter_trans:
        Transmission curve of the comparison filter.
    catalogue_mag_col:
        Column name in *drpall_row* holding the Gaia catalogue magnitude.

    Returns
    -------
    dict with keys ``"synth_mag"``, ``"cat_mag"``, and ``"delta_mag"``.
    """
    flux = hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(hdul)

    synth_mag = float(np.nanmedian(synthetic_magnitude(flux, wave, filter_wave, filter_trans)))
    cat_mag = float(drpall_row[catalogue_mag_col]) if catalogue_mag_col in drpall_row.colnames else np.nan

    return {
        "synth_mag": synth_mag,
        "cat_mag": cat_mag,
        "delta_mag": synth_mag - cat_mag,
    }
