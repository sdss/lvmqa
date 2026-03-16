"""
Wavelength calibration and spectral resolution QA (requirements 13, 14).

13.  Accuracy of the wavelength calibration: Δλ from night-sky lines, fiber by
     fiber and on average, for a set of prominent emission lines covering the
     full wavelength range (use CFrames and DAP outputs).
14.  Accuracy of the LOSVD / spectral resolution measured in night-sky lines
     (use CFrames and DAP outputs).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

from lvmqa.utils import get_wavelength_array

__all__ = [
    "wavelength_calibration_accuracy",
    "spectral_resolution",
]

# Rest-frame wavelengths of sky lines used for calibration checks (Å).
# Selected to span the full spectral range and be isolated / bright.
_CALIB_SKY_LINES: list[float] = [
    5577.3,   # OI
    6300.3,   # OI
    6364.0,   # OI
    7316.0,   # OH
    7750.0,   # OH
    8344.6,   # OH
    8827.1,   # OH
]

# Half-width of the fitting window around each sky line (Å)
_FIT_HALF_WIDTH = 15.0


def _gaussian(x: np.ndarray, amp: float, cen: float, sigma: float, cont: float) -> np.ndarray:
    """1-D Gaussian with a flat continuum offset."""
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2) + cont


def _fit_sky_line(
    wave: np.ndarray, flux_1d: np.ndarray, line_center: float, half_width: float
) -> tuple[float, float]:
    """Fit a Gaussian to a single sky line in a 1-D spectrum.

    Returns
    -------
    (fitted_center, fitted_sigma)
        Both in Ångström.  Returns ``(NaN, NaN)`` on failure.
    """
    mask = (wave >= line_center - half_width) & (wave <= line_center + half_width)
    if np.sum(mask) < 5:
        return np.nan, np.nan

    x = wave[mask]
    y = flux_1d[mask]

    # Initial guesses
    amp0 = float(np.nanmax(y) - np.nanmin(y))
    cen0 = float(x[np.argmax(y)])
    sig0 = 2.0
    cont0 = float(np.nanmin(y))

    try:
        popt, _ = curve_fit(
            _gaussian, x, y,
            p0=[amp0, cen0, sig0, cont0],
            bounds=([0, line_center - half_width, 0.1, -np.inf],
                    [np.inf, line_center + half_width, half_width, np.inf]),
            maxfev=2000,
        )
        return float(popt[1]), float(abs(popt[2]))
    except (RuntimeError, ValueError):
        return np.nan, np.nan


def wavelength_calibration_accuracy(
    hdul: fits.HDUList,
    sky_lines: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Measure Δλ offsets of sky lines per fiber (req. 13).

    For each sky line the Gaussian centroid is fitted in every fiber and the
    offset from the known rest wavelength is computed.

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file.
    sky_lines:
        Rest-frame sky-line wavelengths (Å) to analyse.  Defaults to
        ``_CALIB_SKY_LINES``.

    Returns
    -------
    dict with:
    - ``"line_wave"``        — 1-D array of sky line wavelengths used.
    - ``"delta_lambda"``     — 2-D array ``(n_lines, n_fibers)`` of offsets (Å).
    - ``"mean_delta_lambda"``— 1-D array of per-line mean offset across fibers.
    - ``"std_delta_lambda"`` — 1-D array of per-line scatter across fibers.
    """
    if sky_lines is None:
        sky_lines = _CALIB_SKY_LINES

    wave = get_wavelength_array(hdul)
    flux = hdul["FLUX"].data.astype(float)
    n_fibers = flux.shape[0]

    used_lines: list[float] = []
    delta_lambda_rows: list[np.ndarray] = []

    for lw in sky_lines:
        deltas = np.full(n_fibers, np.nan)
        for f in range(n_fibers):
            cen, _ = _fit_sky_line(wave, flux[f], lw, _FIT_HALF_WIDTH)
            if np.isfinite(cen):
                deltas[f] = cen - lw
        used_lines.append(lw)
        delta_lambda_rows.append(deltas)

    delta_lambda = np.array(delta_lambda_rows)  # shape (n_lines, n_fibers)
    return {
        "line_wave": np.asarray(used_lines),
        "delta_lambda": delta_lambda,
        "mean_delta_lambda": np.nanmean(delta_lambda, axis=1),
        "std_delta_lambda": np.nanstd(delta_lambda, axis=1),
    }


def spectral_resolution(
    hdul: fits.HDUList,
    sky_lines: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Measure the spectral resolution (FWHM) from sky lines per fiber (req. 14).

    The Gaussian sigma of each sky line is converted to FWHM and then to the
    resolving power *R = λ / ΔFWHM*.

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file.
    sky_lines:
        Rest-frame sky-line wavelengths (Å) to analyse.  Defaults to
        ``_CALIB_SKY_LINES``.

    Returns
    -------
    dict with:
    - ``"line_wave"``     — 1-D array of sky line wavelengths.
    - ``"fwhm"``          — 2-D array ``(n_lines, n_fibers)`` in Å.
    - ``"resolving_power"``— 2-D array ``(n_lines, n_fibers)`` R = λ/FWHM.
    - ``"mean_fwhm"``     — 1-D array of per-line mean FWHM.
    - ``"mean_R"``        — 1-D array of per-line mean resolving power.
    """
    if sky_lines is None:
        sky_lines = _CALIB_SKY_LINES

    wave = get_wavelength_array(hdul)
    flux = hdul["FLUX"].data.astype(float)
    n_fibers = flux.shape[0]

    used_lines: list[float] = []
    fwhm_rows: list[np.ndarray] = []

    for lw in sky_lines:
        fwhms = np.full(n_fibers, np.nan)
        for f in range(n_fibers):
            _, sigma = _fit_sky_line(wave, flux[f], lw, _FIT_HALF_WIDTH)
            if np.isfinite(sigma):
                fwhms[f] = 2.3548 * sigma
        used_lines.append(lw)
        fwhm_rows.append(fwhms)

    fwhm_arr = np.array(fwhm_rows)  # (n_lines, n_fibers)
    lw_arr = np.asarray(used_lines)

    with np.errstate(divide="ignore", invalid="ignore"):
        resolving_power = lw_arr[:, np.newaxis] / fwhm_arr

    return {
        "line_wave": lw_arr,
        "fwhm": fwhm_arr,
        "resolving_power": resolving_power,
        "mean_fwhm": np.nanmean(fwhm_arr, axis=1),
        "mean_R": np.nanmean(resolving_power, axis=1),
    }
