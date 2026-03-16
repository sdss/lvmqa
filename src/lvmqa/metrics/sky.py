"""
Sky-background QA metrics (requirements 8, 9, 10, 12).

8.   Sky brightness during the observations (use CFrames and drpall).
9.   Residual RMS / Flux of night-sky emission lines after sky subtraction
     (use CFrames and drpall).
10.  Residual RMS / Flux of the night-sky continuum (use CFrames and drpall).
12.  Night-sky extinction (use drpall and CFrames).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Row

from lvmqa.utils import get_wavelength_array, median_flux_in_window

__all__ = [
    "sky_brightness",
    "sky_line_residuals",
    "sky_continuum_residuals",
    "sky_extinction",
]

# Prominent night-sky emission lines (Å) used in residual analysis.
# These are a subset of the brightest OH and OI lines across the full range.
_SKY_EMISSION_LINES: list[float] = [
    5577.3,   # OI green line
    6300.3,   # OI red line
    6363.8,   # OI red line (weaker)
    7316.0,   # OH doublet
    8344.6,   # OH band head
    8399.0,   # OH band head
]

# Half-width used to integrate each sky line (Å)
_LINE_HALF_WIDTH = 5.0

# Continuum windows for sky brightness and sky continuum measurements (Å)
_SKY_CONT_WINDOWS = [
    (5540.0, 5570.0),
    (6200.0, 6260.0),
    (7170.0, 7230.0),
]


def sky_brightness(
    hdul: fits.HDUList,
    sky_fibers: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure sky surface brightness from CFrame sky fibers (req. 8).

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file with ``FLUX`` and ``SKY`` extensions.
        If a ``SKY`` extension is present its data are used directly; otherwise
        the full ``FLUX`` array is used (assuming sky-only fibers are selected
        via *sky_fibers*).
    sky_fibers:
        Optional 1-D integer array of fiber indices that are pure sky.  When
        ``None`` and no ``SKY`` extension is found the entire ``FLUX`` array is
        used (not recommended for science frames).

    Returns
    -------
    dict with keys ``"sky_median"`` (median sky flux in the continuum windows,
    in erg s⁻¹ cm⁻² Å⁻¹ arcsec⁻²) and ``"sky_std"`` (scatter across fibers).
    """
    wave = get_wavelength_array(hdul)

    if "SKY" in hdul:
        sky_flux = hdul["SKY"].data.astype(float)
    else:
        flux = hdul["FLUX"].data.astype(float)
        sky_flux = flux[sky_fibers] if sky_fibers is not None else flux

    sky_per_fiber: list[float] = []
    for win_min, win_max in _SKY_CONT_WINDOWS:
        med = median_flux_in_window(sky_flux, wave, win_min, win_max)
        sky_per_fiber.extend(med[np.isfinite(med)].tolist())

    if len(sky_per_fiber) == 0:
        return {"sky_median": np.nan, "sky_std": np.nan}

    arr = np.asarray(sky_per_fiber)
    return {"sky_median": float(np.nanmedian(arr)), "sky_std": float(np.nanstd(arr))}


def sky_line_residuals(
    cframe_hdul: fits.HDUList,
    sframe_hdul: fits.HDUList,
    sky_lines: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Measure RMS and flux of sky emission-line residuals (req. 9).

    The *before* (CFrame) and *after* (SFrame) spectra are compared in narrow
    windows around prominent sky emission lines.

    Parameters
    ----------
    cframe_hdul:
        HDUList of the CFrame (before sky subtraction).
    sframe_hdul:
        HDUList of the SFrame (after sky subtraction).
    sky_lines:
        Central wavelengths (Å) to analyse.  Defaults to ``_SKY_EMISSION_LINES``.

    Returns
    -------
    dict with keys ``"line_wave"``, ``"flux_before"`` (median in CFrame),
    ``"rms_after"`` (RMS in SFrame), and ``"rms_over_flux"`` — all 1-D arrays
    of length equal to the number of sky lines.
    """
    if sky_lines is None:
        sky_lines = _SKY_EMISSION_LINES

    wave = get_wavelength_array(cframe_hdul)
    flux_c = cframe_hdul["FLUX"].data.astype(float)
    flux_s = sframe_hdul["FLUX"].data.astype(float)

    line_waves: list[float] = []
    flux_before: list[float] = []
    rms_after: list[float] = []

    for lw in sky_lines:
        mask = (wave >= lw - _LINE_HALF_WIDTH) & (wave <= lw + _LINE_HALF_WIDTH)
        if not np.any(mask):
            continue
        line_waves.append(lw)
        flux_before.append(float(np.nanmedian(flux_c[:, mask])))
        rms_after.append(float(np.nanstd(flux_s[:, mask])))

    flux_arr = np.asarray(flux_before)
    rms_arr = np.asarray(rms_after)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_over_flux = np.where(flux_arr != 0.0, rms_arr / np.abs(flux_arr), np.nan)

    return {
        "line_wave": np.asarray(line_waves),
        "flux_before": flux_arr,
        "rms_after": rms_arr,
        "rms_over_flux": rms_over_flux,
    }


def sky_continuum_residuals(
    cframe_hdul: fits.HDUList,
    sframe_hdul: fits.HDUList,
) -> dict[str, float]:
    """Measure RMS/Flux of the night-sky continuum residuals (req. 10).

    Parameters
    ----------
    cframe_hdul:
        HDUList of the CFrame (before sky subtraction).
    sframe_hdul:
        HDUList of the SFrame (after sky subtraction).

    Returns
    -------
    dict with ``"cont_flux_before"``, ``"cont_rms_after"``, and
    ``"cont_rms_over_flux"``.
    """
    wave = get_wavelength_array(cframe_hdul)
    flux_c = cframe_hdul["FLUX"].data.astype(float)
    flux_s = sframe_hdul["FLUX"].data.astype(float)

    fluxes_c: list[float] = []
    fluxes_s: list[float] = []
    for win_min, win_max in _SKY_CONT_WINDOWS:
        med_c = median_flux_in_window(flux_c, wave, win_min, win_max)
        med_s = median_flux_in_window(flux_s, wave, win_min, win_max)
        fluxes_c.extend(med_c[np.isfinite(med_c)].tolist())
        fluxes_s.extend(med_s[np.isfinite(med_s)].tolist())

    flux_before = float(np.nanmedian(fluxes_c)) if fluxes_c else np.nan
    rms_after = float(np.nanstd(fluxes_s)) if fluxes_s else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_over_flux = (rms_after / abs(flux_before)) if (np.isfinite(flux_before) and flux_before != 0.0) else np.nan

    return {
        "cont_flux_before": flux_before,
        "cont_rms_after": rms_after,
        "cont_rms_over_flux": float(rms_over_flux),
    }


def sky_extinction(
    hdul: fits.HDUList,
    drpall_row: Row,
    reference_mag_col: str = "GAIA_GMAG",
    wave_min: float = 5400.0,
    wave_max: float = 5600.0,
) -> dict[str, float]:
    """Estimate the night-sky extinction from CFrame photometry (req. 12).

    The measured instrumental flux is compared with the reference catalogue
    magnitude to derive the extinction (Δmag).

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file.
    drpall_row:
        Row from the drpall table for this exposure.
    reference_mag_col:
        Column name in *drpall_row* holding the reference catalogue magnitude.
    wave_min, wave_max:
        Wavelength window used to estimate the broadband flux level (Å).

    Returns
    -------
    dict with ``"flux_level"``, ``"ref_mag"``, and ``"extinction_delta_mag"``.
    """
    wave = get_wavelength_array(hdul)
    flux = hdul["FLUX"].data.astype(float)
    flux_level = float(np.nanmedian(median_flux_in_window(flux, wave, wave_min, wave_max)))

    ref_mag = float(drpall_row[reference_mag_col]) if reference_mag_col in drpall_row.colnames else np.nan

    # Convert flux level to instrumental AB magnitude (arbitrary zero-point)
    with np.errstate(divide="ignore", invalid="ignore"):
        inst_mag = -2.5 * np.log10(flux_level) if flux_level > 0 else np.nan

    extinction = (inst_mag - ref_mag) if np.isfinite(inst_mag) and np.isfinite(ref_mag) else np.nan

    return {
        "flux_level": flux_level,
        "ref_mag": ref_mag,
        "extinction_delta_mag": float(extinction),
    }
