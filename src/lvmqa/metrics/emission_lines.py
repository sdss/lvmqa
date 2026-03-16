"""
Emission-line QA metrics (requirements 4, 6, 17, 18).

4.   S/N of Hα (use SFrames).
6.   Weakest detected emission line — data depth (use SFrames and DAP outputs).
17.  RMS_res / Flux for different emission lines measured by DAP.
18.  Compare reported errors in emission lines with residual spectra analysis.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from lvmqa.utils import get_wavelength_array, snr_per_fiber

__all__ = [
    "ha_snr",
    "weakest_emission_line",
    "emission_line_rms_over_flux",
    "emission_line_error_comparison",
]

# Rest-frame wavelength of Hα (Å)
_HA_WAVE = 6562.8
# Half-width of the window used to compute Hα S/N
_HA_HALF_WIDTH = 10.0

# Narrow continuum windows flanking Hα used for local background subtraction
_HA_CONT_BLUE = (6500.0, 6540.0)
_HA_CONT_RED = (6590.0, 6630.0)


def ha_snr(hdul: fits.HDUList, redshift: float = 0.0) -> np.ndarray:
    """Compute per-fiber S/N of the Hα emission line (req. 4).

    A local continuum is estimated from two flanking windows, subtracted, and
    then the S/N is evaluated over a ±10 Å window centred on Hα.

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file with ``FLUX`` and ``IVAR`` extensions.
    redshift:
        Systemic redshift of the target field (default 0 → Milky Way).

    Returns
    -------
    numpy.ndarray
        Per-fiber Hα S/N (mean over the line window).
    """
    flux = hdul["FLUX"].data.astype(float)
    ivar = hdul["IVAR"].data.astype(float)
    wave = get_wavelength_array(hdul)

    # Shift windows to observed frame
    z1 = 1.0 + redshift
    ha_obs = _HA_WAVE * z1
    win_min = ha_obs - _HA_HALF_WIDTH
    win_max = ha_obs + _HA_HALF_WIDTH

    # Continuum windows
    cb_min, cb_max = _HA_CONT_BLUE[0] * z1, _HA_CONT_BLUE[1] * z1
    cr_min, cr_max = _HA_CONT_RED[0] * z1, _HA_CONT_RED[1] * z1

    mask_cont = ((wave >= cb_min) & (wave <= cb_max)) | ((wave >= cr_min) & (wave <= cr_max))
    cont_level = np.nanmedian(flux[:, mask_cont], axis=1) if np.any(mask_cont) else np.zeros(flux.shape[0])

    flux_cont_sub = flux - cont_level[:, np.newaxis]
    return snr_per_fiber(flux_cont_sub, ivar, wave, win_min, win_max)


def weakest_emission_line(
    dap_table: Table,
    flux_col: str = "LINE_FLUX",
    snr_col: str = "LINE_SNR",
    snr_threshold: float = 3.0,
) -> dict[str, float]:
    """Find the weakest detected emission line in a DAP table (req. 6).

    Parameters
    ----------
    dap_table:
        DAP emission-line table.
    flux_col:
        Column with emission-line fluxes.
    snr_col:
        Column with per-line S/N ratios.
    snr_threshold:
        Minimum S/N considered a detection.

    Returns
    -------
    dict with keys ``"min_flux"``, ``"median_snr_at_detection"``, and
    ``"n_detections"``.
    """
    if flux_col not in dap_table.colnames or snr_col not in dap_table.colnames:
        return {"min_flux": np.nan, "median_snr_at_detection": np.nan, "n_detections": 0}

    fluxes = np.asarray(dap_table[flux_col], dtype=float)
    snrs = np.asarray(dap_table[snr_col], dtype=float)

    detected = np.isfinite(fluxes) & np.isfinite(snrs) & (snrs >= snr_threshold)
    n_det = int(np.sum(detected))
    if n_det == 0:
        return {"min_flux": np.nan, "median_snr_at_detection": np.nan, "n_detections": 0}

    return {
        "min_flux": float(np.nanmin(fluxes[detected])),
        "median_snr_at_detection": float(np.nanmedian(snrs[detected])),
        "n_detections": n_det,
    }


def emission_line_rms_over_flux(
    dap_table: Table,
    flux_col: str = "LINE_FLUX",
    rms_col: str = "LINE_RMS",
) -> dict[str, np.ndarray]:
    """Compute RMS_residual / Flux for each emission line in the DAP table (req. 17).

    Parameters
    ----------
    dap_table:
        DAP emission-line table.
    flux_col:
        Column holding the fitted line flux.
    rms_col:
        Column holding the residual RMS around the line fit.

    Returns
    -------
    dict with keys ``"rms_over_flux"`` (array) and ``"line_names"`` (array of
    str if a ``LINE_NAME`` column exists, else empty list).
    """
    if flux_col not in dap_table.colnames or rms_col not in dap_table.colnames:
        return {"rms_over_flux": np.array([np.nan]), "line_names": []}

    flux = np.asarray(dap_table[flux_col], dtype=float)
    rms = np.asarray(dap_table[rms_col], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rms_over_flux = np.where(flux != 0.0, rms / flux, np.nan)

    line_names = list(dap_table["LINE_NAME"]) if "LINE_NAME" in dap_table.colnames else []
    return {"rms_over_flux": rms_over_flux, "line_names": line_names}


def emission_line_error_comparison(
    hdul: fits.HDUList,
    dap_table: Table,
    wave_col: str = "LINE_WAVE",
    flux_col: str = "LINE_FLUX",
    flux_err_col: str = "LINE_FLUX_ERR",
    half_width: float = 20.0,
) -> dict[str, np.ndarray]:
    """Compare DAP reported flux errors with residual-spectrum scatter (req. 18).

    For each entry in the DAP table the spectral residuals in a window
    ``±half_width`` Å around the line are measured from the SFrame.  The ratio
    ``reported_error / residual_std`` quantifies whether the propagated error
    budget is consistent with the actual scatter.

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file.
    dap_table:
        DAP emission-line table.
    wave_col:
        Column with the central wavelength (Å) of each line.
    flux_col:
        Column with fitted line fluxes.
    flux_err_col:
        Column with reported flux uncertainties.
    half_width:
        Half-width of the residual window in Å.

    Returns
    -------
    dict with keys ``"reported_err"``, ``"residual_std"``, and
    ``"err_ratio"`` (1-D arrays, one entry per DAP table row).
    """
    required = {wave_col, flux_col, flux_err_col}
    if not required.issubset(dap_table.colnames):
        nan_arr = np.full(len(dap_table), np.nan)
        return {"reported_err": nan_arr, "residual_std": nan_arr, "err_ratio": nan_arr}

    flux_rss = hdul["FLUX"].data.astype(float)
    wave = get_wavelength_array(hdul)

    line_waves = np.asarray(dap_table[wave_col], dtype=float)
    reported_errs = np.asarray(dap_table[flux_err_col], dtype=float)

    residual_stds = np.full(len(dap_table), np.nan)
    for i, lw in enumerate(line_waves):
        mask = (wave >= lw - half_width) & (wave <= lw + half_width)
        if np.any(mask):
            window_flux = flux_rss[:, mask]
            # Use the standard deviation across fibers and wavelength pixels
            residual_stds[i] = float(np.nanstd(window_flux))

    with np.errstate(divide="ignore", invalid="ignore"):
        err_ratio = np.where(residual_stds != 0.0, reported_errs / residual_stds, np.nan)

    return {
        "reported_err": reported_errs,
        "residual_std": residual_stds,
        "err_ratio": err_ratio,
    }
