"""
Main QA orchestrator for LVM data products.

``run_qa`` collects all available metrics for a single exposure and writes the
results to a FITS binary table.  ``build_qa_table`` iterates over a list of
exposures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from lvmqa.metrics.calibration import spectral_resolution, wavelength_calibration_accuracy
from lvmqa.metrics.continuum import arm_color_jumps, continuum_snr, weakest_continuum
from lvmqa.metrics.emission_lines import (
    emission_line_error_comparison,
    emission_line_rms_over_flux,
    ha_snr,
    weakest_emission_line,
)
from lvmqa.metrics.errors import error_propagation_accuracy
from lvmqa.metrics.photometry import absolute_photometry, color_comparison
from lvmqa.metrics.saturation import detect_saturated_pixels
from lvmqa.metrics.sky import (
    sky_brightness,
    sky_continuum_residuals,
    sky_extinction,
    sky_line_residuals,
)
from lvmqa.metrics.stellar import (
    emission_line_map_comparison,
    parametric_nonparametric_comparison,
    stellar_gaia_comparison,
)
from lvmqa.metrics.telluric import telluric_ew
from lvmqa.utils import load_dap_table, load_drpall, load_rss

__all__ = ["run_qa", "build_qa_table", "write_qa_table"]

log = logging.getLogger(__name__)


def _safe(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *func* and return its result; return ``None`` on any exception."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        log.warning("QA metric failed (%s): %s", func.__name__, exc)
        return None


def run_qa(
    exposure_id: str,
    sframe_path: str | Path | None = None,
    cframe_path: str | Path | None = None,
    dap_path: str | Path | None = None,
    dap_model_path: str | Path | None = None,
    drpall_path: str | Path | None = None,
    drpall_row_index: int | None = None,
    gaia_table: Table | None = None,
    catalogue_table: Table | None = None,
    redshift: float = 0.0,
) -> dict[str, Any]:
    """Run all available QA metrics for a single exposure.

    Each metric is attempted independently; failures are logged as warnings and
    the metric is skipped.

    Parameters
    ----------
    exposure_id:
        Human-readable identifier (e.g. the exposure number or filename stem).
    sframe_path:
        Path to the SFrame FITS file (sky-subtracted RSS).
    cframe_path:
        Path to the CFrame FITS file (non-sky-subtracted RSS).
    dap_path:
        Path to the DAP emission-line / stellar-population FITS table.
    dap_model_path:
        Path to the DAP best-fit model FITS file.
    drpall_path:
        Path to the drpall summary FITS table.
    drpall_row_index:
        Row index in the drpall table corresponding to this exposure.
    gaia_table:
        Pre-loaded Gaia catalogue ``astropy.table.Table`` (optional).
    catalogue_table:
        Pre-loaded external emission-line map catalogue (optional).
    redshift:
        Systemic redshift of the target field (used for Hα window).

    Returns
    -------
    dict
        Flat dictionary of all computed metrics keyed by metric name.
    """
    metrics: dict[str, Any] = {"exposure_id": exposure_id}

    sframe_hdul = fits.open(sframe_path) if sframe_path else None
    cframe_hdul = fits.open(cframe_path) if cframe_path else None
    dap_table = load_dap_table(dap_path) if dap_path else None
    dap_model_hdul = fits.open(dap_model_path) if dap_model_path else None
    drpall = load_drpall(drpall_path) if drpall_path else None
    drpall_row = drpall[drpall_row_index] if (drpall is not None and drpall_row_index is not None) else None

    try:
        # ------------------------------------------------------------------
        # Req. 1 — arm colour jumps (CFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None:
            metrics["arm_color_jumps"] = _safe(arm_color_jumps, cframe_hdul)

        # ------------------------------------------------------------------
        # Reqs. 2–3 — photometry comparisons (SFrame + drpall)
        # ------------------------------------------------------------------
        # These require filter transmission curves that are site-specific;
        # callers should pass them in via the appropriate metrics directly.
        # Here we store placeholders.
        metrics["color_comparison"] = None
        metrics["absolute_photometry"] = None

        # ------------------------------------------------------------------
        # Req. 4 — Hα S/N (SFrame)
        # ------------------------------------------------------------------
        if sframe_hdul is not None:
            metrics["ha_snr"] = _safe(ha_snr, sframe_hdul, redshift)

        # ------------------------------------------------------------------
        # Req. 5 — continuum S/N (SFrame)
        # ------------------------------------------------------------------
        if sframe_hdul is not None:
            metrics["continuum_snr"] = _safe(continuum_snr, sframe_hdul)

        # ------------------------------------------------------------------
        # Req. 6 — weakest emission line (DAP)
        # ------------------------------------------------------------------
        if dap_table is not None:
            metrics["weakest_emission_line"] = _safe(weakest_emission_line, dap_table)

        # ------------------------------------------------------------------
        # Req. 7 — weakest continuum (SFrame)
        # ------------------------------------------------------------------
        if sframe_hdul is not None:
            metrics["weakest_continuum"] = _safe(weakest_continuum, sframe_hdul)

        # ------------------------------------------------------------------
        # Req. 8 — sky brightness (CFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None:
            metrics["sky_brightness"] = _safe(sky_brightness, cframe_hdul)

        # ------------------------------------------------------------------
        # Req. 9 — sky line residuals (CFrame + SFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None and sframe_hdul is not None:
            metrics["sky_line_residuals"] = _safe(sky_line_residuals, cframe_hdul, sframe_hdul)

        # ------------------------------------------------------------------
        # Req. 10 — sky continuum residuals (CFrame + SFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None and sframe_hdul is not None:
            metrics["sky_continuum_residuals"] = _safe(sky_continuum_residuals, cframe_hdul, sframe_hdul)

        # ------------------------------------------------------------------
        # Req. 11 — telluric EW (SFrame)
        # ------------------------------------------------------------------
        if sframe_hdul is not None:
            metrics["telluric_ew"] = _safe(telluric_ew, sframe_hdul)

        # ------------------------------------------------------------------
        # Req. 12 — night-sky extinction (CFrame + drpall)
        # ------------------------------------------------------------------
        if cframe_hdul is not None and drpall_row is not None:
            metrics["sky_extinction"] = _safe(sky_extinction, cframe_hdul, drpall_row)

        # ------------------------------------------------------------------
        # Req. 13 — wavelength calibration (CFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None:
            metrics["wavelength_calibration"] = _safe(wavelength_calibration_accuracy, cframe_hdul)

        # ------------------------------------------------------------------
        # Req. 14 — spectral resolution (CFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None:
            metrics["spectral_resolution"] = _safe(spectral_resolution, cframe_hdul)

        # ------------------------------------------------------------------
        # Req. 15 — error propagation (SFrame + DAP model)
        # ------------------------------------------------------------------
        if sframe_hdul is not None and dap_model_hdul is not None:
            metrics["error_propagation"] = _safe(
                error_propagation_accuracy, sframe_hdul, dap_model_hdul
            )

        # ------------------------------------------------------------------
        # Req. 16 — saturated pixels (CFrame)
        # ------------------------------------------------------------------
        if cframe_hdul is not None:
            metrics["saturation"] = _safe(detect_saturated_pixels, cframe_hdul)

        # ------------------------------------------------------------------
        # Req. 17 — RMS_res / Flux for emission lines (DAP)
        # ------------------------------------------------------------------
        if dap_table is not None:
            metrics["emission_rms_over_flux"] = _safe(emission_line_rms_over_flux, dap_table)

        # ------------------------------------------------------------------
        # Req. 18 — emission line error comparison (DAP + SFrame)
        # ------------------------------------------------------------------
        if sframe_hdul is not None and dap_table is not None:
            metrics["emission_error_comparison"] = _safe(
                emission_line_error_comparison, sframe_hdul, dap_table
            )

        # ------------------------------------------------------------------
        # Req. 19 — parametric vs non-parametric (DAP)
        # ------------------------------------------------------------------
        if dap_table is not None:
            metrics["param_nonparam_comparison"] = _safe(
                parametric_nonparametric_comparison, dap_table
            )

        # ------------------------------------------------------------------
        # Req. 20 — emission line map comparison (DAP + catalogue)
        # ------------------------------------------------------------------
        if dap_table is not None and catalogue_table is not None:
            metrics["emission_map_comparison"] = _safe(
                emission_line_map_comparison, dap_table, catalogue_table
            )

        # ------------------------------------------------------------------
        # Req. 21 — stellar properties vs Gaia (DAP + Gaia)
        # ------------------------------------------------------------------
        if dap_table is not None and gaia_table is not None:
            metrics["stellar_gaia_comparison"] = _safe(
                stellar_gaia_comparison, dap_table, gaia_table
            )

    finally:
        for hdul in (sframe_hdul, cframe_hdul, dap_model_hdul):
            if hdul is not None:
                hdul.close()

    return metrics


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metric dicts to scalar values for FITS table columns."""
    flat: dict[str, Any] = {"EXPOSURE_ID": str(metrics.get("exposure_id", ""))}

    def _scalar(val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return float(np.nanmedian(val)) if val.size else np.nan
        if isinstance(val, (int, float, np.floating, np.integer)):
            return float(val)
        return np.nan

    # Arm colour jumps
    jumps = metrics.get("arm_color_jumps") or {}
    for key, arr in (jumps.items() if isinstance(jumps, dict) else {}.items()):
        flat[f"JUMP_{key.upper()}_MEDIAN"] = _scalar(arr)

    # Continuum S/N
    snr_c = metrics.get("continuum_snr")
    flat["CONT_SNR_MEDIAN"] = _scalar(snr_c) if snr_c is not None else np.nan

    # Hα S/N
    snr_ha = metrics.get("ha_snr")
    flat["HA_SNR_MEDIAN"] = _scalar(snr_ha) if snr_ha is not None else np.nan

    # Weakest emission line
    wel = metrics.get("weakest_emission_line") or {}
    flat["WEAKEST_LINE_FLUX"] = float(wel.get("min_flux", np.nan))
    flat["WEAKEST_LINE_N"] = int(wel.get("n_detections", 0))

    # Weakest continuum
    flat["WEAKEST_CONT_FLUX"] = float(metrics.get("weakest_continuum") or np.nan)

    # Sky brightness
    sky = metrics.get("sky_brightness") or {}
    flat["SKY_BRIGHTNESS"] = float(sky.get("sky_median", np.nan))
    flat["SKY_BRIGHTNESS_STD"] = float(sky.get("sky_std", np.nan))

    # Sky continuum residuals
    sky_cont = metrics.get("sky_continuum_residuals") or {}
    flat["SKY_CONT_RMS_OVER_FLUX"] = float(sky_cont.get("cont_rms_over_flux", np.nan))

    # Telluric EW
    tell = metrics.get("telluric_ew") or {}
    for band, vals in (tell.items() if isinstance(tell, dict) else {}.items()):
        flat[f"TELL_EW_{band}"] = float(vals.get("ew_median", np.nan)) if isinstance(vals, dict) else np.nan

    # Sky extinction
    ext = metrics.get("sky_extinction") or {}
    flat["SKY_EXTINCTION"] = float(ext.get("extinction_delta_mag", np.nan))

    # Wavelength calibration
    calib = metrics.get("wavelength_calibration") or {}
    mean_dl = calib.get("mean_delta_lambda", np.array([np.nan]))
    flat["WAVE_CALIB_MEAN_DL"] = _scalar(np.asarray(mean_dl))
    flat["WAVE_CALIB_STD_DL"] = float(np.nanstd(np.asarray(mean_dl)))

    # Spectral resolution
    res = metrics.get("spectral_resolution") or {}
    flat["SPEC_RES_MEAN_R"] = _scalar(np.asarray(res.get("mean_R", np.array([np.nan]))))

    # Error propagation
    err = metrics.get("error_propagation") or {}
    flat["CHI_MEAN"] = float(err.get("chi_mean", np.nan))
    flat["CHI_STD"] = float(err.get("chi_std", np.nan))
    flat["CHI_KS_PVALUE"] = float(err.get("ks_pvalue", np.nan))

    # Saturation
    sat = metrics.get("saturation") or {}
    flat["N_SATURATED"] = int(sat.get("n_saturated", 0))
    flat["FRAC_SATURATED"] = float(sat.get("frac_saturated", 0.0))

    return flat


def build_qa_table(all_metrics: list[dict[str, Any]]) -> Table:
    """Build an Astropy Table from a list of per-exposure metric dicts.

    Parameters
    ----------
    all_metrics:
        List of dicts as returned by :func:`run_qa`, one per exposure.

    Returns
    -------
    astropy.table.Table
        One row per exposure, one column per scalar metric.
    """
    rows = [_flatten_metrics(m) for m in all_metrics]
    if not rows:
        return Table()
    # Stack using vstack so missing keys become masked values
    return Table(rows=rows)


def write_qa_table(table: Table, path: str | Path) -> None:
    """Write the QA summary table to a FITS binary table file.

    Parameters
    ----------
    table:
        QA summary table as returned by :func:`build_qa_table`.
    path:
        Output FITS file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(str(path), format="fits", overwrite=True)
    log.info("QA table written to %s (%d rows)", path, len(table))
