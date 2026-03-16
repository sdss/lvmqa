"""
Diagnostic plotting utilities for per-exposure QA.

Each function produces a single ``matplotlib.figure.Figure`` that can be
saved to disk with ``fig.savefig(path)``.
"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_arm_color_jumps",
    "plot_continuum_snr",
    "plot_ha_snr",
    "plot_sky_brightness",
    "plot_sky_line_residuals",
    "plot_calibration_offsets",
    "plot_spectral_resolution",
    "plot_telluric_ew",
    "plot_error_distribution",
    "plot_saturation_map",
    "plot_emission_rms_over_flux",
    "plot_summary_dashboard",
]


def _default_fig(nrows: int = 1, ncols: int = 1, **kwargs: Any) -> tuple:
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    fig.set_layout_engine("tight")
    return fig, axes


# ---------------------------------------------------------------------------
# Individual diagnostic plots
# ---------------------------------------------------------------------------

def plot_arm_color_jumps(jump_results: dict[str, np.ndarray]) -> plt.Figure:
    """Histogram of arm colour-jump flux ratios (req. 1)."""
    fig, axes = _default_fig(1, len(jump_results), figsize=(5 * len(jump_results), 4))
    if len(jump_results) == 1:
        axes = [axes]
    for ax, (label, ratios) in zip(axes, jump_results.items()):
        valid = ratios[np.isfinite(ratios)]
        if valid.size:
            ax.hist(valid, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(1.0, color="red", lw=1.5, ls="--", label="ideal")
        ax.set_xlabel("Flux ratio (blue/red arm)")
        ax.set_ylabel("N fibers")
        ax.set_title(f"Arm jump: {label}")
        ax.legend(fontsize=8)
    return fig


def plot_continuum_snr(snr: np.ndarray) -> plt.Figure:
    """Histogram of per-fiber continuum S/N (req. 5)."""
    fig, ax = _default_fig(figsize=(6, 4))
    valid = snr[np.isfinite(snr)]
    if valid.size:
        ax.hist(valid, bins=30, color="forestgreen", edgecolor="white")
    ax.axvline(np.nanmedian(snr), color="red", lw=1.5, ls="--",
               label=f"median = {np.nanmedian(snr):.1f}")
    ax.set_xlabel("Mean S/N per fiber")
    ax.set_ylabel("N fibers")
    ax.set_title("Continuum S/N (5400–5600 Å)")
    ax.legend(fontsize=8)
    return fig


def plot_ha_snr(snr: np.ndarray) -> plt.Figure:
    """Histogram of per-fiber Hα S/N (req. 4)."""
    fig, ax = _default_fig(figsize=(6, 4))
    valid = snr[np.isfinite(snr)]
    if valid.size:
        ax.hist(valid, bins=30, color="tomato", edgecolor="white")
    ax.axvline(np.nanmedian(snr), color="darkred", lw=1.5, ls="--",
               label=f"median = {np.nanmedian(snr):.1f}")
    ax.set_xlabel("Mean S/N per fiber")
    ax.set_ylabel("N fibers")
    ax.set_title("Hα S/N")
    ax.legend(fontsize=8)
    return fig


def plot_sky_brightness(sky_result: dict[str, float]) -> plt.Figure:
    """Bar chart of sky brightness statistics (req. 8)."""
    fig, ax = _default_fig(figsize=(5, 4))
    med = sky_result.get("sky_median", np.nan)
    std = sky_result.get("sky_std", np.nan)
    ax.bar(["sky_median"], [med if np.isfinite(med) else 0], yerr=[std if np.isfinite(std) else 0],
           color="navy", alpha=0.7, capsize=5)
    ax.set_ylabel("Sky flux (erg s⁻¹ cm⁻² Å⁻¹ arcsec⁻²)")
    ax.set_title("Sky brightness")
    return fig


def plot_sky_line_residuals(residual_result: dict[str, np.ndarray]) -> plt.Figure:
    """Bar chart of sky emission-line RMS/Flux (req. 9)."""
    fig, ax = _default_fig(figsize=(8, 4))
    lines = residual_result.get("line_wave", np.array([]))
    rof = residual_result.get("rms_over_flux", np.array([]))
    if lines.size:
        ax.bar(lines.astype(str), rof, color="darkorange", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Sky line wavelength (Å)")
    ax.set_ylabel("RMS / Flux")
    ax.set_title("Sky emission-line residuals")
    return fig


def plot_calibration_offsets(calib_result: dict[str, np.ndarray]) -> plt.Figure:
    """Box plot of Δλ per sky line across fibers (req. 13)."""
    fig, ax = _default_fig(figsize=(9, 4))
    lines = calib_result.get("line_wave", np.array([]))
    delta = calib_result.get("delta_lambda", np.array([[np.nan]]))
    if lines.size and delta.size:
        data = [delta[i][np.isfinite(delta[i])] for i in range(len(lines))]
        ax.boxplot(data, tick_labels=[f"{lw:.0f}" for lw in lines], sym=".", whis=1.5)
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_xlabel("Sky line wavelength (Å)")
    ax.set_ylabel("Δλ (Å)")
    ax.set_title("Wavelength calibration offset")
    return fig


def plot_spectral_resolution(res_result: dict[str, np.ndarray]) -> plt.Figure:
    """Plot mean FWHM and resolving power R vs wavelength (req. 14)."""
    fig, (ax1, ax2) = _default_fig(1, 2, figsize=(10, 4))
    lines = res_result.get("line_wave", np.array([]))
    fwhm = res_result.get("mean_fwhm", np.array([]))
    R = res_result.get("mean_R", np.array([]))

    if lines.size:
        ax1.plot(lines, fwhm, "o-", color="purple")
        ax1.set_xlabel("Wavelength (Å)")
        ax1.set_ylabel("Mean FWHM (Å)")
        ax1.set_title("Spectral resolution — FWHM")

        ax2.plot(lines, R, "s-", color="teal")
        ax2.set_xlabel("Wavelength (Å)")
        ax2.set_ylabel("Resolving power R")
        ax2.set_title("Spectral resolution — R = λ/FWHM")
    return fig


def plot_telluric_ew(telluric_result: dict[str, dict]) -> plt.Figure:
    """Bar chart of telluric absorption EW per band (req. 11)."""
    fig, ax = _default_fig(figsize=(7, 4))
    labels = list(telluric_result.keys())
    ew_vals = [telluric_result[l].get("ew_median", np.nan) for l in labels]
    ew_errs = [telluric_result[l].get("ew_std", 0.0) for l in labels]
    ax.bar(labels, ew_vals, yerr=ew_errs, color="goldenrod", edgecolor="white", capsize=4)
    ax.set_ylabel("EW (Å)")
    ax.set_title("Telluric absorption EW")
    return fig


def plot_error_distribution(error_result: dict[str, object]) -> plt.Figure:
    """Histogram of normalised χ residuals (req. 15)."""
    fig, ax = _default_fig(figsize=(6, 4))
    chi = np.asarray(error_result.get("chi_values", []))
    if chi.size:
        ax.hist(chi, bins=60, density=True, color="cornflowerblue", edgecolor="white", alpha=0.8,
                label="data")
        x = np.linspace(-5, 5, 300)
        ax.plot(x, 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2), "r-", lw=2, label="N(0,1)")
    chi_mean = error_result.get("chi_mean", np.nan)
    chi_std = error_result.get("chi_std", np.nan)
    ax.set_xlabel("χ = (F_obs − F_model) / σ")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Error propagation  μ={chi_mean:.3f}  σ={chi_std:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)
    return fig


def plot_saturation_map(sat_result: dict[str, object]) -> plt.Figure:
    """Bar chart of per-fiber saturated pixel counts (req. 16)."""
    fig, ax = _default_fig(figsize=(8, 4))
    counts = np.asarray(sat_result.get("fiber_sat_counts", []))
    if counts.size:
        ax.bar(np.arange(counts.size), counts, color="crimson", edgecolor="none", width=1.0)
    n_sat = sat_result.get("n_saturated", 0)
    frac = sat_result.get("frac_saturated", 0.0)
    ax.set_xlabel("Fiber index")
    ax.set_ylabel("Saturated pixels")
    ax.set_title(f"Saturated pixels  total={n_sat}  frac={frac:.4f}")
    return fig


def plot_emission_rms_over_flux(
    rms_result: dict[str, object],
    title: str = "Emission-line RMS/Flux",
) -> plt.Figure:
    """Bar chart of RMS_res/Flux per emission line (reqs. 17, 18)."""
    fig, ax = _default_fig(figsize=(8, 4))
    rof = np.asarray(rms_result.get("rms_over_flux", []))
    names = rms_result.get("line_names", [])
    x = np.arange(rof.size)
    if rof.size:
        labels = list(names) if names else [str(i) for i in x]
        ax.bar(labels[:rof.size], rof, color="mediumseagreen", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Emission line")
    ax.set_ylabel("RMS / Flux")
    ax.set_title(title)
    return fig


def plot_summary_dashboard(metrics: dict[str, Any], exposure_id: str = "") -> plt.Figure:
    """Multi-panel summary dashboard for a single exposure.

    Plots a compact 3×3 grid of the most important scalar QA metrics as a
    simple bar / text panel so that the overall health of the exposure can be
    assessed at a glance.

    Parameters
    ----------
    metrics:
        Dict as returned by :func:`lvmqa.qa.run_qa` for a single exposure.
    exposure_id:
        Human-readable label for the exposure (e.g. filename or ID).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(f"QA Dashboard — {exposure_id}", fontsize=13)

    def _text_panel(ax: plt.Axes, title: str, kv: dict) -> None:
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, pad=2)
        lines = [f"{k}: {v:.4g}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}"
                 for k, v in kv.items()]
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=8, family="monospace")

    # Panel 0 — arm colour jumps
    jump = metrics.get("arm_color_jumps", {})
    _text_panel(axes[0, 0], "Arm colour jumps",
                {k: float(np.nanmedian(v)) for k, v in jump.items()})

    # Panel 1 — continuum S/N
    snr_cont = np.asarray(metrics.get("continuum_snr", [np.nan]))
    _text_panel(axes[0, 1], "Continuum S/N",
                {"median": float(np.nanmedian(snr_cont)),
                 "std": float(np.nanstd(snr_cont))})

    # Panel 2 — Hα S/N
    snr_ha = np.asarray(metrics.get("ha_snr", [np.nan]))
    _text_panel(axes[0, 2], "Hα S/N",
                {"median": float(np.nanmedian(snr_ha)),
                 "std": float(np.nanstd(snr_ha))})

    # Panel 3 — sky brightness
    sky = metrics.get("sky_brightness", {})
    _text_panel(axes[1, 0], "Sky brightness", {
        "median": sky.get("sky_median", np.nan),
        "std": sky.get("sky_std", np.nan),
    })

    # Panel 4 — sky continuum residuals
    sky_cont = metrics.get("sky_continuum_residuals", {})
    _text_panel(axes[1, 1], "Sky continuum residuals", {
        "flux_before": sky_cont.get("cont_flux_before", np.nan),
        "rms_after": sky_cont.get("cont_rms_after", np.nan),
        "rms/flux": sky_cont.get("cont_rms_over_flux", np.nan),
    })

    # Panel 5 — saturation
    sat = metrics.get("saturation", {})
    _text_panel(axes[1, 2], "Saturation", {
        "n_saturated": sat.get("n_saturated", 0),
        "frac": sat.get("frac_saturated", 0.0),
        "n_fibers_aff.": sat.get("n_fibers_affected", 0),
    })

    # Panel 6 — wavelength calibration
    calib = metrics.get("wavelength_calibration", {})
    mean_dl = calib.get("mean_delta_lambda", np.array([np.nan]))
    _text_panel(axes[2, 0], "Wavelength calibration", {
        "mean Δλ": float(np.nanmean(mean_dl)),
        "std Δλ": float(np.nanstd(mean_dl)),
    })

    # Panel 7 — error propagation
    err = metrics.get("error_propagation", {})
    _text_panel(axes[2, 1], "Error propagation", {
        "χ mean": err.get("chi_mean", np.nan),
        "χ std": err.get("chi_std", np.nan),
        "KS p": err.get("ks_pvalue", np.nan),
    })

    # Panel 8 — telluric
    tell = metrics.get("telluric_ew", {})
    _text_panel(axes[2, 2], "Telluric EW (Å)", {
        k: v.get("ew_median", np.nan) for k, v in tell.items()
    })

    fig.set_layout_engine("tight")
    return fig
