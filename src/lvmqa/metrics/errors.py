"""
Error-propagation accuracy QA (requirement 15).

15.  Accuracy of the propagated error: RMS(spectra − DAP model) / propagated
     error — checks whether the distribution follows a Gaussian (use DAP output
     and SFrames).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import kstest, norm

from lvmqa.utils import get_wavelength_array

__all__ = ["error_propagation_accuracy"]


def error_propagation_accuracy(
    sframe_hdul: fits.HDUList,
    dap_model_hdul: fits.HDUList,
    flux_extname: str = "FLUX",
    model_extname: str = "MODEL",
    ivar_extname: str = "IVAR",
) -> dict[str, object]:
    """Assess propagated-error accuracy by comparing residuals to errors (req. 15).

    Computes the normalised residuals::

        χ = (F_obs − F_model) / σ

    where ``σ = 1 / sqrt(IVAR)``.  If the error propagation is correct these
    normalised residuals should follow a standard Gaussian (mean ≈ 0, σ ≈ 1).

    A Kolmogorov–Smirnov test against a standard normal is performed to
    quantify the agreement.

    Parameters
    ----------
    sframe_hdul:
        HDUList of the SFrame FITS file (observed spectra + inverse variance).
    dap_model_hdul:
        HDUList of the DAP model FITS file.  Must contain an extension with the
        best-fit model spectrum.
    flux_extname:
        Name of the flux extension in *sframe_hdul*.
    model_extname:
        Name of the model extension in *dap_model_hdul*.
    ivar_extname:
        Name of the inverse-variance extension in *sframe_hdul*.

    Returns
    -------
    dict with:
    - ``"chi_mean"``   — mean of the normalised residuals.
    - ``"chi_std"``    — standard deviation of the normalised residuals.
    - ``"ks_stat"``    — KS test statistic.
    - ``"ks_pvalue"``  — KS test p-value (high → consistent with Gaussian).
    - ``"chi_values"`` — flat 1-D array of all normalised residuals (for histogramming).
    """
    flux_obs = sframe_hdul[flux_extname].data.astype(float)
    ivar = sframe_hdul[ivar_extname].data.astype(float)
    flux_model = dap_model_hdul[model_extname].data.astype(float)

    # Error array; mask pixels with zero or negative IVAR
    good = ivar > 0
    sigma = np.where(good, 1.0 / np.sqrt(ivar), np.nan)

    residuals = flux_obs - flux_model
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(good, residuals / sigma, np.nan)

    chi_flat = chi[np.isfinite(chi)].ravel()

    if chi_flat.size < 10:
        return {
            "chi_mean": np.nan,
            "chi_std": np.nan,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "chi_values": chi_flat,
        }

    ks_stat, ks_pvalue = kstest(chi_flat, "norm", args=(0.0, 1.0))

    return {
        "chi_mean": float(np.mean(chi_flat)),
        "chi_std": float(np.std(chi_flat)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "chi_values": chi_flat,
    }
