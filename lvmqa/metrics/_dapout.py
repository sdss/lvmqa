"""
Helpers for metrics based on the DAP output file (model RSS, *.output.fits.gz).

Planes (lvmdap NAMEk keywords):
    0 org_spec            observed spectrum (= SFrame FLUX of the PT fibers, PT order)
    1 model_spec          stellar continuum model
    2 mod_joint_spec      stellar + NP emission-line model + smooth (Legendre) term
    3 gas_spec            observed - stellar - smooth term
    4 res_joint_spec      observed - mod_joint_spec        (residual of the NP joint model)
    5 no_gas_spec         observed - NP emission-line model
    6 gas_model_NP        emission lines: Gaussians from the NP moments (flux, vel, disp)
    7 gas_model_PEK       emission lines: parametric fit (PM_KEL)
    8 mod_joint_spec_PEK  stellar + parametric emission lines + smooth term
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

ORG, MODEL, JOINT, GAS, RES_NP, NOGAS, GAS_NP, GAS_PM, JOINT_PM = range(9)
C_KMS = 299792.458
SQRT_2_PI = np.sqrt(2.0 / np.pi)


def np_column(ctx, col: str) -> np.ndarray:
    """One NP_ELINES column aligned on PT (NaN if absent)."""
    return np.asarray(ctx.dap_np_columns([col])[col], dtype=float)


def sigma_window(ctx, w0: float, w1: float) -> Tuple[np.ndarray, np.ndarray]:
    """(wave, 1-sigma error) of the SFrame for the PT fibers (NaN where ivar <= 0)."""
    w, iv = ctx.dap_sframe_window("IVAR", w0, w1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(np.isfinite(iv) & (iv > 0), 1.0 / np.sqrt(np.where(iv > 0, iv, 1.0)), np.nan)
    return w, sig


def excess_residual(r: np.ndarray, sig: np.ndarray, model: np.ndarray, use: np.ndarray):
    """
    Unmodelled flux fraction per spectrum in the pixels ``use``:

        fx = (sum|r| - sqrt(2/pi) sum sigma) / sum model      (noise expectation removed)
        e_fx = sqrt(1 - 2/pi) * sqrt(sum sigma^2) / sum model

    fx ~ 0 for a perfect model at any S/N; fx = 0.1 means residuals (in excess of noise) of
    10% of the modelled line flux. Also returns the reduced chi^2 (r/sigma)^2.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = use & np.isfinite(r) & np.isfinite(sig) & np.isfinite(model)
        rr, ss, mm = (np.where(ok, a, 0.0) for a in (r, sig, model))
        fm = mm.sum(axis=1)
        fx = (np.abs(rr).sum(axis=1) - SQRT_2_PI * ss.sum(axis=1)) / fm
        efx = np.sqrt(1 - 2 / np.pi) * np.sqrt((ss ** 2).sum(axis=1)) / fm
        n = ok.sum(axis=1)
        chi2 = np.where(n > 0, ((rr / np.where(ok, ss, 1.0)) ** 2 * ok).sum(axis=1) / np.maximum(n, 1), np.nan)
        bad = (n < 3) | ~(fm > 0)
    fx[bad] = np.nan
    efx[bad] = np.nan
    return fx, efx, chi2, fm


def shifted_mask(wave: np.ndarray, centers: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Boolean (n_fib, n_pix): |wave - center_i| <= half_i."""
    return np.abs(wave[None, :] - centers[:, None]) <= half[:, None]
