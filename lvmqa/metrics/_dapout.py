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


def sframe_on(ctx, ext: str, wave: np.ndarray) -> np.ndarray:
    """SFrame extension ``ext`` for the PT fibers on the DAP-output wavelength grid ``wave``
    (nearest pixel; NaN where no SFrame pixel lies within half a pixel). Safer than slicing the
    SFrame window to the same length: the two windows can differ by a pixel at the edges."""
    wave = np.asarray(wave, float)
    dw = float(np.median(np.diff(wave))) if wave.size > 1 else 1.0
    w, d = ctx.dap_sframe_window(ext, wave[0] - 2 * dw, wave[-1] + 2 * dw)
    if w.size == 0:
        return np.full((d.shape[0], wave.size), np.nan)
    j = np.clip(np.searchsorted(w, wave), 1, max(w.size - 1, 1)) if w.size > 1 else np.zeros(wave.size, int)
    if w.size > 1:
        j = np.where(np.abs(w[j - 1] - wave) < np.abs(w[j] - wave), j - 1, j)
    out = np.array(d[:, j], dtype=float)
    out[:, np.abs(w[j] - wave) > 0.5 * dw] = np.nan
    return out


def sigma_on(ctx, wave: np.ndarray) -> np.ndarray:
    """1-sigma error of the SFrame for the PT fibers on the DAP-output grid (NaN where ivar <= 0)."""
    iv = sframe_on(ctx, "IVAR", wave)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(iv) & (iv > 0), 1.0 / np.sqrt(np.where(iv > 0, iv, 1.0)), np.nan)


def local_baseline(r: np.ndarray, wave: np.ndarray, models, exclude=None, frac: float = 0.02,
                   grow: int = 3, min_pix: int = 5):
    """
    Residual continuum per spectrum: straight line through the medians of ``r`` in the line-free
    pixels left and right of the window centre (constant if only one side has >= min_pix pixels,
    0 if none). Line pixels = where any of the line ``models`` exceeds ``frac`` x its peak in that
    spectrum (grown by ``grow`` pixels), plus ``exclude`` (n_spec, n_pix). Removes the imperfect
    DAP continuum (stellar mismatch, smooth term) under the lines, which otherwise inflates the line
    residuals in fibers with a bright continuum. Returns (r - base, base).
    """
    r = np.asarray(r, float)
    line = np.zeros(r.shape, bool)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for m in models:
            m = np.abs(np.asarray(m, float))
            pk = np.nanmax(np.where(np.isfinite(m), m, np.nan), axis=1)
            line |= np.nan_to_num(m) > frac * np.nan_to_num(pk, nan=np.inf)[:, None]
        if exclude is not None:
            line |= exclude
        if grow > 0:
            g = line.copy()
            for k in range(1, grow + 1):
                g[:, k:] |= line[:, :-k]
                g[:, :-k] |= line[:, k:]
            line = g
        free = ~line & np.isfinite(r)
        wc = 0.5 * (wave[0] + wave[-1])
        L, R = free & (wave < wc)[None, :], free & (wave >= wc)[None, :]
        W = np.broadcast_to(wave, r.shape)
        yl, yr = np.nanmedian(np.where(L, r, np.nan), axis=1), np.nanmedian(np.where(R, r, np.nan), axis=1)
        xl, xr = np.nanmedian(np.where(L, W, np.nan), axis=1), np.nanmedian(np.where(R, W, np.nan), axis=1)
        nl, nr = L.sum(axis=1), R.sum(axis=1)
    both = (nl >= min_pix) & (nr >= min_pix) & (xr > xl)
    base = np.zeros(r.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        base[both] = yl[both, None] + (yr - yl)[both, None] * (wave[None, :] - xl[both, None]) / (xr - xl)[both, None]
    only_l, only_r = (nl >= min_pix) & ~both, (nr >= min_pix) & ~both & (nl < min_pix)
    base[only_l] = yl[only_l, None]
    base[only_r] = yr[only_r, None]
    return r - base, base


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
