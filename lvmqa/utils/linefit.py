"""
Vectorised (batched) least-squares fit of a Gaussian + constant to many 1-D profiles at once.

    model(x) = amp * exp(-0.5 * ((x - mu) / sig)^2) + bkg

All profiles share the same x grid (e.g. one sky line in all fibers of an RSS frame).
Levenberg-Marquardt with an individual damping factor per profile; the 4x4 normal equations
are solved for all profiles simultaneously with numpy, so ~2000 fits take a few ms.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _model_jac(x, p):
    amp, mu, sig, bkg = (p[:, i:i + 1] for i in range(4))
    dx = x[None, :] - mu
    g = np.exp(-0.5 * (dx / sig) ** 2)
    model = amp * g + bkg
    jac = np.empty(g.shape + (4,))
    jac[..., 0] = g
    jac[..., 1] = amp * g * dx / sig ** 2
    jac[..., 2] = amp * g * dx ** 2 / sig ** 3
    jac[..., 3] = 1.0
    return model, jac


def fit_gaussians(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
                  mu0: Optional[np.ndarray] = None, sig0: Optional[np.ndarray] = None,
                  n_iter: int = 30, sig_bounds=(0.05, 10.0)) -> Dict[str, np.ndarray]:
    """
    Fit a Gaussian + constant to each row of ``y`` (n_prof, n_pix) sampled at ``x`` (n_pix).

    w     : weights (inverse variances), same shape as y; NaN / non-positive -> ignored pixel
    mu0   : initial centroids (n_prof,), default = position of the maximum
    sig0  : initial sigmas (n_prof,), default = 1.5 pixel

    Returns dict of arrays (n_prof,): amp, mu, sig, fwhm, bkg, e_amp, e_mu, e_sig, chi2r, ok.
    ``ok`` = finite parameters, positive amplitude, sigma inside ``sig_bounds`` and >= 5
    usable pixels. Errors are scaled by sqrt(reduced chi2) when chi2r > 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.array(y, dtype=float, ndmin=2)
    n, npix = y.shape
    if w is None:
        w = np.ones_like(y)
    w = np.array(w, dtype=float)
    good = np.isfinite(y) & np.isfinite(w) & (w > 0)
    w = np.where(good, w, 0.0)
    y = np.where(good, y, 0.0)
    ngood = good.sum(axis=1)

    dxp = float(np.median(np.diff(x))) if npix > 1 else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ymask = np.where(good, y, np.nan)
        bkg0 = np.nanpercentile(ymask, 10, axis=1)
        imax = np.nanargmax(np.where(good, y, -np.inf), axis=1)
    if mu0 is None:
        mu0 = x[imax]
    if sig0 is None:
        sig0 = np.full(n, 1.5 * dxp)
    amp0 = y[np.arange(n), np.argmin(np.abs(x[None, :] - np.asarray(mu0)[:, None]), axis=1)] - bkg0
    p = np.column_stack([amp0, np.broadcast_to(mu0, (n,)), np.broadcast_to(sig0, (n,)), bkg0]).astype(float)
    p = np.nan_to_num(p)
    lam = np.full(n, 1e-3)

    def chi2_of(pp):
        m, _ = _model_jac(x, pp)
        return np.sum(w * (y - m) ** 2, axis=1)

    chi2 = chi2_of(p)
    eye = np.eye(4)
    for _ in range(n_iter):
        model, jac = _model_jac(x, p)
        r = y - model
        jw = jac * w[..., None]
        a = np.einsum("npi,npj->nij", jw, jac)
        b = np.einsum("npi,np->ni", jw, r)
        diag = np.einsum("nii->ni", a)
        a_d = a + lam[:, None, None] * diag[:, :, None] * eye[None]
        a_d += 1e-30 * eye[None]
        try:
            step = np.linalg.solve(a_d, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.zeros_like(p)
            for k in range(n):
                try:
                    step[k] = np.linalg.solve(a_d[k], b[k])
                except np.linalg.LinAlgError:
                    pass
        p_new = p + step
        p_new[:, 2] = np.clip(np.abs(p_new[:, 2]), 1e-3, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chi2_new = chi2_of(p_new)
        better = np.isfinite(chi2_new) & (chi2_new <= chi2)
        p[better] = p_new[better]
        chi2[better] = chi2_new[better]
        lam = np.where(better, lam * 0.3, lam * 10.0)
        lam = np.clip(lam, 1e-9, 1e9)

    model, jac = _model_jac(x, p)
    jw = jac * w[..., None]
    a = np.einsum("npi,npj->nij", jw, jac)
    dof = np.maximum(ngood - 4, 1)
    chi2r = chi2 / dof
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            cov = np.linalg.inv(a + 1e-30 * eye[None])
        except np.linalg.LinAlgError:
            cov = np.full_like(a, np.nan)
        err = np.sqrt(np.abs(np.einsum("nii->ni", cov))) * np.sqrt(np.maximum(chi2r, 1.0))[:, None]
    amp, mu, sig, bkg = p.T
    ok = (np.all(np.isfinite(p), axis=1) & (amp > 0) & (sig > sig_bounds[0]) & (sig < sig_bounds[1])
          & (ngood >= 5) & (mu > x[0]) & (mu < x[-1]))
    return dict(amp=amp, mu=mu, sig=sig, fwhm=sig * FWHM_PER_SIGMA, bkg=bkg,
                e_amp=err[:, 0], e_mu=err[:, 1], e_sig=err[:, 2], chi2r=chi2r, ok=ok)
