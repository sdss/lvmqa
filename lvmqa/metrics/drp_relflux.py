"""
qa_relflux_br / qa_relflux_rz - relative flux calibration: continuum continuity across the
b-r and r-z channel boundaries (DR20 logic, scripts/continuum/DRPCubeQA.py).

Per science fiber, FLUX + SKY in [blue_win[0] - margin, red_win[1] + margin] is sigma-clipped
(3 sigma around the median, up to 10 iterations) and fitted with a 2nd-order polynomial.
The jump is the difference of the median residuals in a window just blueward and just
redward of the channel boundary, in units of the larger of the two robust (MAD) scatters:

    jump = (med_res_blue - med_res_red) / max(madstd_blue, madstd_red)

Exposure: fraction of science fibers with |jump| > jump_warn (1) and > jump_bad (2).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec


def _madstd(x, axis=1):
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def clipped_poly_residuals(wave, y, order=2, sigma=3.0, maxiters=10):
    """Vectorised version of DRPCubeQA._get_flux_residual (one row per fiber)."""
    valid = np.isfinite(y)
    good = valid.copy()
    for _ in range(maxiters):
        yy = np.where(good, y, np.nan)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cen = np.nanmedian(yy, axis=1, keepdims=True)
            std = np.nanstd(yy, axis=1, keepdims=True)
        new = valid & (np.abs(y - cen) <= sigma * std)
        if np.array_equal(new, good):
            break
        good = new
    x = (wave - 0.5 * (wave[0] + wave[-1])) / (0.5 * (wave[-1] - wave[0]))
    A = np.vander(x, order + 1, increasing=True)              # (nw, k)
    W = good.astype(float)
    yz = np.where(good, y, 0.0)
    M = np.einsum("fw,wi,wj->fij", W, A, A)
    b = np.einsum("fw,wi->fi", W * yz, A)
    ok = good.sum(axis=1) > order
    coef = np.full((y.shape[0], order + 1), np.nan)
    if ok.any():
        coef[ok] = np.linalg.solve(M[ok], b[ok][..., None])[..., 0]
    return y - coef @ A.T


def _window_stats(wave, res, wlo, whi):
    """DRPCubeQA._integrated_median: argmin-based index window [li, ui)."""
    li = int(np.argmin(np.abs(wave - wlo)))
    ui = int(np.argmin(np.abs(wave - whi)))
    if ui < li:
        li, ui = ui, li
    if ui == li:
        ui = min(li + 1, res.shape[1])
    sub = res[:, li:ui]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(sub, axis=1), _madstd(sub)


class _RelFluxBase(Metric):
    group = GROUP_DRP
    requires = ("sframe_data",)
    boundary = ""
    values = (
        ValueSpec("n_fibers", "i8", "", "Science fibers (denominator, as in DR20)"),
        ValueSpec("n_valid", "i8", "", "Science fibers with a measured jump"),
        ValueSpec("frac_warn", "f8", "", "Fraction of science fibers with |jump| > jump_warn"),
        ValueSpec("frac_bad", "f8", "", "Fraction of science fibers with |jump| > jump_bad"),
        ValueSpec("median_jump", "f8", "", "Median jump (in units of the MAD scatter)"),
        ValueSpec("p95_abs_jump", "f8", "", "95th percentile of |jump|"),
    )
    main_value = "frac_warn"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"<= {p['frac_warn_thresh']:.0%} of fibers with continuum jump at the "
                     f"{self.boundary} boundary > {p['jump_warn']:g} sigma"),
            "WARNING": (f"Sky+stellar continuum jump between channels > {p['jump_warn']:g} sigma in "
                        f"> {p['frac_warn_thresh']:.0%} of fibers, but > {p['jump_bad']:g} sigma in "
                        f"<= {p['frac_bad_thresh']:.0%}"),
            "MAJOR ISSUE": (f"Continuum jump > {p['jump_bad']:g} sigma in > "
                            f"{p['frac_bad_thresh']:.0%} of fibers"),
            "UNDEFINED": "SFrame missing or no valid science fibers",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        sci = ctx.sci_fiber_index
        w0 = p["blue_win"][0] - p["fit_margin"]
        w1 = p["red_win"][1] + p["fit_margin"]
        wave, flux = ctx.sframe_window("FLUX", w0, w1, rows=sci)
        _, sky = ctx.sframe_window("SKY", w0, w1, rows=sci)
        y = flux + sky
        res = clipped_poly_residuals(wave, y, p["poly_order"], p["clip_sigma"], p["clip_iters"])
        mb, sb = _window_stats(wave, res, *p["blue_win"])
        mr, sr = _window_stats(wave, res, *p["red_win"])
        denom = np.fmax(sb, sr)
        denom[~np.isfinite(denom) | (denom == 0)] = np.nan
        jump = (mb - mr) / denom
        fin = np.isfinite(jump)
        n = int(len(sci))
        if not fin.any() or n == 0:
            return self.result(QAFlag.UNDEFINED, note="no valid fibers", n_fibers=n, n_valid=0)
        fw = float(np.sum(np.abs(jump[fin]) > p["jump_warn"])) / n
        fb = float(np.sum(np.abs(jump[fin]) > p["jump_bad"])) / n
        vals = dict(n_fibers=n, n_valid=int(fin.sum()), frac_warn=fw, frac_bad=fb,
                    median_jump=float(np.median(jump[fin])),
                    p95_abs_jump=float(np.percentile(np.abs(jump[fin]), 95)))
        if fb > p["frac_bad_thresh"]:
            return self.result(QAFlag.MAJOR, note=f"{fb:.0%} fibers jump > {p['jump_bad']:g} sigma", **vals)
        if fw > p["frac_warn_thresh"]:
            return self.result(QAFlag.WARNING, note=f"{fw:.0%} fibers jump > {p['jump_warn']:g} sigma", **vals)
        return self.result(QAFlag.GOOD, **vals)


_COMMON = dict(fit_margin=300.0, poly_order=2, clip_sigma=3.0, clip_iters=10,
               jump_warn=1.0, jump_bad=2.0, frac_warn_thresh=0.05, frac_bad_thresh=0.05)

from ..registry import register  # noqa: E402


@register
class RelFluxBR(_RelFluxBase):
    name = "qa_relflux_br"
    slot = 4
    boundary = "b-r"
    title = "Relative flux calibration: b-r boundary"
    description = "Relative flux calibration: continuity of the continuum at the b-r channel boundary"
    defaults = dict(blue_win=(5600.0, 5750.0), red_win=(5850.0, 6000.0), **_COMMON)


@register
class RelFluxRZ(_RelFluxBase):
    name = "qa_relflux_rz"
    slot = 5
    boundary = "r-z"
    title = "Relative flux calibration: r-z boundary"
    description = "Relative flux calibration: continuity of the continuum at the r-z channel boundary"
    defaults = dict(blue_win=(7300.0, 7500.0), red_win=(7650.0, 7900.0), **_COMMON)
