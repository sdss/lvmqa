"""
qa_relflux_br / qa_relflux_rz - relative flux calibration: continuum continuity across the
b-r and r-z channel boundaries.

v1.9 (flagging): step of the continuum at the channel overlap (lvmdrp fluxCalMethod overlaps
b-r 5775-5825, r-z 7520-7580 A) as a fraction of the continuum. Per fiber, the unmasked pixels of
each side window are reduced to 10-A bin medians (robust to residual line spikes) and a straight
line is fitted per side and extrapolated to the overlap centre:
    J = (L_red - L_blue) / mean(L_red, L_blue)
Side windows: b-r 5525-5775 | 5825-6075 A; r-z 7180-7520 | 7740-8100 A (the red r-z side starts
after the O2 A band and after the bluest z pixels, whose line residuals steepened the fit).
Telluric bands are excluded (O2 A 7590-7710, H2O 7150-7350 A). Two signals:
  * 'src' = FLUX of fibers with continuum S/N >= 5 (stars, bright nebulae); masked: sky lines
            (sky > 1.5 x its own running 20th percentile) and emission lines in the source stack
            (> 10% above its running 20th percentile). Adopted if >= 10 fibers.
  * 'obs' = FLUX + SKY of all good science fibers (fallback, b-r only); masked: pixels > 10%
            above the running 20th percentile of the median spectrum. Biased by the sky-continuum
            shape (b-r +0.04...+0.13, r-z +0.16...+0.37 on DRP 1.3.2), so at most WARNING at b-r
            and not used at r-z (UNDEFINED without bright fibers).
GOOD |median J| <= 3%, WARNING 3-8%, MAJOR > 8%.
(v1.8 used 5% masks on the obs stack for both signals and pixel-level fits: only ~30 red r-z
pixels survived.) DRP 1.2.0: b-r steps -21 ... +17% (sign varies per exposure); DRP 1.3.2:
b-r -2 ... +5% (tile 11111 ~+10%).

DR20 statistic (still reported, values *_dr20; scripts/continuum/DRPCubeQA.py):

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


TELLURIC = ((7590.0, 7710.0), (7150.0, 7350.0), (6860.0, 6940.0))


def _runpct(a, k, q=20):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(sliding_window_view(pad, k), q, axis=1)


def _binned(w, y, m, binw):
    """Per-row medians of the unmasked pixels m in bins of binw A -> (bin centres, (nrow, nbin))."""
    edges = np.arange(w[m].min(), w[m].max() + binw, binw) if m.any() else np.array([0.0, 1.0])
    xc, cols = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = m & (w >= a) & (w < b)
        if sel.sum() < 3:
            continue
        xc.append(np.median(w[sel]))
        cols.append(np.nanmedian(y[:, sel], axis=1))
    if not cols:
        return np.zeros(0), np.zeros((y.shape[0], 0))
    return np.array(xc), np.vstack(cols).T


def _side_fit(w, y, m, x0, binw=10.0, min_bins=4):
    """Straight line per row through the 10-A bin medians of the unmasked pixels m (robust to
    residual line spikes); value at x0 (NaN with < min_bins bins)."""
    x, Y = _binned(w, y, m, binw)
    if x.size < min_bins:
        return np.full(y.shape[0], np.nan)
    x = x - x0
    ok = np.isfinite(Y)
    W = ok.astype(float)
    Y0 = np.where(ok, Y, 0.0)
    S, Sx, Sxx = W.sum(1), (W * x).sum(1), (W * x * x).sum(1)
    Sy, Sxy = Y0.sum(1), (Y0 * x).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        det = S * Sxx - Sx ** 2
        a = (Sy * Sxx - Sx * Sxy) / det
    a[S < min_bins] = np.nan
    return a


def line_mask_from(w, G, thresh, k=61, dilate=1):
    """Pixels where the median spectrum G exceeds its running 20th percentile by > thresh."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env = _runpct(G, k)
        lm = (G / env - 1.0) > thresh
    if dilate:
        lm = np.convolve(lm.astype(float), np.ones(2 * dilate + 1), mode="same") > 0
    return lm


def boundary_jumps(w, y, overlap, blue, red, line_mask):
    """Per-row fractional continuum step at the overlap centre: straight lines through the bin
    medians of the unmasked pixels in the blue and red side windows, extrapolated to the overlap
    centre. Returns (J, n_pix_blue, n_pix_red)."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        lm = line_mask.copy()
        for a, b in TELLURIC:
            lm |= (w >= a) & (w <= b)
        mb = (w >= blue[0]) & (w <= blue[1]) & ~lm
        mr = (w >= red[0]) & (w <= red[1]) & ~lm
        x0 = 0.5 * (overlap[0] + overlap[1])
        lb, lr = _side_fit(w, y, mb, x0), _side_fit(w, y, mr, x0)
        J = (lr - lb) / (0.5 * (lr + lb))
    return J, int(mb.sum()), int(mr.sum())


def measure_steps(ctx, p):
    """Shared by the metric and the diagnostics: per-fiber steps for 'obs' and 'src' signals."""
    ov, blue, red = p["overlap"], p["blue_side"], p["red_side"]
    rows = ctx.good_sci_index
    w, f = ctx.sframe_window("FLUX", blue[0], red[1], rows=rows)
    _, sk = ctx.sframe_window("SKY", blue[0], red[1], rows=rows)
    _, iv = ctx.sframe_window("IVAR", blue[0], red[1], rows=rows)
    good = np.isfinite(iv) & (iv > 0)
    out = dict(w=w)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        obs = np.where(good, f + sk, np.nan)
        Go = np.nanmedian(obs / np.nanmedian(obs, axis=1)[:, None], axis=0)
        lm_obs = line_mask_from(w, Go, p["mask_obs"])
        Jo, nb, nr = boundary_jumps(w, obs, ov, blue, red, lm_obs)
        out.update(J_obs=Jo, npix_blue=nb, npix_red=nr, stack_obs=Go, mask_obs=lm_obs)
        snr = np.nanmedian(np.where(good & ~lm_obs[None, :], f * np.sqrt(np.where(good, iv, 0.0)), np.nan), axis=1)
        src = snr >= p["min_snr_src"]
        out["J_src"] = np.array([])
        if src.sum() >= 5:
            fs_ = np.where(good, f, np.nan)[src]
            norm = np.nanmedian(fs_, axis=1)[:, None]
            Gs = np.nanmedian(fs_ / norm, axis=0)
            skyrel = np.nanmedian(np.where(good, sk, np.nan)[src] / norm, axis=0)
            # src mask: sky lines (sky > (1 + mask_sky) x its own running 20th percentile: their
            # residuals add noise) and emission lines in the source stack (> mask_src above its
            # envelope); independent of the source brightness (nebular fibers in bright HII fields)
            lm_src = line_mask_from(w, skyrel, p["mask_sky"]) | line_mask_from(w, Gs, p["mask_src"])
            Js, nbs, nrs = boundary_jumps(w, fs_, ov, blue, red, lm_src)
            out.update(J_src=Js, npix_blue_src=nbs, npix_red_src=nrs, stack_src=Gs, mask_src=lm_src,
                       n_src_sel=int(src.sum()))
    return out


class _RelFluxBase(Metric):
    group = GROUP_DRP
    requires = ("sframe_data",)
    boundary = ""
    values = (
        ValueSpec("median_step", "f8", "", "Median fractional continuum step at the overlap (adopted signal)"),
        ValueSpec("nmad_step", "f8", "", "NMAD of the per-fiber steps (adopted signal)"),
        ValueSpec("signal", "U4", "", "Adopted signal: src (FLUX, fibers with continuum S/N >= 5) or obs (FLUX+SKY, all)"),
        ValueSpec("median_step_obs", "f8", "", "Median step, FLUX+SKY of all good science fibers"),
        ValueSpec("median_step_src", "f8", "", "Median step, FLUX of fibers with continuum S/N >= 5"),
        ValueSpec("n_src", "i8", "", "Fibers used for median_step_src"),
        ValueSpec("npix_blue", "i8", "", "Unmasked pixels blueward of the overlap (obs)"),
        ValueSpec("npix_red", "i8", "", "Unmasked pixels redward of the overlap (obs)"),
        ValueSpec("npix_blue_src", "i8", "", "Unmasked pixels blueward of the overlap (src)"),
        ValueSpec("npix_red_src", "i8", "", "Unmasked pixels redward of the overlap (src)"),
        ValueSpec("n_fibers", "i8", "", "Science fibers (denominator, as in DR20)"),
        ValueSpec("n_valid", "i8", "", "Science fibers with a measured DR20 jump"),
        ValueSpec("frac_warn", "f8", "", "DR20: fraction of science fibers with |jump| > jump_warn"),
        ValueSpec("frac_bad", "f8", "", "DR20: fraction of science fibers with |jump| > jump_bad"),
        ValueSpec("median_jump", "f8", "", "DR20: median jump (in units of the MAD scatter)"),
        ValueSpec("p95_abs_jump", "f8", "", "DR20: 95th percentile of |jump|"),
    )
    main_value = "median_step"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"|median fractional continuum step at the {self.boundary} overlap "
                     f"{p['overlap'][0]:.0f}-{p['overlap'][1]:.0f} A| <= {p['step_warn']:.0%} (straight lines through "
                     f"10-A bin medians in {p['blue_side'][0]:.0f}-{p['blue_side'][1]:.0f} and "
                     f"{p['red_side'][0]:.0f}-{p['red_side'][1]:.0f} A, lines/tellurics masked; FLUX of fibers with "
                     f"S/N >= {p['min_snr_src']:g}, else FLUX+SKY of all fibers, then at most WARNING)"),
            "WARNING": f"{p['step_warn']:.0%} < |median step| <= {p['step_major']:.0%}",
            "MAJOR ISSUE": f"|median step| > {p['step_major']:.0%}",
            "UNDEFINED": "SFrame missing or too few unmasked continuum pixels / fibers",
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
        # ---- v1.9 fractional continuum step
        r_ = measure_steps(ctx, p)
        Jo, Js = r_["J_obs"], r_["J_src"]
        vals.update({k: r_[k] for k in ("npix_blue", "npix_red", "npix_blue_src", "npix_red_src") if k in r_})
        nb, nr = r_["npix_blue"], r_["npix_red"]
        fo, fs = np.isfinite(Jo), np.isfinite(Js)
        if fo.sum() >= p["min_fibers"] and min(nb, nr) >= p["min_pix"]:
            vals["median_step_obs"] = float(np.median(Jo[fo]))
        if fs.sum() >= 5:
            vals.update(median_step_src=float(np.median(Js[fs])), n_src=int(fs.sum()))
        if fs.sum() >= p["min_src"]:
            J, sig = Js[fs], "src"
        elif "median_step_obs" in vals and p["obs_fallback"]:
            J, sig = Jo[fo], "obs"
        else:
            return self.result(QAFlag.UNDEFINED, note="too few continuum pixels / fibers", **vals)
        med = float(np.median(J))
        vals.update(median_step=med, nmad_step=float(1.4826 * np.median(np.abs(J - med))), signal=sig)
        note = f"step {med:+.1%} ({sig})"
        if sig == "obs":
            flag = QAFlag.WARNING if abs(med) > p["step_major"] else QAFlag.GOOD
            return self.result(flag, note=note + " (sky-dominated signal: at most WARNING)", **vals)
        if abs(med) > p["step_major"]:
            return self.result(QAFlag.MAJOR, note=note, **vals)
        if abs(med) > p["step_warn"]:
            return self.result(QAFlag.WARNING, note=note, **vals)
        return self.result(QAFlag.GOOD, note=note, **vals)


_COMMON = dict(fit_margin=300.0, poly_order=2, clip_sigma=3.0, clip_iters=10,
               jump_warn=1.0, jump_bad=2.0, frac_warn_thresh=0.05, frac_bad_thresh=0.05,
               step_warn=0.03, step_major=0.08, min_snr_src=5.0, min_src=10, min_fibers=50, min_pix=40,
               mask_obs=0.10,   # obs: pixels > 10% above the running 20th percentile of the median spectrum
               mask_src=0.10,   # src: emission lines > 10% above the envelope of the source stack
               mask_sky=0.5,    # src: sky-line pixels (sky > 1.5 x its own continuum envelope)
               obs_fallback=True)  # without enough src fibers: FLUX+SKY signal (at most WARNING)

from ..registry import register  # noqa: E402


@register
class RelFluxBR(_RelFluxBase):
    name = "qa_relflux_br"
    slot = 4
    boundary = "b-r"
    title = "Relative flux calibration: b-r boundary"
    description = "Relative flux calibration: continuity of the continuum at the b-r channel boundary"
    defaults = dict(blue_win=(5600.0, 5750.0), red_win=(5850.0, 6000.0), overlap=(5775.0, 5825.0),
                    blue_side=(5525.0, 5775.0), red_side=(5825.0, 6075.0), **_COMMON)


@register
class RelFluxRZ(_RelFluxBase):
    name = "qa_relflux_rz"
    slot = 5
    boundary = "r-z"
    title = "Relative flux calibration: r-z boundary"
    description = "Relative flux calibration: continuity of the continuum at the r-z channel boundary"
    defaults = dict(blue_win=(7300.0, 7500.0), red_win=(7650.0, 7900.0), overlap=(7520.0, 7580.0),
                    # red side starts after the O2 A band and the bluest z pixels (steepened by line
                    # residuals); wider than b-r to keep enough continuum between the OH lines
                    # FLUX+SKY is biased by +0.2...+0.35 over these windows (sky-continuum shape):
                    # no fallback, UNDEFINED without bright fibers
                    blue_side=(7180.0, 7520.0), red_side=(7740.0, 8100.0), **{**_COMMON, "obs_fallback": False})
