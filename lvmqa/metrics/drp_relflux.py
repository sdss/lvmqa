"""
qa_relflux_br / qa_relflux_rz - relative flux calibration: continuum continuity across the
b-r and r-z channel boundaries.

v1.13 (flagging): continuum continuity at the channel overlap (lvmdrp fluxCalMethod overlaps
b-r 5775-5825, r-z 7520-7580 A). Per fiber, the unmasked pixels of each side window are reduced to
10-A bin medians (robust to residual line spikes) and a straight line (level a, slope b at the
overlap centre x0) is fitted per side. Two statistics (medians over the fibers, normalised by the
fiber level (a_blue + a_red)/2):
    step           J = (a_red - a_blue) / level         jump of the continuum at x0
    slope mismatch K = (b_red - b_blue) * L / level      divergence of the two side continua L = 250 A
                                                         from x0 (a kink: opposite / different gradients)
A common gradient gives J = K = 0. Intrinsic curvature of stellar continua gives |K| <~ 0.01-0.03
(blackbodies 3500-20000 K). Errors: 1.2533 NMAD / sqrt(n_fibers).
GOOD     |J| <= 3% and |K| <= 5%, or not significant (< 3 sigma)
WARNING  |J| > 3% or |K| > 5% (significant)
MAJOR    |J| > 8% or |K| > 10% (significant)
(v1.10-1.12 used J only: a strong kink with a small step at x0, e.g. 1.2.0 7959 b-r J +1.8%,
K +14%, was GOOD; insignificant steps of faint fields, e.g. 1.3.2 5353 b-r J -3.2 +- 2.4%, were
WARNING.)
Side windows: b-r 5525-5775 | 5825-6075 A; r-z 6950-7520 | 7740-8100 A (v1.13: the r side was
7180-7520 A, of which only 7350-7520 A survived the H2O mask -> noisy, biased slopes; now 7040-7350
is masked (H2O, TiO 7054 head, HeI 7065, [ArIII] 7136), leaving 6950-7040 + 7350-7520 A; the red
side starts after the O2 A band and the bluest z pixels, whose line residuals steepened the fit).
Telluric bands are excluded (O2 A 7590-7710, H2O 7150-7350, O2 B 6860-6940 A). Two signals:
  * 'src' = FLUX of fibers with continuum S/N >= 5 (stars, bright nebulae); masked: sky lines
            (sky > 1.5 x its own running 20th percentile) and emission lines in the source stack
            (> 10% above its running 20th percentile). Adopted if >= 10 fibers.
  * 'obs' = FLUX + SKY of all good science fibers (fallback, b-r only); masked: pixels > 10%
            above the running 20th percentile of the median spectrum. Biased by the sky-continuum
            shape (b-r +0.04...+0.13, r-z +0.16...+0.37 on DRP 1.3.2), so at most WARNING at b-r
            and not used at r-z (UNDEFINED without bright fibers).

v1.15:
  * the ``edge`` = 100 A of each side window next to the boundary (b-r 5675-5775 | 5825-5925,
    r-z 7420-7520 | 7740-7840 A) are used only for the step (levels at x0: median of the 10-A
    bin medians in the zone projected to x0 with the side slope), the slopes come from the outer
    parts (b-r 5525-5675 | 5925-6075, r-z 6950-7420 | 7840-8100 A): near the boundary a step or a
    stitching artefact otherwise biases the gradients.
  * slope-mismatch veto on the stack of the adopted signal (slope_model()): straight lines through
    the 10-A bin medians of the stack in the outer windows; the scatter of the bin medians about
    them (absorption features, residual lines, outliers) gives the error of the stack slope
    mismatch. If it is not significant (|K_stack| <= stack_nsig x e_K), one common continuum slope
    (inverse-variance weighted: the cleaner side dominates) explains both sides: slope_model =
    'joint', the slope mismatch is not counted (median_kink_raw, K_stack reported) and the step is
    measured with the common slope. Typical case: one side window biased by absorption features.

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


def _side_line(w, y, m, x0, binw=10.0, min_bins=4):
    """Straight line per row through the 10-A bin medians of the unmasked pixels m (robust to
    residual line spikes): (level at x0, slope per A); NaN with < min_bins bins."""
    x, Y = _binned(w, y, m, binw)
    if x.size < min_bins:
        nan = np.full(y.shape[0], np.nan)
        return nan, nan.copy()
    x = x - x0
    ok = np.isfinite(Y)
    W = ok.astype(float)
    Y0 = np.where(ok, Y, 0.0)
    S, Sx, Sxx = W.sum(1), (W * x).sum(1), (W * x * x).sum(1)
    Sy, Sxy = Y0.sum(1), (Y0 * x).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        det = S * Sxx - Sx ** 2
        a = (Sy * Sxx - Sx * Sxy) / det
        b = (S * Sxy - Sx * Sy) / det
    bad = S < min_bins
    a[bad] = np.nan
    b[bad] = np.nan
    return a, b


def _side_fit(w, y, m, x0, binw=10.0, min_bins=4):
    """Level at x0 of the straight-line side fit (see _side_line)."""
    return _side_line(w, y, m, x0, binw, min_bins)[0]


def line_mask_from(w, G, thresh, k=61, dilate=1):
    """Pixels where the median spectrum G exceeds its running 20th percentile by > thresh."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env = _runpct(G, k)
        lm = (G / env - 1.0) > thresh
    if dilate:
        lm = np.convolve(lm.astype(float), np.ones(2 * dilate + 1), mode="same") > 0
    return lm


def fit_mask(w, line_mask, extra=()):
    """Pixels excluded from the side fits: lines + telluric bands + extra windows."""
    lm = line_mask.copy()
    for a, b in tuple(TELLURIC) + tuple(extra):
        lm |= (w >= a) & (w <= b)
    return lm


def zones(blue, red, edge):
    """Slope windows and jump zones (v1.15): the ``edge`` A of each side next to the boundary are
    used only for the levels (jump), the rest of the side window only for the slopes."""
    return dict(blue_slope=(blue[0], blue[1] - edge), red_slope=(red[0] + edge, red[1]),
                blue_zone=(blue[1] - edge, blue[1]), red_zone=(red[0], red[0] + edge))


def _common_slope(xb, Yb, xr, Yr):
    """Per row: one slope for both sides with separate intercepts (least squares on the bins)."""
    num = np.zeros(Yb.shape[0])
    den = np.zeros(Yb.shape[0])
    for x, Y in ((xb, Yb), (xr, Yr)):
        ok = np.isfinite(Y)
        W = ok.astype(float)
        n = np.maximum(W.sum(1), 1)
        xm = (W * x).sum(1) / n
        ym = np.where(ok, Y, 0.0).sum(1) / n
        dx = np.where(ok, x[None, :] - xm[:, None], 0.0)
        num += (dx * np.where(ok, Y - ym[:, None], 0.0)).sum(1)
        den += (dx ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def slope_model(w, stack, mb, mr, x0, lever=250.0, nsig=3.0, binw=10.0, min_bins=4):
    """
    Is the slope mismatch real at the level of the stacked normalised spectrum (v1.15)?
    Straight lines through the 10-A bin medians of the stack in the two outer slope windows; the
    scatter of the bin medians about each line (1.4826 MAD: absorption features, residual lines,
    outliers) gives the slope errors, hence the error of the stack slope mismatch
        K_stack = (b_r - b_b) * lever / level,  e_K = lever / level * sqrt(e_b_blue^2 + e_b_red^2).
    If |K_stack| <= nsig * e_K, one common continuum slope explains both sides within their scatter
    (typically one side's slope is biased by absorption features / outliers): model 'joint', with
    the common slope = inverse-variance weighted mean of the side slopes (weights w_blue, 1 - w_blue:
    the cleaner side dominates). Otherwise 'sides'. Returns (model, info dict).
    """
    xb, Yb = _binned(w, stack[None, :], mb, binw)
    xr, Yr = _binned(w, stack[None, :], mr, binw)
    if xb.size < min_bins or xr.size < min_bins:
        return "sides", {}
    Yb, Yr = Yb[0], Yr[0]
    okb, okr = np.isfinite(Yb), np.isfinite(Yr)
    if okb.sum() < min_bins or okr.sum() < min_bins:
        return "sides", {}
    out = {}
    for name, x, Y in (("blue", xb[okb] - x0, Yb[okb]), ("red", xr[okr] - x0, Yr[okr])):
        c = np.polyfit(x, Y, 1)
        res = Y - np.polyval(c, x)
        sig = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), 1e-4)
        sxx = float(np.sum((x - x.mean()) ** 2))
        out[name] = (float(c[0]), float(c[1]), sig, sig / np.sqrt(sxx), sxx)
    (bb, ab, sb, eb, xxb), (br, ar, sr, er, xxr) = out["blue"], out["red"]
    lev = 0.5 * (ab + ar)
    if not lev > 0:
        return "sides", {}
    K = (br - bb) * lever / lev
    eK = lever / lev * np.hypot(eb, er)
    wb = (1 / eb ** 2) / (1 / eb ** 2 + 1 / er ** 2)
    info = dict(K_stack=float(K), e_K_stack=float(eK), w_blue=float(wb), scat_blue=sb, scat_red=sr)
    return ("joint" if abs(K) <= nsig * eK else "sides"), info


def boundary_shape(w, y, overlap, blue, red, line_mask, lever=250.0, extra=(), edge=100.0, model="sides",
                   w_blue=None, binw=10.0, min_bins=4, zone_min_bins=3):
    """Per-row continuum shape at the overlap (v1.15).
    Slopes from straight lines through the 10-A bin medians of the unmasked pixels of the outer
    side windows (side window minus the ``edge`` A next to the boundary); levels at the overlap
    centre x0 from the jump zones (the ``edge`` A next to the boundary: median of the bin medians
    projected to x0 with the side slope; if a zone has < zone_min_bins bins, e.g. r-z red zone in
    the O2 A band, the side line extrapolated to x0). ``model`` (see slope_model): 'sides' = each
    side its own slope; 'joint' = one slope for both sides, w_blue x b_blue + (1 - w_blue) x b_red
    (slope mismatch 0).
      J = (a_r - a_b) / level              fractional step at x0
      K = (b_r - b_b) * lever / level      slope mismatch with the adopted model (0 unless 'sides')
      K_raw                                slope mismatch of the side lines
      G = (b_b + b_r) / 2 * 100 / level    common gradient per 100 A
    with level = (a_b + a_r) / 2. Returns dict(J, K, K_raw, G, npix_blue, npix_red, zones)."""
    z = zones(blue, red, edge)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        lm = fit_mask(w, line_mask, extra)

        def win(a):
            return (w >= a[0]) & (w <= a[1]) & ~lm

        mb, mr = win(z["blue_slope"]), win(z["red_slope"])
        zb, zr = win(z["blue_zone"]), win(z["red_zone"])
        x0 = 0.5 * (overlap[0] + overlap[1])
        (ab0, bb), (ar0, br) = _side_line(w, y, mb, x0, binw, min_bins), _side_line(w, y, mr, x0, binw, min_bins)
        k_raw_slopes = (bb, br)
        if model != "sides":
            xb, Yb = _binned(w, y, mb, binw)
            xr, Yr = _binned(w, y, mr, binw)
            wb = 0.5 if w_blue is None or not np.isfinite(w_blue) else float(w_blue)
            sl = np.where(np.isfinite(bb) & np.isfinite(br), wb * bb + (1 - wb) * br, np.where(np.isfinite(bb), bb, br))
            bb_m = br_m = sl
            # side levels at x0 with the common slope (fallback when a zone is empty)
            ab0 = np.nanmedian(Yb - sl[:, None] * (xb - x0)[None, :], axis=1) if xb.size else ab0
            ar0 = np.nanmedian(Yr - sl[:, None] * (xr - x0)[None, :], axis=1) if xr.size else ar0
        else:
            bb_m, br_m = bb, br

        def zone_level(m, slope, fallback):
            x, Y = _binned(w, y, m, binw)
            if x.size < zone_min_bins:
                return fallback
            lvl = np.nanmedian(Y - slope[:, None] * (x - x0)[None, :], axis=1)
            nb = np.isfinite(Y).sum(1)
            return np.where(nb >= zone_min_bins, lvl, fallback)

        ab = zone_level(zb, bb_m, ab0)
        ar = zone_level(zr, br_m, ar0)
        lev = 0.5 * (ab + ar)
        lev = np.where(lev > 0, lev, np.nan)
        return dict(J=(ar - ab) / lev, K=(br_m - bb_m) * lever / lev,
                    K_raw=(k_raw_slopes[1] - k_raw_slopes[0]) * lever / lev,
                    G=0.5 * (bb_m + br_m) * 100.0 / lev,
                    npix_blue=int(mb.sum() + zb.sum()), npix_red=int(mr.sum() + zr.sum()), zones=z)


def boundary_jumps(w, y, overlap, blue, red, line_mask, extra=()):
    """Per-row fractional continuum step at the overlap centre (see boundary_shape).
    Returns (J, n_pix_blue, n_pix_red)."""
    r = boundary_shape(w, y, overlap, blue, red, line_mask, extra=extra)
    return r["J"], r["npix_blue"], r["npix_red"]


def robust_median(v, min_n=5):
    """(median, error of the median = 1.2533 * NMAD / sqrt(n), n) of the finite values."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < min_n:
        return np.nan, np.nan, int(v.size)
    med = float(np.median(v))
    return med, float(1.2533 * 1.4826 * np.median(np.abs(v - med)) / np.sqrt(v.size)), int(v.size)


def classify(p, step, e_step, kink, e_kink, signal, model="sides", kink_raw=np.nan):
    """Flag from the adopted step / slope mismatch (shared by compute() and lvmqa.recalc).
    A component counts only if it exceeds its threshold AND nsig x its error. ``model`` != 'sides':
    the side slopes differ (kink_raw) but one common continuum slope explains both sides (stack),
    so the slope mismatch does not count. Returns (flag, note)."""
    def level(x, ex, warn, major):
        if not np.isfinite(x):
            return 0
        sig = not np.isfinite(ex) or abs(x) > p["nsig"] * ex
        if not sig:
            return 0
        return 2 if abs(x) > major else (1 if abs(x) > warn else 0)

    model = str(model).strip() or "sides"
    ls = level(step, e_step, p["step_warn"], p["step_major"])
    lk = level(kink, e_kink, p["kink_warn"], p["kink_major"]) if model == "sides" else 0
    lvl = max(ls, lk)
    parts = [f"step {step:+.1%}" + (f" +- {e_step:.1%}" if np.isfinite(e_step) else "")]
    if model == "sides" and np.isfinite(kink):
        parts.append(f"slope mismatch {kink:+.1%} at +-{p['lever']:.0f} A" +
                     (f" +- {e_kink:.1%}" if np.isfinite(e_kink) else ""))
    elif model != "sides":
        parts.append(f"side slopes differ by {kink_raw:+.1%} at +-{p['lever']:.0f} A, not significant against "
                     f"the scatter of the stacked spectrum (common slope adopted, not counted)")
    note = ", ".join(parts) + f" ({signal})"
    if lvl and ls == 0:
        note += "; slope mismatch"
    elif lvl and lk == 0:
        note += "; step"
    elif lvl:
        note += "; step and slope mismatch"
    if signal == "obs":
        return (QAFlag.WARNING if lvl == 2 else QAFlag.GOOD), note + " (sky-dominated signal: at most WARNING)"
    return (QAFlag.MAJOR, QAFlag.WARNING, QAFlag.GOOD)[2 - lvl], note


def _shape(w, y, stack, lm, p):
    """boundary_shape with the slope model decided on the stack (v1.15)."""
    ov, blue, red = p["overlap"], p["blue_side"], p["red_side"]
    extra = tuple(p.get("mask_windows", ()))
    edge = p.get("edge", 0.0)
    z = zones(blue, red, edge)
    x0 = 0.5 * (ov[0] + ov[1])
    model, dev = "sides", {}
    if p.get("stack_nsig", 0) and stack is not None:
        lmf = fit_mask(w, lm, extra)
        mb = (w >= z["blue_slope"][0]) & (w <= z["blue_slope"][1]) & ~lmf
        mr = (w >= z["red_slope"][0]) & (w <= z["red_slope"][1]) & ~lmf
        model, dev = slope_model(w, stack, mb, mr, x0, p.get("lever", 250.0), p["stack_nsig"])
    r = boundary_shape(w, y, ov, blue, red, lm, p.get("lever", 250.0), extra, edge, model, dev.get("w_blue"))
    r.update(model=model, dev=dev)
    return r


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
        extra = tuple(p.get("mask_windows", ()))
        so = _shape(w, obs, Go, lm_obs, p)
        out.update(J_obs=so["J"], K_obs=so["K"], Kraw_obs=so["K_raw"], G_obs=so["G"], model_obs=so["model"],
                   dev_obs=so["dev"], npix_blue=so["npix_blue"], npix_red=so["npix_red"], stack_obs=Go,
                   mask_obs=lm_obs, extra_mask=extra, zones=so["zones"])
        snr = np.nanmedian(np.where(good & ~lm_obs[None, :], f * np.sqrt(np.where(good, iv, 0.0)), np.nan), axis=1)
        src = snr >= p["min_snr_src"]
        out["J_src"] = out["K_src"] = out["Kraw_src"] = out["G_src"] = np.array([])
        if src.sum() >= 5:
            fs_ = np.where(good, f, np.nan)[src]
            norm = np.nanmedian(fs_, axis=1)[:, None]
            Gs = np.nanmedian(fs_ / norm, axis=0)
            skyrel = np.nanmedian(np.where(good, sk, np.nan)[src] / norm, axis=0)
            # src mask: sky lines (sky > (1 + mask_sky) x its own running 20th percentile: their
            # residuals add noise) and emission lines in the source stack (> mask_src above its
            # envelope); independent of the source brightness (nebular fibers in bright HII fields)
            lm_src = line_mask_from(w, skyrel, p["mask_sky"]) | line_mask_from(w, Gs, p["mask_src"])
            ss = _shape(w, fs_, Gs, lm_src, p)
            out.update(J_src=ss["J"], K_src=ss["K"], Kraw_src=ss["K_raw"], G_src=ss["G"], model_src=ss["model"],
                       dev_src=ss["dev"], npix_blue_src=ss["npix_blue"], npix_red_src=ss["npix_red"],
                       stack_src=Gs, mask_src=lm_src, n_src_sel=int(src.sum()))
    return out


class _RelFluxBase(Metric):
    group = GROUP_DRP
    requires = ("sframe_data",)
    boundary = ""
    values = (
        ValueSpec("median_step", "f8", "", "Median fractional continuum step at the overlap (adopted signal)"),
        ValueSpec("e_median_step", "f8", "", "Error of median_step (1.2533 NMAD / sqrt(n))"),
        ValueSpec("nmad_step", "f8", "", "NMAD of the per-fiber steps (adopted signal)"),
        ValueSpec("median_kink", "f8", "", "Median slope mismatch: divergence of the side continua at +-lever A (fraction)"),
        ValueSpec("e_median_kink", "f8", "", "Error of median_kink"),
        ValueSpec("median_gradient", "f8", "", "Median common continuum gradient (fraction per 100 A)"),
        ValueSpec("median_kink_raw", "f8", "", "Median slope mismatch of the individual side lines (before the slope-model veto)"),
        ValueSpec("slope_model", "U5", "", "Continuum slope model adopted on the stack: sides (own slopes) or joint "
                                           "(mismatch not significant against the bin scatter: not counted)"),
        ValueSpec("kink_stack", "f8", "", "Slope mismatch of the stacked spectrum (outer windows)"),
        ValueSpec("e_kink_stack", "f8", "", "Error of kink_stack from the scatter of the 10-A bin medians"),
        ValueSpec("signal", "U4", "", "Adopted signal: src (FLUX, fibers with continuum S/N >= 5) or obs (FLUX+SKY, all)"),
        ValueSpec("median_step_obs", "f8", "", "Median step, FLUX+SKY of all good science fibers"),
        ValueSpec("e_median_step_obs", "f8", "", "Error of median_step_obs"),
        ValueSpec("median_kink_obs", "f8", "", "Median slope mismatch, FLUX+SKY"),
        ValueSpec("e_median_kink_obs", "f8", "", "Error of median_kink_obs"),
        ValueSpec("median_step_src", "f8", "", "Median step, FLUX of fibers with continuum S/N >= 5"),
        ValueSpec("e_median_step_src", "f8", "", "Error of median_step_src"),
        ValueSpec("median_kink_src", "f8", "", "Median slope mismatch, FLUX of fibers with continuum S/N >= 5"),
        ValueSpec("e_median_kink_src", "f8", "", "Error of median_kink_src"),
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
        msk = "".join(f", {a:.0f}-{b:.0f}" for a, b in p.get("mask_windows", ()))
        return {
            "GOOD": (f"|step| <= {p['step_warn']:.0%} and |slope mismatch| <= {p['kink_warn']:.0%}, or not significant "
                     f"(< {p['nsig']:g} sigma). Step = jump of the continuum at the {self.boundary} overlap centre, "
                     f"slope mismatch = divergence of the two side continua {p['lever']:.0f} A from it; straight lines "
                     f"through 10-A bin medians in {p['blue_side'][0]:.0f}-{p['blue_side'][1]:.0f} and "
                     f"{p['red_side'][0]:.0f}-{p['red_side'][1]:.0f} A (lines, tellurics{msk} masked), medians over "
                     f"fibers with S/N >= {p['min_snr_src']:g} (else FLUX+SKY of all fibers, then at most WARNING)"),
            "WARNING": f"|step| > {p['step_warn']:.0%} or |slope mismatch| > {p['kink_warn']:.0%} (significant)",
            "MAJOR ISSUE": f"|step| > {p['step_major']:.0%} or |slope mismatch| > {p['kink_major']:.0%} (significant)",
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
        vals.update({k: r_[k] for k in ("npix_blue", "npix_red", "npix_blue_src", "npix_red_src") if k in r_})
        nb, nr = r_["npix_blue"], r_["npix_red"]
        Jo, Js = r_["J_obs"], r_["J_src"]
        fo, fs = np.isfinite(Jo), np.isfinite(Js)
        if fo.sum() >= p["min_fibers"] and min(nb, nr) >= p["min_pix"]:
            m_, e_, _ = robust_median(Jo)
            k_, ek_, _ = robust_median(r_["K_obs"])
            vals.update(median_step_obs=m_, e_median_step_obs=e_, median_kink_obs=k_, e_median_kink_obs=ek_)
        if fs.sum() >= 5:
            m_, e_, n_ = robust_median(Js)
            k_, ek_, _ = robust_median(r_["K_src"])
            vals.update(median_step_src=m_, e_median_step_src=e_, median_kink_src=k_, e_median_kink_src=ek_,
                        n_src=n_)
        if fs.sum() >= p["min_src"]:
            sig = "src"
        elif "median_step_obs" in vals and p["obs_fallback"]:
            sig = "obs"
        else:
            return self.result(QAFlag.UNDEFINED, note="too few continuum pixels / fibers", **vals)
        J, K, Kr, G, model = r_[f"J_{sig}"], r_[f"K_{sig}"], r_[f"Kraw_{sig}"], r_[f"G_{sig}"], r_[f"model_{sig}"]
        med, e_med, _ = robust_median(J)
        kin, e_kin, _ = robust_median(K)
        gra = robust_median(G)[0]
        Jf = J[np.isfinite(J)]
        vals.update(median_step=med, e_median_step=e_med, nmad_step=float(1.4826 * np.median(np.abs(Jf - med))),
                    median_kink=kin, e_median_kink=e_kin, median_gradient=gra, signal=sig,
                    median_kink_raw=robust_median(Kr)[0], slope_model=model,
                    kink_stack=r_.get(f"dev_{sig}", {}).get("K_stack", np.nan),
                    e_kink_stack=r_.get(f"dev_{sig}", {}).get("e_K_stack", np.nan))
        flag, note = classify(p, med, e_med, kin, e_kin, sig, model, vals["median_kink_raw"])
        return self.result(flag, note=note, **vals)


_COMMON = dict(fit_margin=300.0, poly_order=2, clip_sigma=3.0, clip_iters=10,
               jump_warn=1.0, jump_bad=2.0, frac_warn_thresh=0.05, frac_bad_thresh=0.05,
               step_warn=0.03, step_major=0.08, min_snr_src=5.0, min_src=10, min_fibers=50, min_pix=40,
               kink_warn=0.05, kink_major=0.10,  # slope mismatch (divergence of the side continua at +-lever A)
               lever=250.0, nsig=3.0,            # lever arm [A]; significance of step / slope mismatch
               edge=100.0,      # v1.15: A next to the boundary used only for the levels (step), not the slopes
               stack_nsig=2.0,  # v1.15: slope mismatch must also be significant on the stack (slope_model); 0 = off
               mask_windows=(),                  # extra masked wavelength windows (A)
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
                    # v1.13: r side 6950-7520 A with 7040-7350 masked (H2O 7150-7350 is masked anyway; TiO
                    # 7054 head, HeI 7065, [ArIII] 7136): 6950-7040 + 7350-7520 A, i.e. ~2.3x more pixels
                    # and a longer lever arm than 7350-7520 A alone (slope errors halved)
                    blue_side=(6950.0, 7520.0), red_side=(7740.0, 8100.0),
                    **{**_COMMON, "obs_fallback": False, "mask_windows": ((7040.0, 7350.0),)})
