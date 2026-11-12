"""
Diagnostics of an existing QA run.

1) Diagnostic PDF (make_pdf): for every metric that analyses spectra, up to ``n`` random exposures
   per status (MAJOR ISSUE, WARNING, GOOD) with their SFrame / DAP / DAP-output data and models in
   the relevant wavelength range (one page per metric), each followed by the distributions of the
   metric's per-exposure values with the adopted limits and statistics (v1.15, lvmqa.histpages;
   histogram pages of the metrics without example spectra follow); then the qa_fluxmap and
   qa_gradient map pages and their histograms. Written automatically at the end of every pipeline run
   (``python -m lvmqa ...``; --no-pdf to skip).

2) CLI = re-derivation of the flags from the stored measurements (no metric is recomputed):

    python -m lvmqa.diagnostics DRPVER [DAPVER] [--tag T] [--output-dir D] [--set METRIC.PARAM=VALUE ...]
                                [--pdf [--n 5] [--seed 1] [-j 3]] [--check] [--list-params]

   reads <base>.fits / <base>_supl.fits, re-classifies with the current thresholds (+ --set),
   recomputes summaries, qa_overall, qa_bitmask, and writes <base>_recalculated.{fits,xlsx},
   _supl.{fits,xlsx}, .html and _changes.csv. The PDF (which re-reads the data of the plotted
   exposures) is regenerated only with --pdf. Same as ``python -m lvmqa.recalc``.
"""

from __future__ import annotations

import argparse
import logging
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import PipelineConfig
from .context import ExposureContext
from .output import output_basename

log = logging.getLogger("lvmqa")
C_KMS = 299792.458
CLASSES = ("MAJOR ISSUE", "WARNING", "GOOD")
CCOL = {"MAJOR ISSUE": "#d03b3b", "WARNING": "#fab219", "GOOD": "#0ca30c"}
SPEC_COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}

# ----------------------------------------------------------------------------- helpers


def _s(v, default=""):
    v = str(v).strip()
    return default if v in ("", "--", "nan", "None") else v


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _nanmed(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _panel(title, xlabel="Wavelength [A]", ylabel="", **kw):
    p = dict(title=title, xlabel=xlabel, ylabel=ylabel, curves=[], vlines=[], hlines=[], spans=[], text="")
    p.update(kw)
    return p


# ----------------------------------------------------------------------------- windows of the DAP output
OUT_WIN = {
    "Halpha": (6530.0, 6600.0), "Hbeta": (4830.0, 4890.0), "Hgamma": (4310.0, 4370.0),
    "OIII5007": (4980.0, 5030.0), "NII6583": (6560.0, 6610.0), "SII6716": (6690.0, 6750.0),
    "SII6731": (6700.0, 6760.0), "OII3727": (3700.0, 3755.0), "SIII9069": (9040.0, 9100.0),
    "SIII9531": (9500.0, 9560.0), "P9": (9200.0, 9260.0), "HeI5876": (5850.0, 5900.0),
    "HeI6678": (6650.0, 6705.0), "HeI7065": (7040.0, 7090.0),
}


def _out_windows_for(metric: str) -> Dict[str, tuple]:
    from .metrics import dap_absorption, dap_npfluxes, drp_telluric
    allp = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    if metric in ("qa_fit_consistency", "qa_dap_pmfit", "qa_gaskinematics"):
        return {f"dg_{k}": (w0, w1, allp) for k, (w0, w1) in OUT_WIN.items()}
    if metric == "qa_dap_npfluxes":
        return {f"dg_fl_{k}": (w[0] - 25, w[1] + 25, (3, 4, 6)) for k, (_, w) in dap_npfluxes.GROUPS.items()}
    if metric == "qa_absorption":
        return {f"dg_abs_{k}": (b[0][0] - 15, b[2][1] + 15, (2, 5, 6)) for k, b in dap_absorption.FEATURES.items()}
    if metric == "qa_telluric":
        out = {f"dg_tell_{k}": (cl[0] - 5, cr[1] + 5, (1, 2, 5, 6)) for k, (b, cl, cr) in drp_telluric.BANDS.items()}
        out.update(drp_telluric.Telluric().output_windows())     # metric's own windows (EW_dap if not stored)
        return out
    return {}


# ----------------------------------------------------------------------------- extractors
def ex_skyline(ctx, r, cls):
    from .metrics.drp_skyline import SKY_LINES
    sp = int(_f(r.get("skyline_sub_worst_spec", np.nan)) or 1) if np.isfinite(_f(r.get("skyline_sub_worst_spec", np.nan))) else 1
    best, bv = "O2", -1
    for k in ("OH6865", "NaD", "O2", "OI5577"):
        v = abs(_f(r.get(f"skyline_sub_{k}_sp{sp}", np.nan)))
        if np.isfinite(v) and v > bv:
            best, bv = k, v
    (l0, l1), (c0, c1) = SKY_LINES[best]
    rows = ctx.good_sci_index
    rows = rows[np.asarray(ctx.slitmap["spectrographid"])[rows] == sp]
    w, f = ctx.sframe_window("FLUX", c0 - 10, c1 + 10, rows=rows)
    _, s = ctx.sframe_window("SKY", c0 - 10, c1 + 10, rows=rows)
    p = _panel(f"sp{sp} {best}: resid/sky {_f(r.get(f'skyline_sub_{best}_sp{sp}', np.nan)):+.2f}",
               ylabel="flux / max(sky)")
    sk, fl = _nanmed(s), _nanmed(f)
    norm = np.nanmax(np.abs(sk)) or 1.0
    p["curves"] += [(w, sk / norm, "sky model", dict(color="0.6")), (w, fl / norm, "residual (FLUX)", dict(color="k"))]
    p["spans"].append((l0, l1))
    p["hlines"].append(0.0)
    return p


def ex_skycont(ctx, r, cls):
    """Per-pixel S/N of the faintest 20% fibers (ranked as in the metric: by the continuum S/N in
    the other two channels), per spectrograph, in the channel with the worst floor."""
    from .metrics.drp_skycont import WINDOWS, in_window, span
    fr = 0.2
    allf = {}
    for c in WINDOWS:
        for s_ in (1, 2, 3):
            v = _f(r.get(f"skycont_sub_floor_sp{s_}_{c}", np.nan))
            if np.isfinite(v):
                allf[(c, s_)] = v
    for c in WINDOWS:
        v = _f(r.get(f"skycont_sub_floor_{c}", np.nan))
        if np.isfinite(v):
            allf[(c, 0)] = v
    if not allf:
        return _panel("no floors")
    ch = max(allf, key=lambda k: abs(allf[k]))[0]
    rows = ctx.good_sci_index
    spec = np.asarray(ctx.slitmap["spectrographid"])[rows]
    fib, pix = {}, {}
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        for c, win in WINDOWS.items():
            w0, w1 = span(win)
            w, f = ctx.sframe_window("FLUX", w0 - 15, w1 + 15, rows=rows)
            _, iv = ctx.sframe_window("IVAR", w0 - 15, w1 + 15, rows=rows)
            _, m = ctx.sframe_window("MASK", w0 - 15, w1 + 15, rows=rows)
            good = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (m == 0)
            used = in_window(w, win)
            sig = np.where(good, 1.0 / np.sqrt(np.where(good, iv, 1.0)), np.nan)
            fib[c] = (np.nanmedian(np.where(good & used, f, np.nan), axis=1) /
                      np.nanmedian(np.where(used, sig, np.nan), axis=1))
            if c == ch:
                pix = (w, np.where(good, f, np.nan) / sig)
                from .metrics.drp_skycont import ranges
                spans = ranges(win)
        rank = np.nanmean(np.vstack([fib[o] for o in WINDOWS if o != ch]), axis=0)
    w0, w1 = span(WINDOWS[ch])
    p = _panel(f"{ch} {w0:.0f}-{w1:.0f}: floor " + " ".join(
        f"sp{k} {allf.get((ch, k), np.nan):+.2f}" for k in (1, 2, 3)), ylabel="S/N per pixel (faint fibers)")
    w, sn = pix
    for k in (1, 2, 3):
        ok = (spec == k) & np.isfinite(rank) & np.isfinite(fib[ch])
        if ok.sum() < 5:
            continue
        q = ok & (rank <= np.percentile(rank[ok], 100 * fr))
        y = np.nan_to_num(_nanmed(sn[q]))
        p["curves"].append((w, np.convolve(y, np.ones(5) / 5, mode="same"), f"sp{k} (n={q.sum()})",
                            dict(color=SPEC_COL[k])))
    p["hlines"] += [0.0, 1.0, -1.0, -3.0]
    p["spans"] += list(spans)
    return p


def ex_flatfield(ctx, r, cls):
    from .metrics.drp_flatfield import SKY_LINES
    spch = _s(r.get("flatfield_worst_sp_ch", ""), "sp1-r")
    ch = spch.split("-")[-1]
    name = next((n for n, (wl, c) in SKY_LINES.items() if c == ch), list(SKY_LINES)[0])
    wl = SKY_LINES[name][0]
    pr = _minst("qa_flatfield", ctx.cfg).params
    rows = ctx.good_sci_index
    spec = np.asarray(ctx.slitmap["spectrographid"])[rows]
    co = pr["cont_outer"]
    w, f = ctx.sframe_window("FLUX", wl - co - 1, wl + co + 1, rows=rows)
    _, s = ctx.sframe_window("SKY", wl - co - 1, wl + co + 1, rows=rows)
    y = f + s
    # local continuum per fiber (median of the side bands used by the metric) subtracted, as in the
    # metric: the profiles compare the sky-LINE fluxes of the spectrographs only
    d = np.abs(w - wl)
    cont = (d >= pr["cont_inner"]) & (d <= co)
    y = y - _nanmed(y[:, cont], axis=1)[:, None]
    ref = _nanmed(y)
    p = _panel(f"{name}: worst {spch} {_f(r.get('flatfield_max_rel_offset', np.nan)):.1f}% (local continuum "
               f"subtracted)", ylabel="sky line - continuum, / peak of all fibers")
    for sp in (1, 2, 3):
        m = spec == sp
        if m.sum():
            p["curves"].append((w, _nanmed(y[m]) / np.nanmax(ref), f"sp{sp} {_f(r.get(f'flatfield_off_sp{sp}_{ch}', np.nan)):+.1f}%",
                                dict(color=SPEC_COL[sp])))
    p["spans"] += [(wl - co, wl - pr["cont_inner"], "0.85"), (wl + pr["cont_inner"], wl + co, "0.85"),
                   (wl - pr["half_width"], wl + pr["half_width"])]
    p["hlines"].append(0.0)
    return p


def _ex_relflux(ctx, r, cls, key, metric):
    """Median-stacked normalised continuum around the channel overlap (adopted signal), masked
    pixels in grey, 10-A bin medians, straight lines fitted to the outer slope windows of each side
    (solid over the side, dashed across the other side; v1.15: the 100-A jump zones next to the
    boundary, darker shading, enter only the levels), and the common-slope continuum when the
    slope-model veto applies (the metric fits each fiber; shown here for the stack)."""
    from .metrics import drp_relflux as rf
    pr = _minst(metric, ctx.cfg).params
    ov, blue, red = pr["overlap"], pr["blue_side"], pr["red_side"]
    res = rf.measure_steps(ctx, pr)
    w = res["w"]
    sig = _s(r.get(f"relflux_{key}_signal", ""), "obs")
    if sig == "src" and "stack_src" in res:
        stack, lm, n = res["stack_src"], res["mask_src"].copy(), res.get("n_src_sel", 0)
    else:
        sig = "obs"
        stack, lm, n = res["stack_obs"], res["mask_obs"].copy(), "all"
    model = res.get(f"model_{sig}", "sides")
    lm = rf.fit_mask(w, lm, pr.get("mask_windows", ()))
    z = rf.zones(blue, red, pr.get("edge", 0.0))
    x0 = 0.5 * (ov[0] + ov[1])
    g = lambda k: _f(r.get(f"relflux_{key}_{k}", np.nan))  # noqa: E731
    kt = (f"slope mismatch {g('median_kink'):+.1%}\u00b1{g('e_median_kink'):.1%}" if model == "sides" else
          f"side slopes differ {g('median_kink_raw'):+.1%} (stack {g('kink_stack'):+.1%}±{g('e_kink_stack'):.1%}: "
          f"not significant, common slope)")
    p = _panel(f"{key}: step {g('median_step'):+.1%}\u00b1{g('e_median_step'):.1%}, {kt} at \u00b1{pr['lever']:.0f} A\n"
               f"({sig}, n={n}; common gradient {g('median_gradient'):+.1%}/100 A)",
               ylabel="median normalised flux (src: FLUX of bright fibers; obs: FLUX+SKY)")
    p["curves"].append((w, np.where(~lm, stack, np.nan), "unmasked pixels", dict(color="k", lw=0.6)))
    p["curves"].append((w, np.where(lm, stack, np.nan), "masked (lines, tellurics)", dict(color="0.78", lw=0.4)))
    fits_y, coefs = [], {}
    for name, sw, zw, col, lab in (("blue", z["blue_slope"], z["blue_zone"], "#2a78d6", "blue side"),
                                   ("red", z["red_slope"], z["red_zone"], "#d03b3b", "red side")):
        msk = (w >= sw[0]) & (w <= sw[1]) & ~lm
        mz = (w >= zw[0]) & (w <= zw[1]) & ~lm
        xb, yb = rf._binned(w, stack[None, :], msk, 10.0)
        xz, yz = rf._binned(w, stack[None, :], mz, 10.0)
        if xz.size:
            p["curves"].append((xz, yz[0], None, dict(color=col, ls="none", marker="s", ms=2.5, mfc="none")))
        if xb.size >= 4:
            yb = yb[0]
            c = np.polyfit(xb - x0, yb, 1)
            coefs[name] = (c, xb - x0, yb)
            p["curves"].append((xb, yb, f"{lab}: bin medians (squares: jump zone)", dict(color=col, ls="none", marker="o", ms=2.5)))
            side = blue if name == "blue" else red
            xs = np.array([side[0], x0]) if name == "blue" else np.array([x0, side[1]])
            xo = np.array([x0, red[1]]) if name == "blue" else np.array([blue[0], x0])
            p["curves"].append((xs, np.polyval(c, xs - x0), f"{lab} fit", dict(color=col, lw=1.4)))
            p["curves"].append((xo, np.polyval(c, xo - x0), None, dict(color=col, lw=0.9, ls="--")))
            fits_y += list(np.polyval(c, xs - x0)) + list(yb)
    if model != "sides" and len(coefs) == 2:
        (cb, xb_, yb_), (cr, xr_, yr_) = coefs["blue"], coefs["red"]
        wb = res.get(f"dev_{sig}", {}).get("w_blue", 0.5)
        sl = wb * cb[0] + (1 - wb) * cr[0]
        for xx, yy, ls_ in ((xb_, yb_, "-"), (xr_, yr_, "-")):
            a = np.median(yy - sl * xx)
            xl = np.array([blue[0], red[1]]) - x0
            p["curves"].append((xl + x0, a + sl * xl, None, dict(color="#1baf7a", lw=0.8, ls=":")))
        p["curves"][-1] = (p["curves"][-1][0], p["curves"][-1][1], f"common slope (w_blue {wb:.2f})", p["curves"][-1][3])
    p["spans"] += [tuple(ov), (z["blue_slope"][0], z["blue_slope"][1], "#cfe0f5"), (z["red_slope"][0], z["red_slope"][1], "#f6d5d2"),
                   (z["blue_zone"][0], z["blue_zone"][1], "#9fbfe8"), (z["red_zone"][0], z["red_zone"][1], "#eaa9a2")]
    # common y range 0.8-1.2, extended only if the fitted lines / bin medians fall outside
    lo, hi = 0.8, 1.2
    if fits_y:
        lo, hi = min(lo, np.nanmin(fits_y) - 0.02), max(hi, np.nanmax(fits_y) + 0.02)
    p["ylim"] = (lo, hi)

    p["vlines"].append(x0)
    return p


def ex_relflux_br(ctx, r, cls):
    return _ex_relflux(ctx, r, cls, "br", "qa_relflux_br")


def ex_relflux_rz(ctx, r, cls):
    return _ex_relflux(ctx, r, cls, "rz", "qa_relflux_rz")


def _ex_wavecal(ctx, r, cls, ch):
    from .metrics._skylines import SKY_LINES, lines_of
    excl = tuple(_minst(f"qa_wavecal_{ch}", ctx.cfg).params.get("exclude", ()))
    names = [n for n in lines_of(ch) if n not in excl]
    offs = {n: _f(r.get(f"wavecal_{ch}_off_{n}", np.nan)) for n in names}
    name = max(offs, key=lambda n: abs(offs[n]) if np.isfinite(offs[n]) else -1)
    rest = SKY_LINES[name][0]
    rows = ctx.good_sci_index
    spec = np.asarray(ctx.slitmap["spectrographid"])[rows]
    w, f = ctx.sframe_window("FLUX", rest - 4, rest + 4, rows=rows)
    _, s = ctx.sframe_window("SKY", rest - 4, rest + 4, rows=rows)
    _, lsf = ctx.sframe_window("LSF", rest - 1, rest + 1, rows=rows)
    y = f + s
    p = _panel(f"{ch} {name}: median offset {offs[name]:+.3f} A; frac>LSF/5 {_f(r.get(f'wavecal_{ch}_frac_warn', np.nan)):.3f}",
               ylabel="normalised sky line")
    for sp in (1, 2, 3):
        m = spec == sp
        if m.sum():
            prof = _nanmed(y[m])
            p["curves"].append((w, prof / np.nanmax(prof), f"sp{sp}", dict(color=SPEC_COL[sp])))
    L = float(np.nanmedian(lsf))
    p["vlines"].append(rest)
    p["spans"].append((rest - L / 5, rest + L / 5))
    return p


def ex_wavecal_b(ctx, r, cls):
    return _ex_wavecal(ctx, r, cls, "b")


def ex_wavecal_r(ctx, r, cls):
    return _ex_wavecal(ctx, r, cls, "r")


def ex_wavecal_z(ctx, r, cls):
    return _ex_wavecal(ctx, r, cls, "z")


def ex_lsf(ctx, r, cls):
    from .metrics._skylines import SKY_LINES, lines_of
    from .metrics.drp_lsf import REF_RATIO
    ch = _s(r.get("lsf_worst_channel", ""), "r")
    sp = int(_f(r.get("lsf_worst_spec", 1))) if np.isfinite(_f(r.get("lsf_worst_spec", np.nan))) else 1
    name = lines_of(ch)[0]
    rest = SKY_LINES[name][0]
    rows = ctx.good_sci_index
    rows = rows[np.asarray(ctx.slitmap["spectrographid"])[rows] == sp]
    w, f = ctx.sframe_window("FLUX", rest - 5, rest + 5, rows=rows)
    _, s = ctx.sframe_window("SKY", rest - 5, rest + 5, rows=rows)
    _, lsf = ctx.sframe_window("LSF", rest - 1, rest + 1, rows=rows)
    prof = _nanmed(f + s)
    prof = prof - np.nanpercentile(prof, 5)
    prof = prof / np.nanmax(prof)
    mu = np.nansum(w * np.clip(prof, 0, None)) / np.nansum(np.clip(prof, 0, None))
    fw = float(np.nanmedian(lsf)) * REF_RATIO.get(name, 1.0)
    g = np.exp(-0.5 * ((w - mu) / (fw / 2.3548)) ** 2)
    p = _panel(f"{ch} sp{sp} {name}: Q_med {_f(r.get(f'lsf_q_med_{ch}', np.nan)):.3f}, frac|Q-1|>10% "
               f"{_f(r.get(f'lsf_frac_warn_{ch}', np.nan)):.3f}", ylabel="normalised")
    p["curves"] += [(w, prof, "sky line (median)", dict(color="k")),
                    (w, g, "Gaussian LSF x ref", dict(color="#d03b3b", ls="--"))]
    return p


def ex_telluric(ctx, r, cls):
    """Worst band (by flag contribution): median-stacked data and DAP continuum model, both divided
    by the same straight line through the continuum windows, and their ratio (offset by -0.3);
    for bands measured with the local interpolation, FLUX / interpolated continuum."""
    from .metrics.drp_telluric import BANDS
    ews = {b: _f(r.get(f"telluric_ew_{b}", np.nan)) for b in BANDS}
    rms = {b: _f(r.get(f"telluric_rms_{b}", np.nan)) for b in BANDS}
    score = {b: max(abs(ews[b]) / 1.0 if np.isfinite(ews[b]) else -1, rms[b] / 0.04 if np.isfinite(rms[b]) else -1)
             for b in BANDS}
    b = max(score, key=score.get)
    band, cl, cr = BANDS[b]
    meth = dict(x.split(":") for x in _s(r.get("telluric_method", "")).split() if ":" in x).get(b, "-")
    mt = _minst("qa_telluric", ctx.cfg)
    ew_d, ew_i = _f(r.get(f"telluric_ew_dap_{b}", np.nan)), _f(r.get(f"telluric_ew_interp_{b}", np.nan))
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        try:     # not stored (method not needed / older run): measure here for the title
            if not np.isfinite(ew_i):
                rs = mt._band(ctx, ctx.good_sci_index, band, cl, cr)
                ew_i = rs["ew"] if rs is not None else np.nan
            if not np.isfinite(ew_d) and ctx.has_dap and ctx.has_dap_output:
                rd = mt._band_dap(ctx, b, band, cl, cr)
                ew_d = rd["ew"] if rd is not None else np.nan
        except Exception:  # noqa: BLE001
            pass

    def _ew(x):
        return f"{x:+.2f}" if np.isfinite(x) else "n/a"
    mname = {"n": "DAP-normalised", "s": "interpolation"}.get(meth, meth)
    p = _panel(f"{b}: EW {_ew(ews[b])} A, rms {rms[b]:.3f} (adopted: {mname}; EW DAP-norm. {_ew(ew_d)}, "
               f"interp. {_ew(ew_i)})", ylabel="normalised flux; ratio - 0.3")

    def line(x, y):
        ml, mr = (x >= cl[0]) & (x <= cl[1]), (x >= cr[0]) & (x <= cr[1])
        yl, yr = _nanmed(y[:, ml], axis=1), _nanmed(y[:, mr], axis=1)
        xl, xr = np.median(x[ml]), np.median(x[mr])
        return yl[:, None] + (yr - yl)[:, None] * (x - xl)[None, :] / (xr - xl)

    with np.errstate(invalid="ignore", divide="ignore"):
        if ctx.has_dap and ctx.has_dap_output:
            rsp = ctx.dap_table("RSP")
            snr = np.asarray(rsp["med_flux"], float) / np.asarray(rsp["e_med_flux"], float)
            sel = np.where(snr >= 10)[0]
            sel = sel[np.argsort(snr[sel])[::-1][:100]]
            if sel.size:
                wd, pl = ctx.dap_output(f"dg_tell_{b}")
                from .registry import available_metrics
                dm = _minst("qa_telluric", ctx.cfg).params["dap_model"]
                data = pl[5][sel].astype(float)
                model = pl[1][sel].astype(float) if dm == "stellar" else pl[2][sel].astype(float) - pl[6][sel]
                L = line(wd, model)
                dn, mn = _nanmed(data / L), _nanmed(model / L)
                rr = _nanmed((data / model) / line(wd, data / model))
                p["curves"] += [(wd, dn, f"data - NP lines (n={sel.size})", dict(color="k", lw=0.7)),
                                (wd, mn, f"DAP {dm} model", dict(color="#eb6834", lw=0.9)),
                                (wd, rr - 0.3, "ratio (DAP-normalised) - 0.3", dict(color="#2a78d6", lw=0.7))]
        rows = ctx.good_sci_index
        w, f = ctx.sframe_window("FLUX", cl[0], cr[1], rows=rows)
        fl = _nanmed(f[:, (w >= cl[0]) & (w <= cl[1])], axis=1)
        fr = _nanmed(f[:, (w >= cr[0]) & (w <= cr[1])], axis=1)
        top = np.argsort(np.nan_to_num(np.minimum(fl, fr), nan=-np.inf))[::-1][:30]
        off = 0.3 if p["curves"] else 0.0
        p["curves"].append((w, _nanmed(f[top] / line(w, f[top])) - off,
                            "FLUX / interpolation" + (" - 0.3" if off else " (no DAP output)"),
                            dict(color="0.55" if off else "k", lw=0.6)))
    p["spans"] += [tuple(band)]
    p["spans"] += [(cl[0], cl[1], "0.85"), (cr[0], cr[1], "0.85")]
    p["hlines"] += [1.0, 0.7]
    return p


PASCHEN_NAMES = {"HI_9229.02": "P9", "HI_9014.91": "P10", "HI_8862.78": "P11", "HI_8750.47": "P12"}


def _lname(line: str) -> str:
    """'HI_9229.02' -> 'P9 9229', 'Halpha_6562.85' -> 'Halpha 6563', '[SIII]_9531.1' -> '[SIII] 9531'."""
    n, w = line.rsplit("_", 1)
    return f"{PASCHEN_NAMES.get(line, n)} {float(w):.0f}"


def _masked_core(ctx, wl, vel, half=1.5):
    """Per PT fiber: True if SFrame pixels within +-half A of the observed line centre are masked."""
    wm, mk = ctx.dap_sframe_window("MASK", wl - 12.0, wl + 12.0)
    _, fx = ctx.dap_sframe_window("FLUX", wl - 12.0, wl + 12.0)
    lc = wl * (1.0 + np.nan_to_num(vel) / C_KMS)
    core = np.abs(wm[None, :] - lc[:, None]) < half
    return (core & ((np.nan_to_num(mk) > 0) | ~np.isfinite(fx))).any(axis=1), wm, mk


def _pick(score, cls, rng, typical_ok):
    """Fiber for the panel: GOOD -> random typical fiber (typical_ok), else the largest score."""
    if cls == "GOOD":
        cand = np.where(typical_ok)[0]
        if cand.size:
            return int(rng.choice(cand))
    return int(np.nanargmax(score)) if np.isfinite(score).any() else 0


def _mark_masked(p, wm, mk, i):
    """Shade masked SFrame pixels of fiber i."""
    if i < mk.shape[0]:
        for w in wm[np.nan_to_num(mk[i]) > 0]:
            p["spans"].append((w - 0.25, w + 0.25, "0.4"))


def ex_fit_consistency(ctx, r, cls):
    from .metrics._dapout import np_column
    line = _s(r.get("fit_consistency_worst_line", ""), "Halpha_6562.85")
    key = {"Halpha_6562.85": "Halpha", "Hbeta_4861.36": "Hbeta", "[OIII]_5006.84": "OIII5007",
           "[NII]_6583.45": "NII6583", "[SII]_6716.44": "SII6716", "[SII]_6730.82": "SII6731",
           "[OII]_3726.03": "OII3727", "[OII]_3728.82": "OII3727"}.get(line, "Halpha")
    pm = ctx.dap_table("PM_KEL")
    pm = pm[np.char.strip(np.asarray(pm["model"]).astype(str)) == "eline"]
    wl = float(line.rsplit("_", 1)[1])
    sel = np.abs(np.asarray(pm["wl"], float) - wl) < 0.01
    ids = np.asarray(ctx.dap_table("PT")["id"]).astype(str)
    fpm = np.full(len(ids), np.nan)
    idx = {k: i for i, k in enumerate(ids)}
    for k, v in zip(np.asarray(pm["id"]).astype(str)[sel], np.asarray(pm["flux"], float)[sel]):
        if k in idx:
            fpm[idx[k]] = v
    fnp, enp = np_column(ctx, f"flux_{line}"), np_column(ctx, f"e_flux_{line}")
    masked, wm, mk = _masked_core(ctx, wl, np_column(ctx, f"vel_{line}"))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(fpm - fnp) / np.hypot(enp, 0.02 * np.abs(fnp))
        z[~(fnp / enp > 10)] = np.nan
    fmc = _f(r.get("fit_consistency_frac_masked_core", np.nan))
    rng = np.random.default_rng(int(ctx.drp("expnum", 0)))
    if cls != "GOOD" and np.isfinite(fmc) and fmc > 0.02:
        score = np.where(masked, z, np.nan)       # the issue is the masked line cores
    else:
        score = np.where(masked, np.nan, z)
    i = _pick(score, cls, rng, np.isfinite(z) & (z < 3) & ~masked)
    wd, pl = ctx.dap_output(f"dg_{key}")
    p = _panel(f"{_lname(line)} fiber {ids[i]}: F_pm/F_np {fpm[i] / fnp[i]:.2f}, |z| {z[i]:.1f}"
               f"{' MASKED CORE' if masked[i] else ''}; exp: frac_bad "
               f"{_f(r.get('fit_consistency_frac_bad', np.nan)):.2f}, masked {fmc:.3f}", ylabel="flux (gas = data - stellar)")
    p["curves"] += [(wd, pl[3][i], "gas spectrum", dict(color="k", drawstyle="steps-mid")),
                    (wd, pl[6][i], "NP model", dict(color="#2a78d6")),
                    (wd, pl[7][i], "PM model", dict(color="#d03b3b", ls="--"))]
    _mark_masked(p, wm, mk, i)
    return p


def ex_pmfit(ctx, r, cls):
    from .metrics.dap_pmfit import BRIGHT
    from .metrics._dapout import SQRT_2_PI, np_column
    k = _s(r.get("dap_pmfit_worst_line", ""), "Halpha")
    key = {"OII3727": "OII3727", "Hbeta": "Hbeta", "OIII5007": "OIII5007", "Halpha": "Halpha",
           "NII6583": "NII6583", "SII6716": "SII6716", "SII6731": "SII6731"}.get(k, "Halpha")
    from .metrics._dapout import excess_residual, local_baseline, sframe_on, sigma_on
    names = BRIGHT.get(key, BRIGHT["Halpha"])
    pr = _minst("qa_dap_pmfit", ctx.cfg).params
    wd, pl = ctx.dap_output(f"dg_{key}")
    F = sum(np.nan_to_num(np_column(ctx, f"flux_{n}")) for n in names)
    EF = np.sqrt(sum(np.nan_to_num(np_column(ctx, f"e_flux_{n}")) ** 2 for n in names))
    vel, disp = np_column(ctx, f"vel_{names[-1]}"), np_column(ctx, f"disp_{names[-1]}")
    sig = sigma_on(ctx, wd)
    wl = [float(n.rsplit("_", 1)[1]) for n in names]
    z = np.nan_to_num(vel) / C_KMS
    half = np.clip(np.nan_to_num(3.0 * disp, nan=pr["half_width"]), pr["half_width"], 10.0)
    use = (wd[None, :] >= (min(wl) * (1 + z) - half)[:, None]) & (wd[None, :] <= (max(wl) * (1 + z) + half)[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (EF > 0) & (F / EF > pr["min_snr"]) & np.isfinite(vel)
        r_pm = np.where(np.isfinite(sig), pl[0] - pl[8], np.nan)      # masked pixels excluded (as in the metric)
        base = np.zeros_like(r_pm)
        if pr.get("subtract_baseline", True):
            r_pm, base = local_baseline(r_pm, wd, (pl[7], pl[6]), exclude=use)
        fx, _, _, _ = excess_residual(r_pm, sig, pl[7].astype(float), use)
        fx = np.where(ok, fx, np.nan)
    rng = np.random.default_rng(int(ctx.drp("expnum", 0)))
    i = _pick(fx, cls, rng, ok & (fx < np.nanpercentile(fx, 75)) if np.isfinite(fx).any() else ok)
    ids = np.asarray(ctx.dap_table("PT")["id"]).astype(str)
    mk = sframe_on(ctx, "MASK", wd)
    p = _panel(f"{key} fiber {ids[i]}: unmodelled fx {fx[i]:.2f}; exp: frac_poor "
               f"{_f(r.get('dap_pmfit_frac_poor', np.nan)):.2f}",
               ylabel="flux (data - DAP continuum - residual baseline)")
    cont = pl[8][i] - pl[7][i] + base[i]          # stellar + smooth term + residual baseline
    p["curves"] += [(wd, pl[0][i] - cont, "data - continuum", dict(color="k", drawstyle="steps-mid")),
                    (wd, pl[7][i], "PM line model", dict(color="#d03b3b")),
                    (wd, r_pm[i], "residual (baseline-subtracted)", dict(color="#4a3aa7", lw=0.8)),
                    (wd, base[i], "removed residual baseline", dict(color="0.5", ls="--", lw=0.7))]
    _mark_masked(p, wd, mk, i)
    p["spans"].append((wd[use[i]].min(), wd[use[i]].max(), "#fab219") if use[i].any() else (wd[0], wd[0], "w"))
    p["hlines"].append(0.0)
    return p


def ex_npfluxes(ctx, r, cls):
    from .metrics.dap_npfluxes import GROUPS
    from .metrics._dapout import np_column
    g = _s(r.get("dap_npfluxes_worst_group", ""), "Halpha6563")
    if g not in GROUPS:
        g = "Halpha6563"
    from .metrics._dapout import local_baseline
    prim, (w0, w1) = GROUPS[g]
    pr = _minst("qa_dap_npfluxes", ctx.cfg).params
    wd, pl = ctx.dap_output(f"dg_fl_{g}")
    F, E = np_column(ctx, f"flux_{prim}"), np_column(ctx, f"e_flux_{prim}")
    v = np_column(ctx, "vel_Halpha_6562.85")
    with np.errstate(invalid="ignore", divide="ignore"):
        sel = np.where((E > 0) & (F / E >= pr["min_snr"]) & np.isfinite(v))[0]
    p = _panel(f"{g}: net {_f(r.get(f'dap_npfluxes_net_{g}', np.nan)):+.2f} (n={sel.size}; residual baseline "
               f"subtracted)", xlabel="rest wavelength [A]", ylabel="stacked flux (data - DAP continuum)")
    if sel.size:
        dw = float(wd[1] - wd[0])
        sh = np.round(0.5 * (w0 + w1) * v[sel] / C_KMS / dw).astype(int)
        x = np.arange(wd.size)
        res = pl[4][sel].astype(float)
        base = np.zeros_like(res)
        if pr.get("subtract_baseline", True):
            xw = np.arange(int(np.ceil((w0 - wd[0]) / dw)), int(np.floor((w1 - wd[0]) / dw)) + 1)
            win = np.zeros(res.shape, bool)
            np.put_along_axis(win, np.clip(sh[:, None] + xw[None, :], 0, wd.size - 1), True, axis=1)
            res, base = local_baseline(res, wd, (pl[6][sel],), exclude=win)
        idx = np.clip(x[None, :] + sh[:, None], 0, wd.size - 1)
        R = np.nansum(np.take_along_axis(res, idx, axis=1), axis=0)
        B = np.nansum(np.take_along_axis(base, idx, axis=1), axis=0)
        G = np.nansum(np.take_along_axis(pl[6][sel], idx, axis=1), axis=0)
        p["curves"] += [(wd, R + G, "data - continuum (stack)", dict(color="k", drawstyle="steps-mid")),
                        (wd, G, "NP model (all lines)", dict(color="#2a78d6")),
                        (wd, R, "residual", dict(color="#d03b3b", lw=0.8)),
                        (wd, B, "removed residual baseline", dict(color="0.5", ls="--", lw=0.7))]
    p["spans"].append((w0, w1))
    p["hlines"].append(0.0)
    return p


def ex_absorption(ctx, r, cls):
    from .metrics.dap_absorption import FEATURES
    feat = _s(r.get("absorption_worst_feature", ""), "Hb")
    if feat not in FEATURES:
        feat = "Hb"
    wd, pl = ctx.dap_output(f"dg_abs_{feat}")
    rsp = ctx.dap_table("RSP")
    snr = np.asarray(rsp["med_flux"], float) / np.asarray(rsp["e_med_flux"], float)
    sel = np.where(snr >= 10)[0]
    p = _panel(f"{feat}: dI {_f(r.get(f'absorption_dI_{feat}', np.nan)):+.2f} A (n={sel.size})", ylabel="normalised")
    if sel.size:
        data = pl[5][sel].astype(float)
        model = pl[2][sel].astype(float) - pl[6][sel]
        norm = _nanmed(model, axis=1)[:, None]
        p["curves"] += [(wd, _nanmed(data / norm), "data - NP lines", dict(color="k", drawstyle="steps-mid")),
                        (wd, _nanmed(model / norm), "DAP stellar model", dict(color="#d03b3b"))]
    b, c, rr = FEATURES[feat]
    p["spans"] += [tuple(b), tuple(c), tuple(rr)]
    return p


def ex_gaskinematics(ctx, r, cls):
    """The test that drives the flag: (a) stacked normalised profiles of the line pair with the
    largest |dv| (a defined pair: falls back from the stored worst pair to any measured one), or
    (b) when the spatial tests drive the flag (or no pair is measured), the distributions of the
    Halpha velocity residuals w.r.t. the neighbouring fibers per spectrograph (outlier limits as
    vertical lines) with the spectrograph-boundary offsets in the title."""
    from .metrics._dapout import np_column
    from .metrics.dap_gaskinematics import REST, _Neighbours, _col, _np_all, _pairs, pkey
    pr = _minst("qa_gaskinematics", ctx.cfg).params
    win = {"Halpha_6562.85": "Halpha", "Hbeta_4861.36": "Hbeta", "Hgamma_4340.49": "Hgamma",
           "HI_9229.02": "P9", "[SIII]_9531.1": "SIII9531", "[SIII]_9069.0": "SIII9069",
           "HeI_5876.0": "HeI5876", "HeI_6678.15": "HeI6678", "HeI_7065.19": "HeI7065"}
    cands = {}
    for g, ref, o in _pairs():
        k = f"{g}_{pkey(o)}"
        dv, e = _f(r.get(f"gaskinematics_dv_{k}", np.nan)), _f(r.get(f"gaskinematics_e_dv_{k}", np.nan))
        if np.isfinite(dv) and g not in pr["report_only"] and ref in win and o in win:
            cands[k] = (ref, o, dv, e)
    pair_issue = any(abs(dv) > pr["dv_warn"] and abs(dv) > pr["nsig"] * (e if np.isfinite(e) else 0)
                     for _, _, dv, e in cands.values())
    fo = np.nanmax([_f(r.get("gaskinematics_frac_out_v", np.nan)), _f(r.get("gaskinematics_frac_out_disp", np.nan)),
                    -1.0])
    mb = np.nanmax([_f(r.get("gaskinematics_max_boundary_dv", np.nan)),
                    _f(r.get("gaskinematics_max_boundary_ddisp", np.nan)), -1.0])
    spatial_issue = fo > pr["frac_out_warn"] or mb > pr["boundary_warn"]
    if cands and (pair_issue or not spatial_issue or cls == "GOOD"):
        wp = _s(r.get("gaskinematics_worst_pair", ""))
        k = wp if wp in cands else max(cands, key=lambda q: abs(cands[q][2]))
        ref, o, dv, e = cands[k]
        p = _panel(f"{_lname(o)} - {_lname(ref)}: dv {dv:+.1f}+-{e:.1f} km/s; max boundary offset "
                   f"{mb:.1f} km/s; Ha outliers {max(fo, 0):.3f}",
                   xlabel="velocity w.r.t. rest (air) wavelength [km/s]", ylabel="normalised gas profile (stack)")
        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = np_column(ctx, f"flux_{ref}") / np_column(ctx, f"e_flux_{ref}")
            s2 = np_column(ctx, f"flux_{o}") / np_column(ctx, f"e_flux_{o}")
            for thr in (pr["min_snr"], 5.0, 3.0):
                sel = np.where((s1 > thr) & (s2 > thr))[0]
                if sel.size >= 5:
                    break
        for line, col in ((ref, "k"), (o, "#d03b3b")):
            wd, pl = ctx.dap_output(f"dg_{win[line]}")
            rest = REST.get(line, float(line.rsplit("_", 1)[1]))
            vv = (wd / rest - 1) * C_KMS
            if sel.size:
                with np.errstate(invalid="ignore", divide="ignore"):
                    prof = _nanmed(pl[3][sel] / np.nanmax(pl[3][sel], axis=1, keepdims=True))
                m = np.abs(vv) < 300
                p["curves"].append((vv[m], prof[m], f"{_lname(line)} (n={sel.size}, S/N>{thr:g})", dict(color=col)))
        p["vlines"].append(0.0)
        return p
    # (b) spatial consistency of the brightest line
    t = _np_all(ctx)
    line = pr["line"]
    wl = float(line.rsplit("_", 1)[1])
    v = _col(t, f"vel_{line}")
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (_col(t, f"flux_{line}") / _col(t, f"e_flux_{line}") > pr["min_snr"]) & np.isfinite(v)
    bd = "; ".join(f"sp{q} {_f(r.get(f'gaskinematics_dv_sp{q}', np.nan)):+.1f}" for q in (1, 2, 3))
    p = _panel(f"{_lname(line)} spatial: outliers v {_f(r.get('gaskinematics_frac_out_v', np.nan)):.3f}, sigma "
               f"{_f(r.get('gaskinematics_frac_out_disp', np.nan)):.3f}; boundary dv {bd} km/s"
               + ("" if cands else " (no line pair measured)"),
               xlabel="v - median v of neighbours [km/s]", ylabel="fibers (normalised histogram)")
    if ok.sum() < 20:
        return p
    ra, dec = _col(t, "ra"), _col(t, "dec")
    x = (ra - np.nanmedian(ra)) * np.cos(np.deg2rad(np.nanmedian(dec))) * 3600.0
    y = (dec - np.nanmedian(dec)) * 3600.0
    P = np.c_[x, y][ok]
    tree = _Neighbours(P)
    vv = v[ok]
    rv = np.full(vv.size, np.nan)
    for i, nb in enumerate(tree.query_ball_point(P, pr["radius_pitch"] * tree.pitch)):
        nb = [j for j in nb if j != i]
        if len(nb) >= pr["min_neighbours"]:
            rv[i] = vv[i] - np.median(vv[nb])
    rows = ctx.dap_sframe_rows if ctx.has_sframe else None
    if rows is not None:
        spec = np.where(rows >= 0, np.asarray(ctx.slitmap["spectrographid"])[np.clip(rows, 0, None)], -1)
    else:
        spec = (np.asarray(t["fiberid"]).astype(int) - 1) // 648 + 1
    sp = spec[ok]
    lim = max(pr["out_nsig"] * 1.4826 * float(np.nanmedian(np.abs(rv))), pr["out_floor"])
    bins = np.linspace(-3 * lim, 3 * lim, 61)
    xc = 0.5 * (bins[1:] + bins[:-1])
    for q in (1, 2, 3):
        m = (sp == q) & np.isfinite(rv)
        if m.sum() >= 5:
            h, _ = np.histogram(np.clip(rv[m], bins[0], bins[-1]), bins=bins)
            p["curves"].append((xc, h / max(h.max(), 1), f"sp{q} (n={m.sum()}, median {np.median(rv[m]):+.1f})",
                                dict(color=SPEC_COL[q], drawstyle="steps-mid")))
    p["vlines"] += [-lim, lim]
    p["robust_ylim"] = False
    return p


def ex_lineratios(ctx, r, cls):
    """Distributions of R_obs / R_expected per fiber for all sub-tests (fibers with S/N > min_snr
    in all lines of the sub-test, NP fluxes; [SIII] expected ratio reddened with Ha/Hb)."""
    from .metrics.gen_lineratios import BALMER, DOUBLETS
    from .registry import available_metrics
    from .utils.extinction import ebv_from_balmer, k_lambda
    m = _minst("qa_lineratios", ctx.cfg)
    pr = m.params
    fl = m._fluxes(ctx)
    k9069, k9531 = k_lambda([9069.0, 9531.1], pr["rv"], pr["extinction_law"])
    col = {"oiii": "#2a78d6", "nii": "#eb6834", "siii": "#1baf7a", "hahb": "#d03b3b", "hbhg": "#4a3aa7"}
    lab = {"oiii": "[OIII]5007/4959", "nii": "[NII]6584/6548", "siii": "[SIII]9531/9069",
           "hahb": "Ha/Hb", "hbhg": "Hb/Hg"}
    bins = np.geomspace(0.4, 5.0, 64)          # log axis: Balmer decrements of reddened gas reach 2-4x Case B
    xc = np.sqrt(bins[1:] * bins[:-1])
    parts, curves = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for t, (num, den) in {**DOUBLETS, **BALMER}.items():
            keys = [num, den] + (["ha", "hb"] if t == "siii" else [])
            ok = m._good(fl, keys)
            y = fl[num][0][ok] / fl[den][0][ok]
            if t == "siii":
                ebv = ebv_from_balmer(fl["ha"][0][ok] / fl["hb"][0][ok], pr["ratio_hahb"], pr["rv"],
                                      pr["extinction_law"])
                y = y / (pr["ratio_siii"] * 10 ** (0.4 * ebv * (k9069 - k9531)))
            else:
                y = y / pr[f"ratio_{t}"]
            flag = _s(r.get(f"lineratios_{t}_flag", ""), "UNDEFINED")
            if flag != "UNDEFINED":
                if t in DOUBLETS:
                    stat = f"{_f(r.get(f'lineratios_{t}_dev', np.nan)):+.0f}%"
                    if t == "siii" and np.isfinite(_f(r.get("lineratios_siii_dev_raw", np.nan))):
                        stat += f" (uncorr. {_f(r.get('lineratios_siii_dev_raw', np.nan)):+.0f}%)"
                else:
                    stat = (f"median {_f(r.get(f'lineratios_{t}_med', np.nan)):.2f}, "
                            f"{100 * _f(r.get(f'lineratios_{t}_fbelow', np.nan)):.0f}% < Case B")
                parts.append(f"{lab[t]} {stat} ({flag.replace(' ISSUE', '')})")
            elif 0 < y.size < pr["min_fibers"]:
                parts.append(f"{lab[t]} ignored (n={y.size} < {pr['min_fibers']})")
            if y.size >= 5:
                h, _ = np.histogram(np.clip(y, bins[0], bins[-1]), bins=bins)
                fo = float(np.mean((y < bins[0]) | (y > bins[-1])))
                curves.append((xc, h / max(h.max(), 1), f"{lab[t]} (n={y.size}" + (f", {fo:.0%} outside" if fo >= 0.01 else "")
                               + ")", dict(color=col[t], drawstyle="steps-mid",
                                           ls="-" if y.size >= pr["min_fibers"] else ":")))
    p = _panel("; ".join(parts) or "no sub-test evaluable", xlabel="R_obs / R_expected (Balmer: Case B), log scale",
               ylabel="fibers (normalised histogram)")
    p["curves"] += curves
    p["vlines"].append(1.0)
    p["robust_ylim"] = False
    p["xscale"] = "log"
    return p


EXTRACTORS = {
    "qa_lineratios": ex_lineratios, "qa_skyline_sub": ex_skyline, "qa_skycont_sub": ex_skycont, "qa_flatfield": ex_flatfield,
    "qa_relflux_br": ex_relflux_br, "qa_relflux_rz": ex_relflux_rz,
    "qa_wavecal_b": ex_wavecal_b, "qa_wavecal_r": ex_wavecal_r, "qa_wavecal_z": ex_wavecal_z,
    "qa_lsf": ex_lsf, "qa_telluric": ex_telluric, "qa_fit_consistency": ex_fit_consistency,
    "qa_dap_pmfit": ex_pmfit, "qa_dap_npfluxes": ex_npfluxes, "qa_absorption": ex_absorption,
    "qa_gaskinematics": ex_gaskinematics,
}
NEEDS_OUTPUT = {"qa_fit_consistency", "qa_dap_pmfit", "qa_dap_npfluxes", "qa_absorption", "qa_gaskinematics"}


# ----------------------------------------------------------------------------- worker
def _minst(metric, cfg):
    """Metric instance with the parameter overrides of the run (cfg.metric_options)."""
    from .registry import available_metrics
    return available_metrics()[metric](**(getattr(cfg, "metric_options", None) or {}).get(metric, {}))


def _work(args):
    warnings.simplefilter("ignore")
    cfg, row, jobs = args
    out = {}
    with ExposureContext(cfg, row) as ctx:
        for metric, _, _, _ in jobs:
            ctx.request_output(_out_windows_for(metric))
        for metric, cls, slot, supl in jobs:
            try:
                if metric in NEEDS_OUTPUT and not (ctx.has_dap and ctx.has_dap_output):
                    raise RuntimeError("no DAP output")
                p = EXTRACTORS[metric](ctx, supl, cls)
            except Exception as exc:  # noqa: BLE001
                p = _panel(f"{type(exc).__name__}: {str(exc)[:60]}")
            p["expnum"] = int(row["expnum"])
            out[(metric, cls, slot)] = p
    return out


# ----------------------------------------------------------------------------- plotting
def _draw(ax, p, cls):
    import textwrap
    ys = []
    for x, y, lab, st in p["curves"]:
        st = dict(st)
        ds = st.pop("drawstyle", None)
        ax.plot(x, y, label=lab, lw=st.pop("lw", 0.9), drawstyle=ds or "default", **st)
        y = np.asarray(y, float)
        ys.append(y[np.isfinite(y)])
    for h in p["hlines"]:
        ax.axhline(h, color="0.75", lw=0.6, ls=":")
    for v in p["vlines"]:
        ax.axvline(v, color="0.5", lw=0.6, ls="--")
    for sp_ in p["spans"]:
        a, b = sp_[0], sp_[1]
        ax.axvspan(a, b, color=sp_[2] if len(sp_) > 2 else CCOL[cls], alpha=0.25 if len(sp_) > 2 else 0.10, lw=0)
    ys = np.concatenate(ys) if ys else np.zeros(0)
    if p.get("ylim"):
        ax.set_ylim(*p["ylim"])
    elif ys.size > 10 and p.get("robust_ylim", True):
        lo, hi = np.percentile(ys, [0.5, 99.5])
        if p["hlines"]:
            lo, hi = min(lo, min(p["hlines"])), max(hi, max(p["hlines"]))
        if hi > lo:
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
    if p.get("xscale"):
        ax.set_xscale(p["xscale"])
        if p["xscale"] == "log":
            from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
            ax.xaxis.set_major_locator(FixedLocator([0.5, 0.7, 1, 1.5, 2, 3, 4]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
    title = f"{p.get('expnum', '')}  {p['title']}"
    ax.set_title("\n".join(textwrap.wrap(title, 44)), fontsize=5.8, loc="left")
    ax.tick_params(labelsize=5.5)
    ax.yaxis.get_offset_text().set_fontsize(5.5)
    ax.set_xlabel(p["xlabel"], fontsize=6)
    if p["curves"]:
        ax.legend(fontsize=4.8, loc="best", frameon=False)
    for s_ in ax.spines.values():
        s_.set_color(CCOL[cls])
        s_.set_linewidth(1.5)


def _hull(P):
    """Convex hull (monotone chain) of 2D points, counter-clockwise."""
    P = np.unique(P, axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(P), half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def _clusters(ra, dec, link=2.0):
    """Group tile centres closer than ``link`` deg (single linkage) -> list of index arrays."""
    n = len(ra)
    lab = -np.ones(n, int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = c
        stack = [i]
        while stack:
            j = stack.pop()
            d = np.hypot((ra - ra[j]) * np.cos(np.deg2rad(dec[j])), dec - dec[j])
            for k in np.where((d < link) & (lab < 0))[0]:
                lab[k] = c
                stack.append(k)
        c += 1
    groups = [np.where(lab == k)[0] for k in range(c)]
    return sorted(groups, key=lambda g: -len(g))


def _asinh_norm(linear_width, vmin, vmax):
    """asinh colour stretch, clipped to [vmin, vmax]: matplotlib's AsinhNorm (>= 3.6) or an
    equivalent Normalize subclass for older matplotlib versions."""
    try:
        from matplotlib.colors import AsinhNorm
        return AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax, clip=True)
    except ImportError:
        return _AsinhNormCompat(linear_width, vmin, vmax)


def _make_asinh_compat():
    from matplotlib.colors import Normalize

    class AsinhNormCompat(Normalize):
        """y = asinh(x / a), linearly rescaled to [0, 1] between vmin and vmax (values clipped)."""

        def __init__(self, linear_width, vmin, vmax):
            super().__init__(vmin=vmin, vmax=vmax, clip=True)
            self.linear_width = float(linear_width)

        def _f(self, x):
            return np.arcsinh(np.asarray(x, dtype=float) / self.linear_width)

        def __call__(self, value, clip=None):
            val, is_scalar = self.process_value(value)
            data = np.clip(np.ma.getdata(val).astype(float), self.vmin, self.vmax)
            lo, hi = self._f(self.vmin), self._f(self.vmax)
            res = (self._f(data) - lo) / (hi - lo if hi > lo else 1.0)
            res = np.ma.array(res, mask=np.ma.getmaskarray(val) | ~np.isfinite(res))
            return res[0] if is_scalar else res

        def inverse(self, value):
            lo, hi = self._f(self.vmin), self._f(self.vmax)
            return self.linear_width * np.sinh(lo + np.asarray(value, dtype=float) * (hi - lo))

    return AsinhNormCompat


def _AsinhNormCompat(linear_width, vmin, vmax):  # noqa: N802 - factory (matplotlib imported lazily)
    return _make_asinh_compat()(linear_width, vmin, vmax)


def _fm_windows(ra, de, gidx, isflag, max_panels, box=6.0):
    """Panels of the fluxmap page (v1.15): regions = tile groups within 2 deg; a region larger than
    1.4 box is cut into box x box windows centred greedily on the densest concentration of flagged
    exposures. Windows sorted by their number of flagged exposures (then size).
    Returns (list of index arrays of the shown windows, number of windows)."""
    wins = []
    for g in _clusters(ra[gidx], de[gidx]):
        g = gidx[g]
        r0 = float(np.median(ra[g]))
        x = ((ra[g] - r0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(de[g]))
        y = de[g]
        if max(np.ptp(x), np.ptp(y)) <= 1.4 * box:
            wins.append((int(isflag[g].sum()), len(g), g))
            continue
        todo = set(np.where(isflag[g])[0])
        while todo and len(wins) < 4 * max_panels:
            t = np.array(sorted(todo))
            cnt = [np.sum((np.abs(x[t] - x[k]) < box / 2) & (np.abs(y[t] - y[k]) < box / 2)) for k in t]
            k = t[int(np.argmax(cnt))]
            inb = (np.abs(x - x[k]) < box / 2) & (np.abs(y - y[k]) < box / 2)
            wins.append((int(isflag[g][inb].sum()), int(inb.sum()), g[inb]))
            todo -= set(np.where(inb)[0])
    wins.sort(key=lambda w_: (-w_[0], -w_[1]))
    return [w_[2] for w_ in wins[:max_panels]], len(wins)


def _fluxmap_overview(pdf, qa, supl):
    """Overview of qa_fluxmap for the whole run (v1.15): tile centres coloured by the local offset l
    (flagged ones outlined), distribution of l, and l vs MJD with the nightly medians."""
    import matplotlib.pyplot as plt
    if "fluxmap_local_dex" not in supl.colnames or "sci_ra" not in supl.colnames:
        return
    by = {int(r["expnum"]): str(r["qa_fluxmap"]) for r in qa} if "qa_fluxmap" in qa.colnames else {}
    e = np.asarray(supl["expnum"]).astype(int)
    l = np.array([_f(x) for x in supl["fluxmap_local_dex"]])
    ra, de = np.asarray(supl["sci_ra"], float), np.asarray(supl["sci_dec"], float)
    mjd = np.asarray(supl["mjd"], float) if "mjd" in supl.colnames else np.full(len(e), np.nan)
    st = np.array([by.get(int(x), "") for x in e])
    ok = np.isfinite(l) & np.isfinite(ra) & np.isfinite(de) & (st != "UNDEFINED")
    if ok.sum() < 3:
        return
    r0 = float(np.median(ra[ok]))
    x = r0 + (ra - r0 + 180.0) % 360.0 - 180.0
    fig = plt.figure(figsize=(11.7, 8.3))
    a1 = fig.add_axes([0.06, 0.40, 0.88, 0.50])
    ms = float(np.clip(6000.0 / ok.sum(), 6, 120))          # marker size: smaller for large runs
    sc = a1.scatter(x[ok], de[ok], c=np.clip(l[ok], -0.15, 0.15), cmap="coolwarm", vmin=-0.15, vmax=0.15, s=ms,
                    lw=0.2, edgecolors="0.7", rasterized=True)
    for c_, z_ in (("WARNING", 3), ("MAJOR ISSUE", 4)):
        m = ok & (st == c_)
        a1.scatter(x[m], de[m], facecolors="none", edgecolors=CCOL[c_], s=2.5 * ms, lw=0.7, zorder=z_, rasterized=True,
                   label=f"{c_} ({m.sum()})")
    und = np.isfinite(ra) & np.isfinite(de) & (st == "UNDEFINED")
    a1.scatter(x[und], de[und], marker="x", color="0.6", s=6, lw=0.5, rasterized=True, label=f"UNDEFINED ({und.sum()})")
    a1.invert_xaxis()
    a1.set_xlabel("RA [deg]", fontsize=8)
    a1.set_ylabel("Dec [deg]", fontsize=8)
    a1.tick_params(labelsize=7)
    a1.legend(fontsize=7, loc="best")
    cb = fig.colorbar(sc, ax=a1, pad=0.01, fraction=0.03)
    cb.set_label("local offset l [dex]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    a2 = fig.add_axes([0.06, 0.07, 0.25, 0.25])
    bins = np.linspace(-0.4, 0.4, 81)
    a2.hist(np.clip(l[ok], bins[0], bins[-1]), bins=bins, color="0.4")
    for t in ("warn_dex", "major_dex"):
        v = 0.04 if t == "warn_dex" else 0.10
        a2.axvline(v, color=CCOL["WARNING" if t == "warn_dex" else "MAJOR ISSUE"], lw=0.8)
        a2.axvline(-v, color=CCOL["WARNING" if t == "warn_dex" else "MAJOR ISSUE"], lw=0.8)
    a2.set_yscale("log")
    nm = 1.4826 * np.median(np.abs(l[ok] - np.median(l[ok])))
    a2.set_title(f"local offsets: median {np.median(l[ok]):+.3f}, NMAD {nm:.3f} dex (n={ok.sum()})", fontsize=7)
    a2.set_xlabel("l [dex]", fontsize=7)
    a2.tick_params(labelsize=6)
    a3 = fig.add_axes([0.38, 0.07, 0.56, 0.25])
    mm = ok & np.isfinite(mjd)
    a3.plot(mjd[mm], np.clip(l[mm], -0.4, 0.4), ".", ms=2, color="0.55", rasterized=True)
    um = np.unique(mjd[mm])
    med = np.array([np.median(l[mm & (mjd == u)]) for u in um])
    nn = np.array([np.sum(mm & (mjd == u)) for u in um])
    a3.scatter(um[nn >= 3], np.clip(med[nn >= 3], -0.4, 0.4), s=10, color="#d03b3b", zorder=3, label="nightly median (>= 3 exp.)")
    for v in (0.04, 0.10):
        a3.axhline(v, color="0.8", lw=0.5, ls=":")
        a3.axhline(-v, color="0.8", lw=0.5, ls=":")
    a3.set_xlabel("MJD", fontsize=7)
    a3.set_ylabel("l [dex]", fontsize=7)
    a3.tick_params(labelsize=6)
    a3.legend(fontsize=6, loc="best")
    fig.suptitle("qa_fluxmap overview: local Halpha flux-scale offsets of all exposures", fontsize=10)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _fluxmap_page(pdf, qa, supl, cfg, max_panels=6, coords=None, nproc=1, max_labels=8):
    """Halpha maps around the flagged exposures (v1.15: windows <= 6 deg, see _fm_windows; labels
    only for the max_labels largest |l| per panel), after an overview page (_fluxmap_overview).

    Regions are found from the tile centres (``coords`` = {expnum: (ra, dec)} from drpall; DAP
    fiber positions otherwise); DAP maps are read only for the exposures that are drawn (first
    exposure of each pointing in the shown regions), so the page stays cheap for large runs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .metrics.dap_fluxmap import _halpha_map_job, halpha_map, has_halpha_map
    from .paths import dap_path
    coords = coords or {}
    cand, files = [], []
    for row in qa:   # exposures with a DAP file or a cached Halpha map
        if has_halpha_map(cfg, int(row["tileid"]), int(row["mjd"]), int(row["expnum"])):
            cand.append(row)
            files.append(dap_path(cfg, int(row["tileid"]), int(row["mjd"]), int(row["expnum"])))
    if not cand:
        return

    class _Maps(dict):  # Halpha maps (cache or DAP file), read on demand: pts[i] = (row, payload, flux)
        def __missing__(self, i):
            r = cand[i]
            pl = halpha_map(cfg, int(r["tileid"]), int(r["mjd"]), int(r["expnum"]), files[i])
            v = (r, pl, np.asarray(pl["flux"], float))
            self[i] = v
            return v

    pts = _Maps()
    ra0 = np.full(len(cand), np.nan)
    de0 = np.full(len(cand), np.nan)
    for i, row in enumerate(cand):
        c = coords.get(int(row["expnum"]))
        if c is not None and np.all(np.isfinite(c)):
            ra0[i], de0[i] = c
        else:
            ra0[i], de0[i] = np.nanmedian(pts[i][1]["ra"]), np.nanmedian(pts[i][1]["dec"])
    good = np.isfinite(ra0) & np.isfinite(de0)
    if not good.any():
        return
    def _norm(idx):
        """Colour scale of one field (panel): pooled fluxes of its tiles, excluding tiles with absurd
        fluxes (per-tile 99th percentile > 20 x the median of the per-tile values, e.g. a broken
        spectrograph). vmax = 99.2th percentile (faint structures visible, bright cores not overly
        saturated); vmin = 0 (negative noise fluxes clipped to the darkest colour); asinh stretch
        linear around the faint level (20th percentile of the positive fluxes)."""
        fin = [pts[i][2][np.isfinite(pts[i][2])] for i in idx]
        p99 = np.array([np.percentile(x, 99) if x.size else np.nan for x in fin])
        ref = np.nanmedian(p99) if np.isfinite(p99).any() else np.nan
        keep = [x for x, q in zip(fin, p99) if np.isfinite(q) and q < 20 * ref] or fin
        allv = np.concatenate(keep) if keep else np.array([1.0])
        hi = np.percentile(allv, 99.2) if allv.size else 1.0
        hi = hi if np.isfinite(hi) and hi > 0 else 1.0
        pos = allv[allv > 0]
        lw = np.percentile(pos, 20) if pos.size else 1e-3 * hi
        return _asinh_norm(max(lw, 1e-3 * hi), 0.0, hi)

    gidx = np.where(good)[0]
    stat = {i: (str(cand[i]["qa_fluxmap"]) if "qa_fluxmap" in qa.colnames else "") for i in range(len(cand))}
    isflag = np.array([stat[i] in ("MAJOR ISSUE", "WARNING") for i in range(len(cand))])
    shown, n_win = _fm_windows(ra0, de0, gidx, isflag, max_panels)
    groups = [None] * n_win
    # all exposures of the shown regions are averaged in the image: read their maps (Halpha map
    # cache of the run; missing ones from the DAP files, in parallel)
    need = [int(i) for g in shown for i in g if int(i) not in pts]
    if nproc > 1 and len(need) > 4:
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(cfg, int(cand[i]["tileid"]), int(cand[i]["mjd"]), int(cand[i]["expnum"])) for i in need]
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            got = dict(ex.map(_halpha_map_job, jobs, chunksize=4))
        for i in need:
            pl = got.get(int(cand[i]["expnum"]))
            if pl is not None:
                pts[i] = (cand[i], pl, np.asarray(pl["flux"], float))
    nc = 1 if len(shown) == 1 else (2 if len(shown) <= 4 else 3)
    nr = int(np.ceil(len(shown) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11.7, 8.3), squeeze=False)
    by = {int(r["expnum"]): r for r in supl}
    flagged = ("MAJOR ISSUE", "WARNING")
    sev = {"MAJOR ISSUE": 3, "WARNING": 2, "GOOD": 1}
    lcol = "fluxmap_local_dex" if "fluxmap_local_dex" in supl.colnames else "fluxmap_s_dex"
    lval = {i: (_f(by[int(cand[i]["expnum"])][lcol]) if int(cand[i]["expnum"]) in by else np.nan) for i in range(len(cand))}
    for ax, g in zip(axes.flat, shown):
        g = [int(i) for i in g]
        ra_c, de_c = float(np.median(ra0[g])), float(np.median(de0[g]))
        cos0 = max(np.cos(np.deg2rad(de_c)), 1e-3)
        xs, ys, vs = [], [], []
        for i in g:
            pl, lf = pts[i][1], pts[i][2]
            ok = np.isfinite(pl["ra"]) & np.isfinite(pl["dec"]) & np.isfinite(lf)
            dra = (np.asarray(pl["ra"], float)[ok] - ra_c + 180.0) % 360.0 - 180.0  # RA wrap
            xs.append(dra * cos0)
            ys.append(np.asarray(pl["dec"], float)[ok] - de_c)
            vs.append(lf[ok])
        x, y, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(vs)
        pad = 2 * _FIB_CELL_R
        xl, yl = (x.min() - pad, x.max() + pad), (y.min() - pad, y.max() + pad)
        img = _fiber_image(x, y, v, xl, yl)
        ext = (ra_c + xl[0] / cos0, ra_c + xl[1] / cos0, de_c + yl[0], de_c + yl[1])
        cmap = plt.get_cmap("magma").copy() if hasattr(plt.get_cmap("magma"), "copy") else plt.get_cmap("magma")
        cmap.set_bad(alpha=0.0)
        im = ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=ext, cmap=cmap, norm=_norm(g),
                       interpolation="none", aspect=1.0 / cos0)
        # outlines and labels only for WARNING / MAJOR pointings (repeated pointings, centres
        # < 0.05 deg apart, share one outline = worst status); outlines rasterized, text vector
        flg = [i for i in g if stat[i] in flagged]
        lab_ok = set(sorted(flg, key=lambda i: -abs(np.nan_to_num(lval[i])))[:max_labels])
        for sub in _clusters(ra0[g], de0[g], link=0.05):
            idx = [g[k] for k in sub]
            bad = [i for i in idx if stat[i] in flagged]
            if not bad:
                continue
            worst = max((stat[i] for i in bad), key=lambda s_: sev[s_])
            pl = pts[bad[0]][1]
            ok = np.isfinite(pl["ra"]) & np.isfinite(pl["dec"])
            rr = ra_c + (np.asarray(pl["ra"], float)[ok] - ra_c + 180.0) % 360.0 - 180.0
            h = _hull(np.c_[rr, np.asarray(pl["dec"], float)[ok]])
            if len(h) > 2:
                ax.plot(np.r_[h[:, 0], h[0, 0]], np.r_[h[:, 1], h[0, 1]], color=CCOL[worst], lw=1.6,
                        rasterized=True)
            lines = [f"{int(cand[i]['expnum'])} {lval[i]:+.3f} {stat[i][:4]}" for i in bad if i in lab_ok]
            if not lines:
                continue
            ax.text(ra_c + (float(np.median(ra0[bad])) - ra_c + 180.0) % 360.0 - 180.0, float(np.median(de0[bad])),
                    "\n".join(lines), ha="center", va="center", fontsize=4.8, color="k", family="monospace",
                    bbox=dict(facecolor=CCOL[worst], alpha=0.6, lw=0, pad=1.2), clip_on=True)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.set_xlim(ext[1], ext[0])  # RA increases to the left
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("RA [deg]", fontsize=7)
        ax.set_ylabel("Dec [deg]", fontsize=7)
        ax.tick_params(labelsize=6)
        cnt = {c: sum(stat[i] == c for i in g) for c in ("MAJOR ISSUE", "WARNING", "GOOD")}
        nund = len(g) - sum(cnt.values())
        ax.set_title(f"{len(g)} exposures around RA {ra_c:.1f}, Dec {de_c:+.1f}\n{cnt['MAJOR ISSUE']} MAJOR, "
                     f"{cnt['WARNING']} WARNING, {cnt['GOOD']} GOOD, {nund} UNDEFINED", fontsize=7)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=CCOL[c], lw=2, label=c) for c in flagged], loc="lower center",
               ncol=2, fontsize=7,
               title=f"outline = WARNING / MAJOR pointings, label = expnum, local offset l [dex] ({max_labels} largest |l| "
                     "per panel); image = F(Halpha) [1e-16 erg/s/cm2] (DAP NP), mean of overlapping fibers, asinh, own "
                     "scale per panel",
               title_fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.12)
    extra = f" ({len(groups) - len(shown)} more windows not shown)" if len(groups) > len(shown) else ""
    fig.suptitle("qa_fluxmap: Halpha maps around the flagged exposures (windows <= 6 deg, most flags first)" + extra,
                 fontsize=10)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def _gradient_page(pdf, qa, supl, cfg, n_examples=4, seed=1):
    """qa_gradient page (v1.15). Top: Halpha maps of example exposures in the IFU frame (largest
    significant spread among MAJOR, then WARNING, then a random GOOD one; maps from the Halpha map
    cache / DAP files), spectrograph sectors (boundaries at IFU PA 60, 180, 300 deg; layout from the
    fiber positions), the fitted sector offsets and the boundary jumps d12, d23, d31. Bottom: run
    level: distribution of the spread, nightly medians of the sector offsets, and the mosaic IFU
    dipole of qa_fluxmap (arrow towards the brighter IFU side)."""
    import matplotlib.pyplot as plt
    from .metrics.dap_fluxmap import halpha_map, has_halpha_map
    from .utils.ifu_layout import assign_fibers
    if "qa_gradient" not in qa.colnames or "gradient_spread_dex" not in supl.colnames:
        return
    if not np.isfinite(np.array([_f(v) for v in supl["gradient_spread_dex"]], float)).any():
        return
    sup = {int(r["expnum"]): r for r in supl}
    st = {int(r["expnum"]): str(r["qa_gradient"]) for r in qa}
    rows = {int(r["expnum"]): r for r in qa}
    g = lambda e, k: _f(sup[e][f"gradient_{k}"]) if e in sup and f"gradient_{k}" in supl.colnames else np.nan  # noqa: E731
    rng = random.Random(seed)
    ex = []
    for cls in ("MAJOR ISSUE", "WARNING", "GOOD"):
        cand = [e for e in st if st[e] == cls and np.isfinite(g(e, "spread_dex"))
                and has_halpha_map(cfg, int(rows[e]["tileid"]), int(rows[e]["mjd"]), e)]
        if not cand:
            continue
        if cls == "GOOD":
            ex.append((rng.choice(cand), cls))
        else:
            cand.sort(key=lambda e: -g(e, "spread_dex") / max(g(e, "e_spread_dex"), 1e-4))
            ex += [(e, cls) for e in cand[:2 if cls == "MAJOR ISSUE" else 1]]
    ex = ex[:n_examples]
    fig = plt.figure(figsize=(11.7, 8.3))
    W = 0.92 / max(len(ex), 1)
    for k, (e, cls) in enumerate(ex):
        ax = fig.add_axes([0.04 + k * W, 0.47, W * 0.86, 0.40])
        r = rows[e]
        try:
            pl = halpha_map(cfg, int(r["tileid"]), int(r["mjd"]), e)
        except Exception:  # noqa: BLE001
            pl = None
        if pl is None:
            ax.axis("off")
            continue
        pa = _f(sup[e]["sci_pa"]) if "sci_pa" in supl.colnames else 0.0
        lay = assign_fibers(np.asarray(pl["ra"], float), np.asarray(pl["dec"], float), np.nan_to_num(pa))
        x, y = np.asarray(lay["x"], float) / 3600.0, np.asarray(lay["y"], float) / 3600.0
        f = np.asarray(pl["flux"], float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.log10(np.where(f > 0, f, np.nan))
        lim = (-0.26, 0.26)
        img = _fiber_image(x[ok], y[ok], v[ok], lim, lim, npix=300)
        lo, hi = np.nanpercentile(v[ok], [3, 99]) if np.isfinite(v[ok]).any() else (0, 1)
        im = ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=(lim[0], lim[1], lim[0], lim[1]),
                       cmap="magma", vmin=lo, vmax=hi, interpolation="none")
        R = 0.26
        for b in (60.0, 180.0, 300.0):     # sector boundaries (IFU PA east of north)
            t = np.deg2rad(b)
            ax.plot([0, R * np.sin(t)], [0, R * np.cos(t)], color="w", lw=1.0, ls="--")
        for sp, mid in ((1, 120.0), (2, 240.0), (3, 0.0)):
            t = np.deg2rad(mid)
            o = g(e, f"sp{sp}_dex")
            ax.text(0.15 * np.sin(t), 0.15 * np.cos(t), f"sp{sp}\n{o:+.3f}" if np.isfinite(o) else f"sp{sp}",
                    color=SPEC_COL[sp], ha="center", va="center", fontsize=7, fontweight="bold",
                    bbox=dict(facecolor="w", alpha=0.75, lw=0, pad=1))
        for key, b in (("d12_dex", 180.0), ("d23_dex", 300.0), ("d31_dex", 60.0)):
            t = np.deg2rad(b)
            dv = g(e, key)
            if np.isfinite(dv):
                ax.text(0.235 * np.sin(t), 0.235 * np.cos(t), f"{key[:3]} {dv:+.3f}", fontsize=5.5, ha="center",
                        va="center", color="k", bbox=dict(facecolor="w", alpha=0.75, lw=0, pad=0.8))
        ax.set_xlim(lim[1], lim[0])        # east to the left
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("IFU x [deg] (east for sci_pa = 0)", fontsize=6)
        for s_ in ax.spines.values():
            s_.set_color(CCOL[cls])
            s_.set_linewidth(1.5)
        ax.set_title(f"{e} {cls}: spread {g(e, 'spread_dex'):.3f}\u00b1{g(e, 'e_spread_dex'):.3f} dex\n"
                     f"closure {g(e, 'closure_dex'):+.3f}\u00b1{g(e, 'e_closure_dex'):.3f}; "
                     f"{int(np.nan_to_num(g(e, 'n_cross')))} pairs; PA {pa:.0f}", fontsize=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("log F(Halpha)", fontsize=5.5)
    # ---- run level
    e_all = np.array([e for e in st], int)
    sp_ = np.array([g(e, "spread_dex") for e in e_all])
    a1 = fig.add_axes([0.05, 0.07, 0.25, 0.28])
    fin = np.isfinite(sp_)
    if fin.any():
        bins = np.linspace(0, max(0.2, float(np.nanpercentile(sp_[fin], 99.5))), 50)
        for cls in ("GOOD", "WARNING", "MAJOR ISSUE"):
            m = fin & np.array([st[e] == cls for e in e_all])
            if m.any():
                a1.hist(np.clip(sp_[m], bins[0], bins[-1]), bins=bins, color=CCOL[cls], alpha=0.8, label=f"{cls} ({m.sum()})")
        a1.set_yscale("log")
        for v_, c_ in ((0.04, "WARNING"), (0.10, "MAJOR ISSUE")):
            a1.axvline(v_, color=CCOL[c_], lw=0.8, ls=":")
        a1.legend(fontsize=5.5)
    a1.set_xlabel("spread = max - min sector offset [dex]", fontsize=6.5)
    a1.set_title(f"sector spread ({fin.sum()} exposures measured, {(~fin).sum()} not)", fontsize=7)
    a1.tick_params(labelsize=6)
    a2 = fig.add_axes([0.37, 0.07, 0.36, 0.28])
    mjd = np.array([int(rows[e]["mjd"]) for e in e_all])
    for sp in (1, 2, 3):
        o = np.array([g(e, f"sp{sp}_dex") for e in e_all])
        m = np.isfinite(o)
        um = np.unique(mjd[m])
        med = [np.median(o[m & (mjd == u)]) for u in um]
        nn = np.array([np.sum(m & (mjd == u)) for u in um])
        a2.plot(um[nn >= 3], np.array(med)[nn >= 3], ".-", ms=3, lw=0.6, color=SPEC_COL[sp], label=f"sp{sp}")
    a2.axhline(0, color="0.7", lw=0.6)
    a2.set_xlabel("MJD", fontsize=6.5)
    a2.set_ylabel("nightly median offset [dex]", fontsize=6.5)
    a2.set_title("spectrograph sector offsets per night (>= 3 exposures)", fontsize=7)
    a2.legend(fontsize=6)
    a2.tick_params(labelsize=6)
    a3 = fig.add_axes([0.78, 0.07, 0.19, 0.28])
    dp, edp, dpa = (np.nanmedian([g(e, k) for e in e_all]) for k in ("ifu_dipole_dex", "e_ifu_dipole_dex", "ifu_dipole_pa"))
    hx = np.deg2rad(np.arange(0, 361, 60) + 30)
    a3.plot(np.sin(hx), np.cos(hx), color="0.5", lw=1)
    for b in (60.0, 180.0, 300.0):
        t = np.deg2rad(b)
        a3.plot([0, 0.87 * np.sin(t)], [0, 0.87 * np.cos(t)], color="0.8", lw=0.6, ls="--")
    if np.isfinite(dp) and np.isfinite(dpa):
        L = min(0.9, dp / 0.10 * 0.9)
        t = np.deg2rad(dpa)
        a3.annotate("", xy=(L * np.sin(t), L * np.cos(t)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=2))
        a3.set_title(f"mosaic IFU dipole (qa_fluxmap)\n{dp:.3f}\u00b1{edp:.3f} dex towards IFU PA {dpa:.0f} deg\n"
                     f"(arrow length: 0.10 dex = full)", fontsize=6.5)
    else:
        a3.set_title("mosaic IFU dipole: not fitted", fontsize=6.5)
    a3.set_xlim(1.1, -1.1)
    a3.set_ylim(-1.1, 1.1)
    a3.set_aspect("equal")
    a3.axis("off")
    fig.suptitle("qa_gradient: Halpha flux-scale offsets of the spectrograph sectors (examples) and run overview",
                 fontsize=10)
    fig.text(0.04, 0.42, "examples: map = log F(Halpha) (DAP NP) in the IFU frame, dashed = sector boundaries; "
             "sector offsets (sum 0) and boundary jumps d (log F_b/F_a across the boundary, sky structure "
             "removed with control pairs) in dex", fontsize=6.5, color="0.3")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


_FIB_R = 35.3 / 2 / 3600.0               # LVM fiber radius [deg]
_FIB_CELL_R = 37.0 / np.sqrt(3) / 3600.0  # circumradius of the hexagonal fiber cell (pitch 37") [deg]


def _fiber_image(x, y, v, xlim, ylim, npix=900):
    """Raster map of fiber values (x, y in deg on the sky plane, x = dRA cos(dec0)): a pixel takes
    the mean of all fibers whose disk (radius 17.65") covers its centre (uniform within a fiber;
    overlapping fibers of different exposures are averaged). Pixels in the small gaps between the
    fibers of a tile take the mean of the fibers within the hexagonal-cell radius (21.4"); pixels
    without fiber coverage are NaN. Pixel size: longest side / npix, at least fiber radius / 6."""
    w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    pix = max(max(w, h) / npix, _FIB_R / 6.0)
    nx, ny = max(int(np.ceil(w / pix)), 1), max(int(np.ceil(h / pix)), 1)
    fx, fy = (x - xlim[0]) / pix, (y - ylim[0]) / pix  # fiber centres in pixel units
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)

    def accumulate(radius):
        r = radius / pix
        k = int(np.ceil(r)) + 1
        s = np.zeros(nx * ny)
        c = np.zeros(nx * ny)
        for oy in range(-k, k + 1):
            for ox in range(-k, k + 1):
                jx, jy = ix + ox, iy + oy
                d2 = (jx + 0.5 - fx) ** 2 + (jy + 0.5 - fy) ** 2
                m = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                m &= (d2 <= r * r) | ((ox == 0) & (oy == 0))  # a fiber always covers its own pixel
                if not m.any():
                    continue
                flat = jy[m] * nx + jx[m]
                s += np.bincount(flat, weights=v[m], minlength=nx * ny)
                c += np.bincount(flat, minlength=nx * ny)
        return s, c

    s, c = accumulate(_FIB_R)
    img = np.full(nx * ny, np.nan)
    ok = c > 0
    img[ok] = s[ok] / c[ok]
    s2, c2 = accumulate(max(_FIB_CELL_R, 1.0 * pix))
    gap = ~ok & (c2 > 0)
    img[gap] = s2[gap] / c2[gap]
    return img.reshape(ny, nx)


def _hist_order(qa):
    """Metrics with histogram specs, in registry order, present in the QA table."""
    from .histpages import SPECS
    from .registry import available_metrics
    return [m for m in available_metrics() if m in SPECS and m in qa.colnames]


def _hist(pdf, metric, cfg, qa, supl) -> int:
    """Histogram page(s) of the per-exposure values of ``metric`` (lvmqa.histpages)."""
    from .histpages import hist_pages
    try:
        m = _minst(metric, cfg)
        return hist_pages(pdf, metric, m.params, qa, supl, m.short, criteria_text=_criteria_text(metric, cfg))
    except Exception as exc:  # noqa: BLE001
        log.warning("diagnostics PDF: histograms of %s failed: %s", metric, exc)
        return 0


def hist_pdf(qa, supl, cfg, out) -> Path:
    """PDF with only the value-distribution pages of all metrics (no data are read)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for m in _hist_order(qa):
            _hist(pdf, m, cfg, qa, supl)
    return Path(out)


def _criteria_text(metric, cfg, width=190) -> str:
    """Short summary of the GOOD / WARNING / MAJOR definitions of a metric (current parameters)."""
    import textwrap
    from .registry import available_metrics
    try:
        crit = _minst(metric, cfg).criteria
    except Exception:  # noqa: BLE001
        return ""
    lines = []
    for k, lab in (("GOOD", "GOOD   "), ("WARNING", "WARNING"), ("MAJOR ISSUE", "MAJOR  ")):
        txt = " ".join(str(crit.get(k, "")).split())
        w = textwrap.wrap(txt, width) or [""]
        lines.append(f"{lab}: {w[0]}")
        lines += [" " * 9 + x for x in w[1:3]] + ([" " * 9 + "..."] if len(w) > 3 else [])
    return "\n".join(lines)


def _inputs_from_header(cfg, base) -> None:
    """Non-standard inputs of the run (SFrame pattern, no DAP, no drpall; v1.15) from the primary
    header of <base>.fits, unless already set in cfg."""
    try:
        from astropy.io import fits
        h = fits.getheader(f"{base}.fits", 0)
    except Exception:  # noqa: BLE001
        return
    if not cfg.sframe_pattern and h.get("SFPATT"):
        cfg.sframe_pattern = str(h["SFPATT"])
    cfg.no_dap = bool(cfg.no_dap or h.get("NODAP", False))
    cfg.no_drpall = bool(cfg.no_drpall or h.get("NODRPALL", False))


def make_pdf(drp_version, dap_version=None, n=5, seed=1, nproc=3, out=None, tag=None, suffix="",
             metric_options=None, **paths) -> Path:
    """Diagnostics PDF from the written tables <base><suffix>.fits / _supl.fits (re-reads the data of
    the plotted exposures). Called at the end of every pipeline run, and by the recalculation
    (python -m lvmqa.diagnostics ... --pdf). ``paths``: PipelineConfig keywords (redux_dir, ...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from astropy.table import Table

    paths = {k: v for k, v in paths.items() if v is not None}
    cfg = PipelineConfig(drp_version=drp_version, dap_version=dap_version,
                         metric_options=dict(metric_options or {}), **paths)
    _inputs_from_header(cfg, Path(cfg.output_dir) / output_basename(cfg, tag, suffix))
    base = Path(cfg.output_dir) / output_basename(cfg, tag, suffix)
    if not Path(f"{base}.fits").is_file():
        raise FileNotFoundError(f"{base}.fits not found (check --output-dir / --tag / --suffix)")
    qa = Table.read(f"{base}.fits", hdu="QA")
    supl = Table.read(f"{base}_supl.fits", hdu="SUPL")
    from .pipeline import exposure_table, select_exposures, _row_to_dict
    drpall = exposure_table(cfg, list(qa["expnum"]), nthreads=max(16, 2 * nproc))
    rows = {int(r["expnum"]): _row_to_dict(r) for r in select_exposures(drpall, list(qa["expnum"]))}
    sup = {int(r["expnum"]): {c: r[c] for c in supl.colnames} for r in supl}
    # Example selection. Every plotted exposure costs a full read (SFrame windows, DAP tables and
    # one gzip pass through the DAP output cube, ~5-6 s), so the examples are taken from ONE random
    # ranking of all exposures: each metric/status uses the n best-ranked exposures with that
    # status, preferring exposures already picked for another metric. Metrics therefore share
    # their examples wherever the statuses allow (~40 exposures to read for n = 5, independent of
    # the run size, instead of up to 16 metrics x 3 statuses x n = 240). Exposures without
    # a DAP output file are used for the DAP-output metrics only if nothing else is available.
    rng = random.Random(seed)
    order = [int(e) for e in qa["expnum"]]
    rng.shuffle(order)
    rank = {e: i for i, e in enumerate(order)}

    def _has(e, key):
        v = sup[e].get(key, True)
        return bool(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else True

    jobs: Dict[int, List[Tuple]] = {}
    picked: Dict[str, Dict[str, List[int]]] = {}
    for metric in EXTRACTORS:
        if metric not in qa.colnames:
            continue
        picked[metric] = {}
        for cls in CLASSES:
            ex = [int(e) for e, f in zip(qa["expnum"], qa[metric]) if str(f) == cls and int(e) in rows
                  and _has(int(e), "incomplete_sframe_found")]
            dapout = metric in NEEDS_OUTPUT or metric == "qa_telluric"
            ex.sort(key=lambda e: (dapout and not _has(e, "incomplete_dap_output_found"),
                                   e not in jobs, rank[e]))  # already-read exposures first
            ex = ex[:n]
            picked[metric][cls] = ex
            for slot, e in enumerate(ex):
                jobs.setdefault(e, []).append((metric, cls, slot, sup[e]))
    log.info("diagnostics PDF: %d panels from %d exposures", sum(len(j) for j in jobs.values()), len(jobs))
    import time
    t0 = time.time()
    panels: Dict[Tuple, Dict[str, Any]] = {}
    args = [(cfg, rows[e], j) for e, j in jobs.items()]
    if nproc > 1:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for res in ex.map(_work, args):
                panels.update(res)
    else:
        for a in args:
            panels.update(_work(a))
    log.info("diagnostics PDF: data of %d exposures read in %.1f s", len(jobs), time.time() - t0)
    t0 = time.time()
    out = Path(out) if out else Path(f"{base}_diagnostics.pdf")
    with PdfPages(out) as pdf:
        for metric, cl in picked.items():
            fig, axes = plt.subplots(3, n, figsize=(11.7, 8.3), squeeze=False)
            for i, cls in enumerate(CLASSES):
                for j in range(n):
                    ax = axes[i, j]
                    key = (metric, cls, j)
                    if key in panels:
                        _draw(ax, panels[key], cls)
                    else:
                        ax.axis("off")
                        if j == 0:
                            ax.text(0.5, 0.5, f"no {cls}", ha="center", va="center", color=CCOL[cls])
                axes[i, 0].annotate(cls, xy=(-0.36, 0.5), xycoords="axes fraction", rotation=90,
                                    va="center", fontsize=10, color=CCOL[cls], fontweight="bold")
            fig.suptitle(f"{metric}  (DRP {drp_version} / DAP {cfg.dap_label}; up to {n} random exposures per status)",
                         fontsize=11)
            ylab = next((panels[k]["ylabel"] for k in sorted(panels) if k[0] == metric and panels[k].get("ylabel")), "")
            fig.text(0.05, 0.005, _criteria_text(metric, cfg), fontsize=6.3, va="bottom", ha="left",
                     family="monospace", color="0.2")
            if ylab:
                if hasattr(fig, "supylabel"):  # matplotlib >= 3.4
                    fig.supylabel(ylab, fontsize=9, x=0.006)
                else:
                    fig.text(0.006, 0.5, ylab, rotation=90, va="center", ha="left", fontsize=9)
            fig.tight_layout(rect=(0.05, 0.12, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
            _hist(pdf, metric, cfg, qa, supl)            # value distributions with the limits (v1.15)
        log.info("diagnostics PDF: %d metric pages drawn in %.1f s", len(picked), time.time() - t0)
        t0 = time.time()
        late = ("qa_fluxmap", "qa_gradient")
        nh = sum(_hist(pdf, m, cfg, qa, supl) for m in _hist_order(qa) if m not in picked and m not in late)
        log.info("diagnostics PDF: %d histogram pages of the other metrics in %.1f s", nh, time.time() - t0)
        t0 = time.time()
        coords = {e: (float(r.get("sci_ra", np.nan)), float(r.get("sci_dec", np.nan))) for e, r in rows.items()}
        try:
            _fluxmap_overview(pdf, qa, supl)
        except Exception as exc:  # noqa: BLE001
            log.warning("diagnostics PDF: fluxmap overview failed: %s", exc)
        _fluxmap_page(pdf, qa, supl, cfg, coords=coords, nproc=nproc)
        _hist(pdf, "qa_fluxmap", cfg, qa, supl)
        try:
            _gradient_page(pdf, qa, supl, cfg, seed=seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("diagnostics PDF: qa_gradient page failed: %s", exc)
        _hist(pdf, "qa_gradient", cfg, qa, supl)
        log.info("diagnostics PDF: fluxmap page in %.1f s", time.time() - t0)
    return out


def main(argv=None):
    """CLI: ``python -m lvmqa.diagnostics`` = re-derive the QA flags from the stored measurements
    (lvmqa.recalc; writes <base>_recalculated.*), PDF only with --pdf."""
    from .recalc import main as recalc_main
    return recalc_main(argv, prog="python -m lvmqa.diagnostics")


if __name__ == "__main__":
    raise SystemExit(main())
