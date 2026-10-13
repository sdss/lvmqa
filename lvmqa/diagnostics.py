"""
Diagnostics of an existing QA run.

1) Diagnostic PDF (make_pdf): for every metric that analyses spectra, up to ``n`` random exposures
   per status (MAJOR ISSUE, WARNING, GOOD) with their SFrame / DAP / DAP-output data and models in
   the relevant wavelength range (one page per metric); last page: Halpha maps of the tiles (DAP
   NP fluxes) with the qa_fluxmap results. Written automatically at the end of every pipeline run
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
        return {f"dg_tell_{k}": (cl[0] - 5, cr[1] + 5, (1, 2, 5, 6)) for k, (b, cl, cr) in drp_telluric.BANDS.items()}
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
    rows = ctx.good_sci_index
    spec = np.asarray(ctx.slitmap["spectrographid"])[rows]
    w, f = ctx.sframe_window("FLUX", wl - 8, wl + 8, rows=rows)
    _, s = ctx.sframe_window("SKY", wl - 8, wl + 8, rows=rows)
    y = f + s
    ref = _nanmed(y)
    p = _panel(f"{name}: worst {spch} {_f(r.get('flatfield_max_rel_offset', np.nan)):.1f}%", ylabel="sky line / all fibers")
    for sp in (1, 2, 3):
        m = spec == sp
        if m.sum():
            p["curves"].append((w, _nanmed(y[m]) / np.nanmax(ref), f"sp{sp} {_f(r.get(f'flatfield_off_sp{sp}_{ch}', np.nan)):+.1f}%",
                                dict(color=SPEC_COL[sp])))
    return p


def _ex_relflux(ctx, r, cls, key, metric):
    """Median-stacked normalised continuum around the channel overlap (adopted signal), masked
    pixels in grey, 10-A bin medians used in the fits, and the straight-line fits of each side
    extrapolated to the overlap centre (the metric fits each fiber; shown here for the stack)."""
    from .metrics import drp_relflux as rf
    from .registry import available_metrics
    pr = _minst(metric, ctx.cfg).params
    ov, blue, red = pr["overlap"], pr["blue_side"], pr["red_side"]
    res = rf.measure_steps(ctx, pr)
    w = res["w"]
    sig = _s(r.get(f"relflux_{key}_signal", ""), "obs")
    if sig == "src" and "stack_src" in res:
        stack, lm, n = res["stack_src"], res["mask_src"].copy(), res.get("n_src_sel", 0)
    else:
        stack, lm, n = res["stack_obs"], res["mask_obs"].copy(), "all"
    for a, b in rf.TELLURIC:
        lm |= (w >= a) & (w <= b)
    x0 = 0.5 * (ov[0] + ov[1])
    p = _panel(f"{key}: step {_f(r.get(f'relflux_{key}_median_step', np.nan)):+.1%} ({sig}, n={n}); "
               f"obs {_f(r.get(f'relflux_{key}_median_step_obs', np.nan)):+.1%}, src "
               f"{_f(r.get(f'relflux_{key}_median_step_src', np.nan)):+.1%}",
               ylabel="median normalised flux (src: FLUX of bright fibers; obs: FLUX+SKY)")
    p["curves"].append((w, np.where(~lm, stack, np.nan), "unmasked pixels", dict(color="k", lw=0.6)))
    p["curves"].append((w, np.where(lm, stack, np.nan), "masked (lines, tellurics)", dict(color="0.78", lw=0.4)))
    fits_y = []
    for side, col, lab in ((blue, "#2a78d6", "blue side"), (red, "#d03b3b", "red side")):
        msk = (w >= side[0]) & (w <= side[1]) & ~lm
        xb, yb = rf._binned(w, stack[None, :], msk, 10.0)
        if xb.size >= 4:
            yb = yb[0]
            c = np.polyfit(xb - x0, yb, 1)
            p["curves"].append((xb, yb, f"{lab}: 10-A bin medians", dict(color=col, ls="none", marker="o", ms=2.5)))
            xx = np.array([min(xb.min(), x0), max(xb.max(), x0)])
            yy = np.polyval(c, xx - x0)
            p["curves"].append((xx, yy, f"{lab} fit", dict(color=col, lw=1.4)))
            fits_y += list(yy) + list(yb)
    p["spans"] += [tuple(ov), (blue[0], blue[1], "#cfe0f5"), (red[0], red[1], "#f6d5d2")]
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
    names = lines_of(ch)
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
    p = _panel(f"{b}: EW {ews[b]:+.2f} A, rms {rms[b]:.3f} (method {meth}; EW dap "
               f"{_f(r.get(f'telluric_ew_dap_{b}', np.nan)):+.2f}, interp {_f(r.get(f'telluric_ew_interp_{b}', np.nan)):+.2f})",
               ylabel="normalised flux; ratio - 0.3")

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
    names = BRIGHT.get(key, BRIGHT["Halpha"])
    wd, pl = ctx.dap_output(f"dg_{key}")
    F = sum(np.nan_to_num(np_column(ctx, f"flux_{n}")) for n in names)
    EF = np.sqrt(sum(np.nan_to_num(np_column(ctx, f"e_flux_{n}")) ** 2 for n in names))
    _, ivar = ctx.dap_sframe_window("IVAR", wd[0], wd[-1])
    ivar = ivar[:, :wd.size]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (EF > 0) & (F / EF > 10)
        r_pm = np.where(ivar > 0, pl[0] - pl[8], np.nan)      # masked pixels excluded (as in the metric)
        num = np.nansum(np.abs(r_pm), axis=1)
        den = np.nansum(pl[7], axis=1)
        fx = np.where(ok & (den > 0), num / den, np.nan)
    rng = np.random.default_rng(int(ctx.drp("expnum", 0)))
    i = _pick(fx, cls, rng, ok & (fx < np.nanpercentile(fx, 75)))
    ids = np.asarray(ctx.dap_table("PT")["id"]).astype(str)
    _, mk = ctx.dap_sframe_window("MASK", wd[0], wd[-1])
    p = _panel(f"{key} fiber {ids[i]}: |resid|/model {fx[i]:.2f}; exp: frac_poor "
               f"{_f(r.get('dap_pmfit_frac_poor', np.nan)):.2f}", ylabel="flux (data - stellar)")
    p["curves"] += [(wd, pl[0][i] - (pl[8][i] - pl[7][i]), "data - stellar", dict(color="k", drawstyle="steps-mid")),
                    (wd, pl[7][i], "PM line model", dict(color="#d03b3b")),
                    (wd, r_pm[i], "residual", dict(color="#4a3aa7", lw=0.8))]
    _mark_masked(p, wd, mk[:, :wd.size], i)
    p["hlines"].append(0.0)
    return p


def ex_npfluxes(ctx, r, cls):
    from .metrics.dap_npfluxes import GROUPS
    from .metrics._dapout import np_column
    g = _s(r.get("dap_npfluxes_worst_group", ""), "Halpha6563")
    if g not in GROUPS:
        g = "Halpha6563"
    prim, (w0, w1) = GROUPS[g]
    wd, pl = ctx.dap_output(f"dg_fl_{g}")
    F, E = np_column(ctx, f"flux_{prim}"), np_column(ctx, f"e_flux_{prim}")
    v = np_column(ctx, "vel_Halpha_6562.85")
    with np.errstate(invalid="ignore", divide="ignore"):
        sel = np.where((E > 0) & (F / E >= 3) & np.isfinite(v))[0]
    p = _panel(f"{g}: net {_f(r.get(f'dap_npfluxes_net_{g}', np.nan)):+.2f} (n={sel.size})", xlabel="rest wavelength [A]",
               ylabel="stacked flux")
    if sel.size:
        dw = float(wd[1] - wd[0])
        sh = np.round(0.5 * (w0 + w1) * v[sel] / C_KMS / dw).astype(int)
        x = np.arange(wd.size)
        idx = np.clip(x[None, :] + sh[:, None], 0, wd.size - 1)
        R = np.nansum(np.take_along_axis(pl[4][sel], idx, axis=1), axis=0)
        G = np.nansum(np.take_along_axis(pl[6][sel], idx, axis=1), axis=0)
        p["curves"] += [(wd, R + G, "data (stack)", dict(color="k", drawstyle="steps-mid")),
                        (wd, G, "NP model (all lines)", dict(color="#2a78d6")),
                        (wd, R, "residual", dict(color="#d03b3b", lw=0.8))]
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
    from .metrics._dapout import np_column
    from .metrics.dap_gaskinematics import REST
    pair = _s(r.get("gaskinematics_worst_pair", ""), "H_Hbeta4861")
    other = {"Hbeta4861": ("Hbeta", "Hbeta_4861.36"), "Hgamma4340": ("Hgamma", "Hgamma_4340.49"),
             "HI9229": ("P9", "HI_9229.02"), "SIII9531": ("SIII9531", "[SIII]_9531.1"),
             "HeI6678": ("HeI6678", "HeI_6678.15"), "HeI7065": ("HeI7065", "HeI_7065.19")}
    ref = {"H": ("Halpha", "Halpha_6562.85"), "SIII": ("SIII9069", "[SIII]_9069.0"), "HeI": ("HeI5876", "HeI_5876.0")}
    g, _, o = pair.partition("_")
    a = ref.get(g, ref["H"])
    b = other.get(o, other["Hbeta4861"])
    p = _panel(f"{_lname(b[1])} - {_lname(a[1])}: dv {_f(r.get(f'gaskinematics_dv_{pair}', np.nan)):+.1f} km/s; "
               f"max spectrograph-boundary offset {_f(r.get('gaskinematics_max_boundary_dv', np.nan)):.1f} km/s; "
               f"Ha outliers {_f(r.get('gaskinematics_frac_out_v', np.nan)):.3f}",
               xlabel="velocity w.r.t. rest (air) wavelength [km/s]", ylabel="normalised gas profile (stack)")
    s1 = np_column(ctx, f"flux_{a[1]}") / np_column(ctx, f"e_flux_{a[1]}")
    s2 = np_column(ctx, f"flux_{b[1]}") / np_column(ctx, f"e_flux_{b[1]}")
    sel = np.where((s1 > 10) & (s2 > 10))[0]
    for (key, line), col in ((a, "k"), (b, "#d03b3b")):
        wd, pl = ctx.dap_output(f"dg_{key}")
        rest = REST.get(line, float(line.rsplit("_", 1)[1]))
        vv = (wd / rest - 1) * C_KMS
        if sel.size:
            prof = _nanmed(pl[3][sel] / np.nanmax(pl[3][sel], axis=1, keepdims=True))
            m = np.abs(vv) < 300
            p["curves"].append((vv[m], prof[m], _lname(line), dict(color=col)))
    p["vlines"].append(0.0)
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
    bins = np.linspace(0.5, 1.5, 51)
    xc = 0.5 * (bins[1:] + bins[:-1])
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
                stat = (f"{_f(r.get(f'lineratios_{t}_dev', np.nan)):+.0f}%" if t in DOUBLETS
                        else f"{_f(r.get(f'lineratios_{t}_fbelow', np.nan)):.2f} below")
                parts.append(f"{lab[t]} {stat} ({flag.replace(' ISSUE', '')})")
            if y.size >= 5:
                h, _ = np.histogram(np.clip(y, bins[0], bins[-1]), bins=bins)
                curves.append((xc, h / max(h.max(), 1), f"{lab[t]} (n={y.size})",
                               dict(color=col[t], drawstyle="steps-mid")))
    p = _panel("; ".join(parts), xlabel="R_obs / R_expected (Balmer: Case B)",
               ylabel="fibers (normalised histogram)")
    p["curves"] += curves
    p["vlines"].append(1.0)
    p["robust_ylim"] = False
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


def _fluxmap_page(pdf, qa, supl, cfg, max_panels=6, coords=None):
    """Halpha maps of the tiles, one panel per sky region (tiles within 2 deg), max_panels largest.

    Regions are found from the tile centres (``coords`` = {expnum: (ra, dec)} from drpall; DAP
    fiber positions otherwise); DAP maps are read only for the exposures that are drawn (first
    exposure of each pointing in the shown regions), so the page stays cheap for large runs."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .metrics.dap_fluxmap import _payload_from_dap
    from .paths import dap_path
    coords = coords or {}
    cand, files = [], []
    for row in qa:
        f = dap_path(cfg, int(row["tileid"]), int(row["mjd"]), int(row["expnum"]))
        if Path(f).is_file():
            cand.append(row)
            files.append(f)
    if not cand:
        return

    class _Maps(dict):  # lazily read DAP Halpha maps: pts[i] = (row, payload, flux)
        def __missing__(self, i):
            pl = _payload_from_dap(files[i])
            v = (cand[i], pl, np.asarray(pl["flux"], float))
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
    groups = [gidx[g] for g in _clusters(ra0[good], de0[good])]
    shown = groups[:max_panels]
    nc = 1 if len(shown) == 1 else (2 if len(shown) <= 4 else 3)
    nr = int(np.ceil(len(shown) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11.7, 8.3), squeeze=False)
    by = {int(r["expnum"]): r for r in supl}
    sc = None
    for ax, g in zip(axes.flat, shown):
        first = [g[sub[0]] for sub in _clusters(ra0[g], de0[g], link=0.05)]
        norm = _norm(first)
        for i in first:
            row, pl, lf = pts[i]
            sc = ax.scatter(pl["ra"], pl["dec"], c=lf, s=3 if len(first) > 3 else 12, cmap="magma",
                            norm=norm, lw=0)
        # repeated pointings (centres < 0.05 deg apart) share one outline (worst status) and label
        sev = {"MAJOR ISSUE": 3, "WARNING": 2, "GOOD": 1}
        for sub in _clusters(ra0[g], de0[g], link=0.05):
            idx = g[sub]
            lines, worst = [], ""
            for i in idx:
                row = cand[i]
                e = int(row["expnum"])
                st = str(row["qa_fluxmap"]) if "qa_fluxmap" in qa.colnames else ""
                sv = _f(by[e]["fluxmap_s_dex"]) if e in by and "fluxmap_s_dex" in supl.colnames else np.nan
                lines.append(f"{e} {sv:+.3f}" + ("" if st == "GOOD" else f" {st[:4]}"))
                worst = st if sev.get(st, 0) > sev.get(worst, 0) else worst
            pl = pts[idx[0]][1]
            ok = np.isfinite(pl["ra"]) & np.isfinite(pl["dec"])
            h = _hull(np.c_[pl["ra"][ok], pl["dec"][ok]])
            if len(h) > 2:
                ax.plot(np.r_[h[:, 0], h[0, 0]], np.r_[h[:, 1], h[0, 1]], color=CCOL.get(worst, "0.5"), lw=1.8)
            ax.text(np.median(ra0[idx]), np.median(de0[idx]), "\n".join(lines), ha="center", va="center",
                    fontsize=5.5 if len(idx) < 6 else 4.5, color="k", family="monospace",
                    bbox=dict(facecolor=CCOL.get(worst, "0.8"), alpha=0.5, lw=0, pad=1.2))
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        ax.invert_xaxis()
        ax.set_aspect(1.0 / np.cos(np.deg2rad(np.median(de0[g]))), adjustable="datalim")
        ax.set_xlabel("RA [deg]", fontsize=7)
        ax.set_ylabel("Dec [deg]", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{len(g)} exposure(s) around RA {np.median(ra0[g]):.1f}, Dec {np.median(de0[g]):+.1f}",
                     fontsize=7)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=CCOL[c], lw=2, label=c) for c in CLASSES] +
               [Line2D([], [], color="0.5", lw=2, label="UNDEFINED")], loc="lower center", ncol=4, fontsize=7,
               title="outline = qa_fluxmap (worst of repeated pointings); label = expnum, s [dex]; map = first "
                     "exposure; colour = F(Halpha) [1e-16 erg/s/cm2] (DAP NP; asinh stretch, own scale per panel)",
               title_fontsize=7, frameon=False)
    fig.subplots_adjust(bottom=0.13)
    extra = f" ({len(groups) - len(shown)} more regions not shown)" if len(groups) > len(shown) else ""
    fig.suptitle("qa_fluxmap: Halpha maps of the tiles (DAP NP), fitted flux-scale offsets s and status" + extra,
                 fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


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
    base = Path(cfg.output_dir) / output_basename(cfg, tag, suffix)
    if not Path(f"{base}.fits").is_file():
        raise FileNotFoundError(f"{base}.fits not found (check --output-dir / --tag / --suffix)")
    qa = Table.read(f"{base}.fits", hdu="QA")
    supl = Table.read(f"{base}_supl.fits", hdu="SUPL")
    from .pipeline import load_drpall, select_exposures, _row_to_dict
    drpall = load_drpall(cfg)
    rows = {int(r["expnum"]): _row_to_dict(r) for r in select_exposures(drpall, list(qa["expnum"]))}
    sup = {int(r["expnum"]): {c: r[c] for c in supl.colnames} for r in supl}
    rng = random.Random(seed)
    jobs: Dict[int, List[Tuple]] = {}
    picked: Dict[str, Dict[str, List[int]]] = {}
    for metric in EXTRACTORS:
        if metric not in qa.colnames:
            continue
        picked[metric] = {}
        for cls in CLASSES:
            ex = [int(e) for e, f in zip(qa["expnum"], qa[metric]) if str(f) == cls and int(e) in rows
                  and bool(sup[int(e)].get("incomplete_sframe_found", True))]
            ex = rng.sample(ex, min(n, len(ex)))
            picked[metric][cls] = ex
            for slot, e in enumerate(ex):
                jobs.setdefault(e, []).append((metric, cls, slot, sup[e]))
    panels: Dict[Tuple, Dict[str, Any]] = {}
    args = [(cfg, rows[e], j) for e, j in jobs.items()]
    if nproc > 1:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            for res in ex.map(_work, args):
                panels.update(res)
    else:
        for a in args:
            panels.update(_work(a))
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
        coords = {e: (float(r.get("sci_ra", np.nan)), float(r.get("sci_dec", np.nan))) for e, r in rows.items()}
        _fluxmap_page(pdf, qa, supl, cfg, coords=coords)
    return out


def main(argv=None):
    """CLI: ``python -m lvmqa.diagnostics`` = re-derive the QA flags from the stored measurements
    (lvmqa.recalc; writes <base>_recalculated.*), PDF only with --pdf."""
    from .recalc import main as recalc_main
    return recalc_main(argv, prog="python -m lvmqa.diagnostics")


if __name__ == "__main__":
    raise SystemExit(main())
