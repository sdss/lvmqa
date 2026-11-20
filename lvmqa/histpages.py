"""
Diagnostics PDF (v1.15): distributions of the stored metric values of a run with the adopted
GOOD / WARNING / MAJOR limits.

One histogram per quantity that enters a flag (per spectral channel / spectrograph / band / line
where these are evaluated separately); one entry per exposure (values from the SUPL table; per-fiber
values are not stored). Bars are stacked by the exposure's status in that metric. Limits are drawn
as vertical lines (WARNING orange, MAJOR red; dotted = reference value that is applied per fiber /
per point / in combination with other conditions, not directly to this per-exposure value).
Statistics per panel: n, median, mean, std, NMAD, mode (peak of a smoothed histogram), 5-95%
range, and for every limit: fraction of exposures beyond it and its distance from the median in
units of the NMAD (z = (limit - median) / NMAD).

Specs per metric in SPECS: functions of the metric parameters returning panels
    dict(col=<value name>, label=..., lim=[(value, level, side)], ref=[(value, label)], scale=1)
with level 'W' / 'M' / 'R' (reference) and side '+' (above is worse), '-' (below is worse) or
'|' (|x| above is worse, lines at +-value).
"""

from __future__ import annotations

import textwrap
from typing import Callable, Dict, List

import numpy as np

CCOL = {"MAJOR ISSUE": "#d03b3b", "WARNING": "#fab219", "GOOD": "#0ca30c", "UNDEFINED": "#c3c2b7"}
LCOL = {"W": "#e69500", "M": "#d03b3b", "R": "0.35"}


def _p(col, label=None, lim=(), ref=(), scale=1.0, unit="", flag=None):
    """flag: SUPL value holding the status of this sub-test (colours the bars instead of the metric flag)."""
    return dict(col=col, label=label or col, lim=list(lim), ref=list(ref), scale=scale, unit=unit, flag=flag)


def _two(col, w, m, label=None, **kw):
    return _p(col, label, [(w, "W", "|")] + ([(m, "M", "|")] if m is not None else []), **kw)


def _up(col, w, m, label=None, **kw):
    lim = ([(w, "W", "+")] if w is not None else []) + ([(m, "M", "+")] if m is not None else [])
    return _p(col, label, lim, **kw)


SKY_LINES_SKYLINE = ("OI5577", "OH6865", "NaD", "O2")


def _relflux(p):
    return [_two("median_step", p["step_warn"], p["step_major"], "step J (adopted signal)"),
            _two("median_kink", p["kink_warn"], p["kink_major"], f"slope mismatch K (at +-{p['lever']:g} A; 0 if vetoed)"),
            _two("median_kink_raw", p["kink_warn"], p["kink_major"], "slope mismatch of the side lines (before veto)"),
            _two("kink_stack", p["kink_warn"], p["kink_major"], "slope mismatch of the stack"),
            _p("median_gradient", "common gradient per 100 A")]


def _wavecal(p, ch):
    out = [_up("frac_warn", p["frac_thresh"], None, f"fraction of fibers off by > {p['warn_lsf_frac']:g} LSF (or failed)"),
           _up("frac_major", None, p["frac_thresh"], f"fraction of fibers off by > {p['major_lsf_frac']:g} LSF (or failed)"),
           _p("p95_rel_offset", "95th pct over fibers of max |offset| / LSF",
              ref=[(p["warn_lsf_frac"], "per-fiber W"), (p["major_lsf_frac"], "per-fiber M")]),
           _p("frac_failed", "fraction of fibers with < half of the lines fitted", ref=[(p["frac_thresh"], "M if above")]),
           _p("max_spec_offset", "largest |median offset| of a spectrograph [A]")]
    from .metrics._skylines import lines_of
    out += [_p(f"off_{n}", f"median offset {n} [A]" + (" (excluded)" if n in tuple(p.get("exclude", ())) else ""))
            for n in lines_of(ch)]
    out += [_p(f"drp_skyoff_sp{s}", f"DRP sky-line shift sp{s} [A] (header)") for s in (1, 2, 3)]
    return out


SPECS: Dict[str, Callable[[dict], List[dict]]] = {
    "qa_illumination": lambda p: [
        _up("sun_alt", None, p["twilight_sun_alt"], "Sun altitude [deg]"),
        _p("moon_fli", "Moon illumination FLI", ref=[(p["fli_warn"], "W if Moon up & near"),
                                                     (p["fli_major"], "M if Moon up & near")]),
        _p("moon_alt", "Moon altitude [deg]", ref=[(p["moon_alt_min"], "horizon limit")]),
        _p("moon_sep", "Moon separation [deg]", ref=[(p["moon_sep_min"], "near if below")])],
    "qa_brightstars": lambda p: [
        _p("brightest_vmag", "V of the brightest star in the field", [(p["mag_warn"], "W", "-"), (p["mag_major"], "M", "-")]),
        _p("brightest_near_vmag", "V of the brightest star just outside", ref=[(p["mag_warn"], "mag_warn")])],
    "qa_guider": lambda p: [
        _up("fwhm_med", p["fwhm_warn"], p["fwhm_major"], "median guider FWHM [arcsec]"),
        _p("zeropt", "guider zero point"), _p("padrift", "PA drift"), _p("nframes", "guider frames")],
    "qa_incomplete": lambda p: [
        _up("frac_lost", p["warn_frac_lost"], None, "fraction of science fibers lost (any channel)"),
        *[_up(f"frac_lost_{c}", p["warn_frac_lost"], p["major_frac_lost_channel"], f"fraction lost, {c}") for c in "brz"],
        _up("max_frac_lost_spec", None, p["major_frac_lost_spec"], "largest fraction lost in one spectrograph"),
        _p("dap_coverage", "DAP coverage of the usable fibers", [(p["min_dap_coverage"], "W", "-")])],
    "qa_lineratios": lambda p: [
        _two("oiii_dev", p["warn_dev_oiii"], p["major_dev_oiii"], "[OIII] 5007/4959: median R/R_th - 1 [%]", flag="oiii_flag"),
        _two("nii_dev", p["warn_dev_nii"], p["major_dev_nii"], "[NII] 6584/6548: median R/R_th - 1 [%]", flag="nii_flag"),
        _two("siii_dev", p["warn_dev_siii"], p["major_dev_siii"], "[SIII] 9531/9069 (reddened theory) [%]", flag="siii_flag"),
        _two("siii_dev_raw", p["warn_dev_siii"], p["major_dev_siii"], "[SIII] 9531/9069 uncorrected (softening check) [%]",
             flag="siii_flag"),
        _up("hahb_fbelow", p["balmer_warn_frac"], p["balmer_major_frac"], "Ha/Hb: fraction of fibers below Case B",
            flag="hahb_flag"),
        _up("hbhg_fbelow", p["balmer_warn_frac"], p["balmer_major_frac"], "Hb/Hg: fraction of fibers below Case B",
            flag="hbhg_flag"),
        _p("hahb_med", "Ha/Hb median ratio", ref=[(p["ratio_hahb"], "Case B")]),
        _p("hbhg_med", "Hb/Hg median ratio", ref=[(p["ratio_hbhg"], "Case B")]),
        _p("av_med", "median A_V (from Ha/Hb) [mag]")],
    "qa_absflux": lambda p: [
        *[_up(f"z_{c}", p["z_thresh"], None, f"MOD vs SCI sensitivity, {c}: z") for c in "brz"],
        _up("n_outliers", 0.5, p["n_major"] - 0.5, "channels with z > z_thresh")],
    "qa_relflux_br": _relflux,
    "qa_relflux_rz": _relflux,
    "qa_skyline_sub": lambda p: [
        _two(f"{l}_sp{s}", p["warn_any"], p["bad_any"], f"{l} residual / sky, sp{s} (colour: status of sp{s})",
             flag=f"flag_sp{s}")
        for l in SKY_LINES_SKYLINE for s in (1, 2, 3)] + [
        _two("max_abs_resid", p["warn_any"], p["bad_any"], "largest |residual| / sky (exc. OI5577)")],
    "qa_skycont_sub": lambda p: [
        _p(f"floor_sp{s}_{c}", f"continuum floor sp{s} {c} [S/N]", [(p["snr_warn"], "W", "|"), (-p["snr_major"], "M", "-")])
        for c in "brz" for s in (1, 2, 3)] + [
        _up("spec_spread", p["spread_warn"], p["spread_major"], "spread of the spectrograph floors [S/N]")],
    "qa_flatfield": lambda p: [
        _two(f"off_sp{s}_{c}", 100 * p["warn_rel"], 100 * p["major_rel"], f"sky-line flux offset sp{s} {c} [%]")
        for c in "brz" for s in (1, 2, 3)] + [
        _p("fiber_scatter", "per-fiber scatter of the sky-line fluxes")],
    "qa_wavecal_b": lambda p: _wavecal(p, "b"),
    "qa_wavecal_r": lambda p: _wavecal(p, "r"),
    "qa_wavecal_z": lambda p: _wavecal(p, "z"),
    "qa_lsf": lambda p: [
        *[_up(f"frac_warn_{c}", p["frac_thresh"], None, f"fraction of fibers |Q-1| > {p['warn_tol']:g}, {c}") for c in "brz"],
        *[_up(f"frac_major_{c}", None, p["frac_thresh"], f"fraction of fibers |Q-1| > {p['major_tol']:g}, {c}") for c in "brz"],
        *[_p(f"q_med_{c}", f"median Q, {c}", ref=[(1 - p["warn_tol"], "per-fiber W"), (1 + p["warn_tol"], "per-fiber W")])
          for c in "brz"]],
    "qa_telluric": lambda p: [
        *[_two(f"ew_{b}", p["ew_warn"], p["ew_major"], f"EW {b} (adopted) [A]") for b in ("O2B", "H2O72", "H2O82")],
        *[_up(f"rms_{b}", p["rms_warn"], p["rms_major"], f"excess rms {b}") for b in ("O2B", "H2O72", "H2O82")],
        *[_p(f"ew_interp_{b}", f"EW {b}, local interpolation [A]", ref=[(-p["ew_warn"], ""), (p["ew_warn"], "")])
          for b in ("O2B", "H2O72", "H2O82")]],
    "qa_gradient": lambda p: [
        _up("spread_dex", p["warn_dex"], p["major_dex"], "spread of the spectrograph offsets [dex]"),
        *[_two(f"sp{s}_dex", p["warn_dex"], p["major_dex"], f"offset of sp{s} [dex]") for s in (1, 2, 3)],
        *[_p(f"{k}_dex", f"boundary jump {k} [dex]") for k in ("d12", "d23", "d31")],
        _p("closure_dex", "closure d12 + d23 + d31 [dex]", ref=[(0.0, "0")]),
        _up("ifu_dipole_dex", p["warn_dex"], p["major_dex"], "mosaic IFU dipole [dex] (run value)")],
    "qa_fit_consistency": lambda p: [
        _up("frac_bad", p["warn_frac_bad"], p["major_frac_bad"], "fraction of discrepant PM/NP points"),
        _up("frac_masked_core", p["warn_frac_masked"], p["major_frac_masked"], "fraction of masked line cores"),
        _p("median_z", "median |F_pm - F_np| / sigma", ref=[(p["nsigma"], "per-point limit")]),
        *[_p(f"fbad_{k}", f"discrepant fraction {k}", ref=[(p["warn_frac_bad"], "W level"), (p["major_frac_bad"], "M level")])
          for k in ("OII3726", "OII3729", "Hbeta4861", "OIII5007", "Halpha6563", "NII6583", "SII6716", "SII6731")]],
    "qa_dap_pmfit": lambda p: [
        _up("frac_poor", p["frac_thresh"], None, f"fraction of points with fx > {p['fx_warn']:g}"),
        _up("frac_bad", None, p["frac_thresh"], f"fraction of points with fx > {p['fx_major']:g}"),
        _p("median_fx", "median unmodelled flux fraction fx (PM)", ref=[(p["fx_warn"], "per-point W"), (p["fx_major"], "per-point M")]),
        _p("median_fx_np", "median fx (NP model, reference)"),
        *[_p(f"fx_{k}", f"median fx {k}", ref=[(p["fx_warn"], "per-point W"), (p["fx_major"], "per-point M")])
          for k in ("OII3727", "Hbeta", "OIII5007", "Halpha", "NII6583", "SII6716", "SII6731")]],
    "qa_gaskinematics": lambda p: [
        *[_two(f"dv_{k}", p["dv_warn"], p["dv_major"], f"dv {k} [km/s]")
          for k in ("H_Hbeta4861", "H_Hgamma4340", "H_HI9229", "H_HI9015", "SIII_SIII9531", "HeI_HeI6678", "HeI_HeI7065")],
        _up("frac_out_v", p["frac_out_warn"], p["frac_out_major"], "fraction of velocity outliers (Halpha)"),
        _up("frac_out_disp", p["frac_out_warn"], p["frac_out_major"], "fraction of dispersion outliers"),
        *[_two(f"dv_sp{s}", p["boundary_warn"], p["boundary_major"], f"boundary offset in v, sp{s} [km/s]") for s in (1, 2, 3)],
        *[_two(f"ddisp_sp{s}", p["boundary_warn"], p["boundary_major"], f"boundary offset in sigma, sp{s} [km/s]") for s in (1, 2, 3)]],
    "qa_dap_npfluxes": lambda p: [
        _two(f"net_{k}", p["net_warn"], p["net_major"], f"net residual {k}")
        for k in __import__("lvmqa.metrics.dap_npfluxes", fromlist=["GROUPS"]).GROUPS],
    "qa_absorption": lambda p: [
        _two(f"dI_{k}", p["tol_warn"], p["tol_major"], f"dI {k} [A]")
        for k in __import__("lvmqa.metrics.dap_absorption", fromlist=["FEATURES"]).FEATURES],
    "qa_fluxmap": lambda p: [
        _two("local_dex", p["warn_dex"], p["major_dex"], "local Halpha flux-scale offset l [dex]"),
        _p("s_dex", "global solution s [dex]"), _p("night_dex", "median l of the night [dex]"),
        _p("e_local_tot_dex", "error of l used for the significance [dex]"),
        _p("mult_dex", "flux scale from F = g F_ref + c [dex]"), _p("add_frac", "additive term / median flux")],
    "qa_starkinematics": lambda p: [
        _up("max_frac_out", p["frac_warn"], p["frac_major"], "largest fraction of outliers in a region"),
        _p("dv_star_gas", "v(stars) - v(Halpha) [km/s]", [(p["dv_gas"], "W", "|")]),
        _p("v_nmad", "NMAD of stellar velocities [km/s]")],
}


def _stats(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    med = float(np.median(x))
    nmad = 1.4826 * float(np.median(np.abs(x - med)))
    lo, hi = np.percentile(x, [5, 95])
    # mode: peak of a histogram over the central 98%, smoothed with a 3-bin box
    a, b = np.percentile(x, [1, 99])
    mode = med
    if b > a and x.size >= 10:
        h, e = np.histogram(x[(x >= a) & (x <= b)], bins=max(10, min(60, x.size // 10)))
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        k = int(np.argmax(hs))
        mode = float(0.5 * (e[k] + e[k + 1]))
    return dict(n=x.size, med=med, mean=float(np.mean(x)), std=float(np.std(x)), nmad=nmad, mode=mode, p5=lo, p95=hi)


def _beyond(x, v, side):
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    if side == "+":
        return float(np.mean(x > v))
    if side == "-":
        return float(np.mean(x < v))
    return float(np.mean(np.abs(x) > v))


def _fmt(v):
    if not np.isfinite(v):
        return "nan"
    a = abs(v)
    return f"{v:.3g}" if (a >= 1e-3 and a < 1e4) or a == 0 else f"{v:.2e}"


def _panel(ax, x, status, spec):
    lims, refs = spec["lim"], spec["ref"]
    fin = np.isfinite(x)
    st = _stats(x)
    if st is None:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.5")
        ax.set_title(spec["label"], fontsize=6.5, loc="left")
        ax.set_yticks([])
        return
    xf = x[fin]
    lo, hi = np.percentile(xf, [0.5, 99.5])
    # heavy tails (e.g. fx ~ 1e10 from failed fits): limit the axis to 20 x IQR around the quartiles
    q1, q3 = np.percentile(xf, [25, 75])
    if q3 > q1:
        lo, hi = max(lo, q1 - 20 * (q3 - q1)), min(hi, q3 + 20 * (q3 - q1))
    # the axis includes a limit only if it is not far outside the bulk of the data (> 1.5 x the 0.5-99.5%
    # range): otherwise the distribution would shrink to a spike; such limits are marked at the axis edge
    span = max(hi - lo, 1e-12)
    d_lo, d_hi = lo, hi
    for v, lvl, side in lims:
        for vv in ((v, -v) if side == "|" else (v,)):
            if d_lo - 1.5 * span <= vv <= d_hi + 1.5 * span:
                lo, hi = min(lo, vv), max(hi, vv)
    for v, _ in refs:
        if d_lo - 1.5 * span <= v <= d_hi + 1.5 * span:
            lo, hi = min(lo, v), max(hi, v)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    nb = int(np.clip(np.sqrt(xf.size) * 2, 15, 60))
    edges = np.linspace(lo, hi, nb + 1)
    xc = np.clip(xf, edges[0], edges[-1])
    stf = status[fin]
    bottom = np.zeros(nb)
    for cls in ("GOOD", "WARNING", "MAJOR ISSUE", "UNDEFINED"):
        m = stf == cls
        if m.any():
            h, _ = np.histogram(xc[m], bins=edges)
            ax.bar(0.5 * (edges[1:] + edges[:-1]), h, width=np.diff(edges), bottom=bottom, color=CCOL[cls],
                   lw=0, alpha=0.85)
            bottom += h
    n_out = int(np.sum((xf < edges[0]) | (xf > edges[-1])))
    ymax = bottom.max() if bottom.size else 1
    ax.set_ylim(0, ymax * (1.6 + 0.12 * len(lims)) + 0.5)      # room for the statistics box
    lines = []
    sig = st["nmad"] if st["nmad"] > 0 else st["std"]
    off = []
    for v, lvl, side in lims:
        for vv in ((v, -v) if side == "|" else (v,)):
            if lo <= vv <= hi:
                ax.axvline(vv, color=LCOL[lvl], lw=1.1)
            else:
                off.append((vv, lvl))
        frac = _beyond(xf, v, side)
        z = (v - (abs(st["med"]) if side == "|" else st["med"])) / sig if sig > 0 else np.nan
        sym = {"+": ">", "-": "<", "|": "|x|>"}[side]
        lines.append(f"{'WARN' if lvl == 'W' else 'MAJ '} {sym}{_fmt(v)}: {100 * frac:.1f}% beyond, z={_fmt(z)}")
    for vv, lvl in off:     # limit outside the plotted range: arrow at the axis edge
        right = vv > hi
        ax.annotate(f"{'W' if lvl == 'W' else 'M'} {_fmt(vv)}", xy=(1.0 if right else 0.0, 0.08 if lvl == "W" else 0.16),
                    xycoords="axes fraction", ha="right" if right else "left", fontsize=4.8, color=LCOL[lvl],
                    xytext=(-9 if right else 9, 0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=LCOL[lvl], lw=0.8))
    for v, lab in refs:
        if not lo <= v <= hi:
            continue
        ax.axvline(v, color=LCOL["R"], lw=0.8, ls=":")
        if lab:
            ax.text(v, ymax * 1.05, lab, rotation=90, fontsize=4.2, va="bottom", ha="right", color="0.3")
    ax.axvline(st["med"], color="k", lw=0.7, ls="--")
    txt = (f"n={st['n']}  med={_fmt(st['med'])}  mode={_fmt(st['mode'])}\nmean={_fmt(st['mean'])}  "
           f"std={_fmt(st['std'])}  NMAD={_fmt(st['nmad'])}\n5-95%: {_fmt(st['p5'])} .. {_fmt(st['p95'])}"
           + (f"  ({n_out} outside, in end bins)" if n_out else ""))
    if lines:
        txt += "\n" + "\n".join(lines)
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, fontsize=4.6, va="top", ha="left", family="monospace",
            bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=0.5))
    ax.set_title("\n".join(textwrap.wrap(spec["label"], 58)), fontsize=6.3, loc="left")
    ax.tick_params(labelsize=5.5)


def hist_pages(pdf, metric, params, qa, supl, short, criteria_text="", ncols=4, nrows=3, title_extra=""):
    """Histogram page(s) of one metric (see module docstring). Returns the number of pages."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if metric not in SPECS or metric not in qa.colnames:
        return 0
    def col(name):
        out = np.full(len(supl), np.nan)
        for i, v in enumerate(supl[name]):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                pass
        return out

    try:
        specs = [s for s in SPECS[metric](params) if f"{short}_{s['col']}" in supl.colnames]
    except Exception:  # noqa: BLE001
        return 0
    data = {s["col"]: col(f"{short}_{s['col']}") * s["scale"] for s in specs}
    specs = [s for s in specs if np.isfinite(data[s["col"]]).any()]   # quantities not measured in this run: no panel
    if not specs:
        return 0
    by = {int(e): str(f) for e, f in zip(qa["expnum"], qa[metric])}
    status = np.array([by.get(int(e), "UNDEFINED") for e in supl["expnum"]])

    per = ncols * nrows
    npages = 0
    for k0 in range(0, len(specs), per):
        chunk = specs[k0:k0 + per]
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.7, 8.3), squeeze=False)
        for ax, spec in zip(axes.flat, chunk):
            stt = status
            fc = f"{short}_{spec['flag']}" if spec.get("flag") else None
            if fc and fc in supl.colnames:     # status of this sub-test (e.g. per spectrograph / per ratio)
                lab = np.array([str(v).strip() for v in supl[fc]])
                lab = np.where(np.isin(lab, list(CCOL)), lab, "UNDEFINED")
                stt = lab
            _panel(ax, data[spec["col"]], stt, spec)
        for ax in list(axes.flat)[len(chunk):]:
            ax.axis("off")
        pg = f" ({k0 // per + 1}/{(len(specs) - 1) // per + 1})" if len(specs) > per else ""
        cnt = {c: int(np.sum(status == c)) for c in CCOL}
        fig.suptitle(f"{metric}: distributions of the per-exposure values{pg}{title_extra}   "
                     f"[{cnt['GOOD']} G, {cnt['WARNING']} W, {cnt['MAJOR ISSUE']} M, {cnt['UNDEFINED']} U]",
                     fontsize=10, y=0.995)
        handles = [Patch(color=CCOL[c], label=f"exposures {c}") for c in ("GOOD", "WARNING", "MAJOR ISSUE", "UNDEFINED")]
        handles += [Line2D([], [], color=LCOL["W"], lw=1.1, label="WARNING limit"),
                    Line2D([], [], color=LCOL["M"], lw=1.1, label="MAJOR limit"),
                    Line2D([], [], color=LCOL["R"], lw=0.8, ls=":", label="reference (per fiber / point / conditional)"),
                    Line2D([], [], color="k", lw=0.7, ls="--", label="median")]
        fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.955))
        if criteria_text:
            fig.text(0.01, 0.005, criteria_text, fontsize=5.6, va="bottom", ha="left", family="monospace", color="0.2")
        fig.text(0.5, 0.925, "one entry per exposure (SUPL values); z = (limit - median) / NMAD (std if NMAD = 0); 'beyond' = "
                 "fraction of the exposures on the flagged side of the limit", fontsize=5.8, color="0.35", ha="center")
        fig.tight_layout(rect=(0, 0.11, 1, 0.92))
        pdf.savefig(fig)
        plt.close(fig)
        npages += 1
    return npages
