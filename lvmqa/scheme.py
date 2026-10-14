"""
Figure with the scheme of the QA logic: inputs -> measurements -> individual QA metrics
(current thresholds) -> summary flags (qa_obs, science summaries) -> qa_overall / qa_bitmask,
plus the recalculation / comparison paths.

The figure is built from the metric registry, so the metric list, inputs, hard/soft
memberships and threshold values follow the code (thresholds as configured by the
defaults or ``metric_options``).

    python -m lvmqa.scheme [--out DIR] [--set qa_telluric.rms_warn=0.06 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# short description of the measurement and compact threshold templates ({param} = metric option)
MEAS = {
    "qa_illumination": ("Sun altitude, Moon altitude / separation / illumination (FLI)",
                        "Moon up, sep<{moon_sep_min:g}°, FLI≥{fli_warn:g}",
                        "Sun>{twilight_sun_alt:g}°, or Moon up & FLI≥{fli_major:g}"),
    "qa_brightstars": ("V mag of the brightest catalogue star in the field of view",
                       "V<{mag_warn:g}", "V<{mag_major:g}"),
    "qa_guider": ("guider co-add: FWHM, transparency, astrometric solution",
                  "FWHM>{fwhm_warn:g}″ or poor transparency", "FWHM>{fwhm_major:g}″ or no solution"),
    "qa_incomplete": ("fraction of science fibers without usable data; DAP files / coverage",
                      ">{warn_frac_lost:.0%} fibers lost; DAP missing; cov.<{min_dap_coverage:.0%}",
                      "no SFrame; >{major_frac_lost_channel:.0%}/channel or >{major_frac_lost_spec:.0%}/spectr. lost"),
    "qa_lineratios": ("[OIII], [NII], [SIII] doublet ratios vs theory; Hα/Hβ, Hβ/Hγ vs Case B",
                      "dev.>{warn_dev_oiii:g}/{warn_dev_nii:g}/{warn_dev_siii:g}%; Balmer>{balmer_warn_frac:.0%} fib.",
                      "dev.>{major_dev_oiii:g}/{major_dev_nii:g}/{major_dev_siii:g}%; Balmer>{balmer_major_frac:.0%}"),
    "qa_absflux": ("MOD vs SCI flux calibration per channel (z-score)",
                   ">{z_thresh:g}σ in 1 channel", ">{z_thresh:g}σ in ≥{n_major} ch.; no MOD / fluxcal"),
    "qa_relflux_br": ("continuum step + slope mismatch at the b/r overlap (side line fits, source fibers)",
                      "|step|>{step_warn:.0%} or |kink|>{kink_warn:.0%} ({nsig:g}σ)",
                      "|step|>{step_major:.0%} or |kink|>{kink_major:.0%} ({nsig:g}σ)"),
    "qa_relflux_rz": ("continuum step + slope mismatch at the r/z overlap (side line fits, source fibers)",
                      "|step|>{step_warn:.0%} or |kink|>{kink_warn:.0%} ({nsig:g}σ)",
                      "|step|>{step_major:.0%} or |kink|>{kink_major:.0%} ({nsig:g}σ)"),
    "qa_skyline_sub": ("residual / sky flux of OI5577, OH6865, NaD, O2 per spectrograph",
                       "|res/sky|>{warn_any:g} in any", ">{bad_all_lines:g} in all, or >{bad_any:g} in any"),
    "qa_skycont_sub": ("continuum floor of the faintest fibers in inter-line windows (S/N units)",
                       "floor<−{snr_warn:g} or >+{spread_warn:g}", "floor<−{snr_major:g}"),
    "qa_flatfield": ("median relative sky-line flux offset per spectrograph and channel",
                     ">{warn_rel:.0%} (>{nsig:g}σ)", ">{major_rel:.0%} (>{nsig:g}σ)"),
    "qa_wavecal_b": ("sky-line centroid offsets vs LSF per fiber (b)",
                     ">{frac_thresh:.0%} fib. off.>LSF/5", ">{frac_thresh:.0%} fib. off.>LSF"),
    "qa_wavecal_r": ("sky-line centroid offsets vs LSF per fiber (r)",
                     ">{frac_thresh:.0%} fib. off.>LSF/5", ">{frac_thresh:.0%} fib. off.>LSF"),
    "qa_wavecal_z": ("sky-line centroid offsets vs LSF per fiber (z)",
                     ">{frac_thresh:.0%} fib. off.>LSF/5", ">{frac_thresh:.0%} fib. off.>LSF"),
    "qa_lsf": ("Q = (FWHM_sky / LSF) / ref − 1 per fiber and channel",
               ">{frac_thresh:.0%} fib. |Q|>{warn_tol:.0%}", ">{frac_thresh:.0%} fib. |Q|>{major_tol:.0%}"),
    "qa_telluric": ("EW and rms of telluric residuals (O2 B, H2O) vs DAP stellar model",
                    "|EW|>{ew_warn:g} Å or rms>{rms_warn:g}", "|EW|>{ew_major:g} Å or rms>{rms_major:g}"),
    "qa_fit_consistency": ("parametric vs non-parametric bright-line fluxes; masked line cores",
                           ">{warn_frac_bad:.0%} discrepant / >{warn_frac_masked:.0%} masked",
                           ">{major_frac_bad:.0%} discrepant / >{major_frac_masked:.0%} masked"),
    "qa_dap_pmfit": ("unmodelled flux around bright lines in the parametric-fit residuals",
                     ">{frac_thresh:.0%} pts fx>{fx_warn:.0%}", ">{frac_thresh:.0%} pts fx>{fx_major:.0%}"),
    "qa_gaskinematics": ("line-pair Δv, spatial outliers in v / σ, spectrograph-boundary offsets",
                         "|Δv|>{dv_warn:g} km/s; out.>{frac_out_warn:.0%}; bnd>{boundary_warn:g}",
                         "|Δv|>{dv_major:g} km/s; out.>{frac_out_major:.0%}; bnd>{boundary_major:g}"),
    "qa_dap_npfluxes": ("net residual flux in the non-parametric line windows (line groups)",
                        "1 grp |net|>{net_major:.0%}, or >{frac_poor_warn:.0%} grp >{net_warn:.0%}",
                        "≥{frac_bad_major:.0%} grp |net|>{net_major:.0%}"),
    "qa_absorption": ("absorption indices: data vs DAP stellar model (ΔI)",
                      "1 feat. |ΔI|>{tol_major:g} Å, or >{frac_poor_warn:.0%} >{tol_warn:g} Å",
                      "≥{frac_bad_major:.0%} feat. |ΔI|>{tol_major:g} Å"),
    "qa_fluxmap": ("Hα flux-scale offset vs tiles within {local_radius:g}° (overlap graph, IFU dipole fitted)",
                   "|l|>{warn_dex:g} dex ({nsig:g}σ)", "|l|>{major_dex:g} dex ({nsig:g}σ)"),
    "qa_gradient": ("Hα jumps across spectrograph sector boundaries; mosaic IFU dipole",
                    "spread or dipole>{warn_dex:g} dex ({nsig:g}σ)", "spread or dipole>{major_dex:g} dex"),
    "qa_starkinematics": ("stellar v / σ outliers in connected regions; degenerate σ; v★ − v(Hα)",
                          ">{frac_warn:.0%} outl.; degenerate; |Δv|>{dv_gas:g}", ">{frac_major:.0%} outliers"),
    "qa_status": ("detected emission lines / stellar continuum (booleans, information only)", "–", "–"),
}

INPUTS = (("drpall", "drpall"), ("sframe_header", "SFrame\nheader"), ("sframe_data", "SFrame\ndata"),
          ("dap", "DAP\ntables"), ("dap_output", "DAP\noutput"))
FCOL = {"GOOD": "#2ca02c", "WARNING": "#ffb000", "MAJOR": "#d62728", "UNDEF": "0.55"}


def _fmt(tpl, opts):
    try:
        return tpl.format(**opts)
    except Exception:  # noqa: BLE001
        return tpl


def draw_scheme(out, metric_options=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from .registry import (available_metrics, GROUP_OBS, GROUP_GEN, GROUP_DRP, GROUP_DAP, GROUP_STATUS)

    metric_options = metric_options or {}
    reg = available_metrics()
    inst = {n: c(**metric_options.get(n, {})) for n, c in reg.items()}
    sums = [n for n, m in inst.items() if getattr(m, "summary", False) and n != "qa_obs"]
    sum_cols = ["qa_obs"] + sums
    order = []
    for g in (GROUP_OBS, GROUP_GEN, GROUP_DRP, GROUP_DAP, GROUP_STATUS):
        order += [(g, n) for n, m in inst.items() if m.group == g and not getattr(m, "summary", False)]

    W, H = 16.5, 11.7
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, txt, fc="white", ec="0.3", fs=8, weight="normal", ha="center", lw=1.0, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=lw))
        tx = x + w / 2 if ha == "center" else x + 0.08
        ax.text(tx, y + h / 2, txt, ha=ha, va="center", fontsize=fs, weight=weight, **kw)

    def arrow(x0, y0, x1, y1, color="0.25", ls="-", rad=0.0, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=color,
                                     lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))

    ax.text(W / 2, H - 0.3, "LVM QA (lvmqa) – logic of the QA flags", ha="center", va="center",
            fontsize=15, weight="bold")

    # ---------------------------------------------------------------- flow strip
    y0, h0 = H - 1.75, 1.05
    steps = [
        ("INPUTS", "drpall · SFrame header\nSFrame (FLUX, IVAR, SKY,\nLSF, MASK) · DAP tables\n· DAP output cube", "#e8eef7"),
        ("MEASUREMENTS", "per exposure (multi-exposure\nfor qa_fluxmap)\nvalues → _supl FITS/XLSX", "#e8eef7"),
        ("CLASSIFICATION", "thresholds of each metric\n→ GOOD / WARNING /\nMAJOR / UNDEFINED", "#fdf3e1"),
        ("INDIVIDUAL FLAGS", "obs. components (no slot)\nindividual metrics (slots)\nqa_status (information)", "#fdf3e1"),
        ("SUMMARIES", "qa_obs + 5 science\nsummaries: worst of the\nhard / soft contributions", "#e9f5e9"),
        ("OVERALL", "qa_overall = worst of all\nbitmask flags; qa_bitmask\n(2 bits per slot)", "#e9f5e9"),
        ("OUTPUTS", "QA FITS/XLSX (flags, bitmask,\nDESC) · _supl (values, notes,\ntimings) · HTML · PDF", "#eeeeee"),
    ]
    bw, gap = 2.05, 0.28
    x = 0.35
    xs = []
    for i, (t, s, c) in enumerate(steps):
        box(x, y0, bw, h0, "", fc=c)
        ax.text(x + bw / 2, y0 + h0 - 0.16, t, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x + bw / 2, y0 + h0 / 2 - 0.12, s, ha="center", va="center", fontsize=7)
        xs.append(x)
        if i:
            arrow(x - gap, y0 + h0 / 2, x, y0 + h0 / 2)
        x += bw + gap
    # recalc loop: _supl values -> classification -> ... (lvmqa.recalc), and compare
    yb = y0 - 0.06
    arrow(xs[1] + bw * 0.75, yb, xs[2] + bw * 0.25, yb, color="#7b3fa0", ls="--", rad=0.35)
    ax.text((xs[1] + xs[2]) / 2 + bw / 2, yb - 0.42,
            "python -m lvmqa.recalc: re-classify the stored _supl values with new thresholds → *_recalculated "
            "(no re-measurement)", ha="center", fontsize=7, color="#7b3fa0", style="italic")
    ax.text(xs[6] + bw / 2, yb - 0.42, "lvmqa.compare A.fits B.fits → HTML", ha="center",
            fontsize=7, color="#7b3fa0", style="italic")

    # ---------------------------------------------------------------- table
    top = y0 - 0.85
    bottom = 1.55
    nrow = len(order) + len(sum_cols) + 2
    rh = (top - bottom) / nrow
    cx_in = 0.45
    w_in = 0.40
    x_name = cx_in + len(INPUTS) * w_in + 0.08
    x_meas = x_name + 1.6
    x_warn = x_meas + 3.95
    x_maj = x_warn + 2.45
    x_sum = x_maj + 2.4
    w_s = 0.52
    x_ov = x_sum + len(sum_cols) * w_s
    x_end = x_ov + 0.5

    # header
    yh = top - rh
    for k, (key, lab) in enumerate(INPUTS):
        ax.text(cx_in + (k + 0.5) * w_in, yh + rh * 0.35, lab, ha="center", va="center", fontsize=6.3,
                weight="bold")
    for xx, lab in ((x_name, "metric (slot)"), (x_meas, "measurement"), (x_warn, "WARNING if"),
                    (x_maj, "MAJOR ISSUE if")):
        ax.text(xx + 0.05, yh + rh * 0.35, lab, ha="left", va="center", fontsize=8, weight="bold",
                color={"WARNING if": "#b07800", "MAJOR ISSUE if": FCOL["MAJOR"]}.get(lab, "k"))
    for k, s in enumerate(sum_cols):
        lab = {"qa_abs_calibration": "abs\ncalib", "qa_general_drp": "general\ndrp",
               "qa_line_fluxes": "line\nfluxes"}.get(s, s.replace("qa_", ""))
        ax.text(x_sum + (k + 0.5) * w_s, yh + rh * 0.35, lab, ha="center", va="center", fontsize=6,
                weight="bold")
    ax.text(x_ov + 0.25, yh + rh * 0.35, "overall", ha="center", va="center", fontsize=6.5, weight="bold")
    ax.text(x_sum + len(sum_cols) * w_s / 2, top + 0.12, "contribution to the summary flags",
            ha="center", fontsize=8, weight="bold")
    ax.text(cx_in + len(INPUTS) * w_in / 2, top + 0.12, "inputs", ha="center", fontsize=8, weight="bold")

    gcol = {GROUP_OBS: "#eaf1fb", GROUP_GEN: "#f4f4f4", GROUP_DRP: "#fff6e6", GROUP_DAP: "#eef8ee",
            GROUP_STATUS: "#f6eef8"}
    y = yh
    prev = None
    rows_y = {}
    for g, n in order:
        y -= rh
        m = inst[n]
        if g != prev:
            n_g = sum(1 for gg, _ in order if gg == g)
            ax.text(0.22, y + rh - n_g * rh / 2, g.replace(" (information)", "").replace(" conditions", "\nconditions")
                    .replace(" quality", "\nquality"), ha="center", va="center", fontsize=6.3, style="italic",
                    color="0.3", rotation=90)
            ax.plot([0.35, x_end], [y + rh, y + rh], color="0.6", lw=0.6)
            prev = g
        ax.add_patch(plt.Rectangle((0.35, y), x_end - 0.35, rh, color=gcol[g], lw=0, zorder=0))
        rows_y[n] = y
        req = set(getattr(m, "requires", ()) or ())
        for k, (key, _) in enumerate(INPUTS):
            if key in req:
                ax.plot(cx_in + (k + 0.5) * w_in, y + rh / 2, "o", ms=4.2, color="#3465a4")
        slot = "" if m.slot is None else f" ({m.slot})"
        ax.text(x_name + 0.05, y + rh / 2, n + slot, ha="left", va="center", fontsize=7, weight="bold",
                family="monospace")
        meas, tw, tm = MEAS.get(n, (getattr(m, "title", ""), "", ""))
        opts = dict(m.params)
        meas = _fmt(meas, opts)
        ax.text(x_meas + 0.05, y + rh / 2, meas, ha="left", va="center", fontsize=6.4)
        ax.text(x_warn + 0.05, y + rh / 2, _fmt(tw, opts), ha="left", va="center", fontsize=6.4, color="#8a5d00")
        ax.text(x_maj + 0.05, y + rh / 2, _fmt(tm, opts), ha="left", va="center", fontsize=6.4,
                color="#a01818")
        # memberships
        for k, s in enumerate(sum_cols):
            w = None
            if s == "qa_obs":
                if m.group == GROUP_OBS:
                    w = getattr(m, "obs_weight", "hard")
            else:
                w = inst[s].components.get(n)
            if w:
                xc = x_sum + (k + 0.5) * w_s
                ax.add_patch(plt.Rectangle((xc - w_s * 0.4, y + rh * 0.12), w_s * 0.8, rh * 0.76,
                                           color="#1f4e79" if w == "hard" else "#9dc3e6", lw=0))
                ax.text(xc, y + rh / 2, "H" if w == "hard" else "S", ha="center", va="center", fontsize=6.5,
                        color="white" if w == "hard" else "k", weight="bold")
        if m.in_bitmask:
            ax.plot(x_ov + 0.25, y + rh / 2, "s", ms=5, color="0.2")
        elif n == "qa_status":
            ax.text(x_ov + 0.25, y + rh / 2, "info", ha="center", va="center", fontsize=6, color="0.4")

    # summary rows
    y -= rh * 0.6
    ax.plot([0.35, x_end], [y, y], color="0.3", lw=1.0)
    ax.text(0.22, y - len(sum_cols) * rh / 2, "Summaries", ha="center", va="center", fontsize=6.3,
            style="italic", color="0.3", rotation=90)
    for k, s in enumerate(sum_cols):
        y -= rh
        m = inst[s]
        ax.add_patch(plt.Rectangle((0.35, y), x_end - 0.35, rh, color="#e3f1e3", lw=0, zorder=0))
        slot = "" if m.slot is None else f" ({m.slot})"
        ax.text(x_name + 0.05, y + rh / 2, s + slot, ha="left", va="center", fontsize=7, weight="bold",
                family="monospace")
        if s == "qa_obs":
            txt = "worst contribution of the observational components (hard / soft)"
        else:
            nd = ", ".join(getattr(m, "needs", ())) or "none"
            txt = (f"worst contribution of its components (column); qa_incomplete only for the inputs it "
                   f"needs: {nd}")
        ax.text(x_meas + 0.05, y + rh / 2, getattr(m, "title", "") + " – " + txt, ha="left", va="center",
                fontsize=6.4)
        xc = x_sum + (k + 0.5) * w_s
        ax.plot(xc, y + rh / 2, "*", ms=9, color="#2ca02c")
        ax.plot([xc, xc], [y + rh * 0.8, yh], color="#2ca02c", lw=0.5, ls=":", zorder=0.5)
        if m.in_bitmask:
            ax.plot(x_ov + 0.25, y + rh / 2, "s", ms=5, color="0.2")

    # ---------------------------------------------------------------- legend / rules
    yl = bottom - 0.2
    rules = [
        "Flags (2 bits per slot in qa_bitmask): GOOD = 00, WARNING = 01, MAJOR ISSUE = 10, UNDEFINED = 11 "
        "(not measurable: missing input, too few fibers / points).",
        "H = hard component: WARNING → WARNING, MAJOR → MAJOR.   S = soft component: WARNING → GOOD, MAJOR → WARNING.   "
        "Summary = worst contribution; UNDEFINED components are ignored (all UNDEFINED → UNDEFINED).",
        "Observational components (qa_illumination, qa_brightstars, qa_guider) have no slot; only qa_obs is stored "
        "in the bitmask.  qa_status is information only (no flag).",
        "qa_overall = worst over all metrics with a slot (■), including the summaries (MAJOR > WARNING > "
        "UNDEFINED > GOOD).  Thresholds shown are the current values of the metric options.",
    ]
    for i, r in enumerate(rules):
        ax.text(0.35, yl - i * 0.27, r, ha="left", va="center", fontsize=7.2)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "png"):
        f = out / f"qa_logic_scheme.{ext}"
        fig.savefig(f, dpi=170)
        files.append(f)
    plt.close(fig)
    return files


def main(argv=None):
    import ast
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/Users/mors/Science/LVM/QA/DR21_dev/Claude")
    ap.add_argument("--set", action="append", default=[], metavar="METRIC.PARAM=VALUE")
    a = ap.parse_args(argv)
    opts = {}
    for it in a.set:
        key, _, val = it.partition("=")
        mname, _, par = key.partition(".")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        opts.setdefault(mname, {})[par] = val
    for f in draw_scheme(a.out, opts):
        print(f)


if __name__ == "__main__":
    main()
