"""
qa_dap_pmfit - quality of the parametric emission-line fits of bright lines.

DAP stores, per fiber and line, the reduced chi^2 in a window around the line of
(gas spectrum - model)^2 / error^2 (extension ELINES_SIGMA_CHI):
    X_sq_pek_<wl>       parametric model with fixed kinematics (PM_KEL)
    X_sq_<name>_<wl>    non-parametric model
Only (fiber, line) points with non-parametric S/N > ``min_snr`` are used.

Two statistics are computed (both stored):

* absolute (DR20 definition): residual / error = sqrt(X_sq_pek); fraction of points above
  3 and 5 sigma.
* relative (default for the flag): X_sq_pek / max(X_sq_np, 1), i.e. the parametric chi^2
  in units of the chi^2 reached by the non-parametric model of the same spectrum. For both
  models chi^2 grows ~ (S/N)^2 for bright lines (line-profile / error systematics shared by
  both), so the absolute criterion flags essentially every bright line; the ratio isolates
  failures of the parametric fit itself.

The flag uses ``mode`` ('relative' or 'absolute').
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from .dap_npfit import BRIGHT_LINES, line_key


@register
class DapPmFit(Metric):
    name = "qa_dap_pmfit"
    slot = 10
    group = GROUP_DAP
    title = "DAP parametric fit residuals"
    description = ("DAP fit quality: residuals of the parametric fit of bright emission lines "
                   "(reduced chi^2 in the line window, relative to the non-parametric model)")
    requires = ("dap",)
    defaults = dict(
        lines=BRIGHT_LINES,
        min_snr=10.0,          # non-parametric S/N to select detected lines
        mode="relative",       # 'relative' (chi2_pm / chi2_np) or 'absolute' (DR20, sqrt(chi2_pm))
        rel_warn=2.0,          # relative: chi2_pm / max(chi2_np, 1) above this -> point 'warn'
        rel_major=4.0,         # relative: ... above this -> point 'major'
        nsig_warn=3.0,         # absolute: residual > 3 sigma -> point 'warn'
        nsig_major=5.0,        # absolute: residual > 5 sigma -> point 'major'
        frac_thresh=0.05,      # exposure flag if fraction of 'warn'/'major' points exceeds this
        min_points=20,
    )
    values = (
        ValueSpec("n_points", "i8", "", "(fiber, line) points with NP S/N > min_snr"),
        ValueSpec("median_chi2_pm", "f8", "", "Median reduced chi^2 of the parametric fit"),
        ValueSpec("median_chi2_np", "f8", "", "Median reduced chi^2 of the non-parametric model"),
        ValueSpec("median_chi2_ratio", "f8", "", "Median chi2_pm / max(chi2_np, 1)"),
        ValueSpec("frac_rel_warn", "f8", "", "Fraction of points with chi2 ratio > rel_warn"),
        ValueSpec("frac_rel_major", "f8", "", "Fraction of points with chi2 ratio > rel_major"),
        ValueSpec("frac_abs_warn", "f8", "", "Fraction of points with sqrt(chi2_pm) > nsig_warn (DR20)"),
        ValueSpec("frac_abs_major", "f8", "", "Fraction of points with sqrt(chi2_pm) > nsig_major (DR20)"),
        ValueSpec("worst_line", "U16", "", "Line with the largest fraction of 'warn' points"),
        ValueSpec("worst_line_frac", "f8", "", "That fraction"),
    ) + tuple(ValueSpec(f"ratio_{line_key(l)}", "f8", "", f"{l}: median chi2_pm / max(chi2_np, 1)")
              for l in BRIGHT_LINES)
    main_value = "median_chi2_ratio"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        if p["mode"] == "absolute":
            w = f"residual > {p['nsig_warn']:g} sigma"
            m = f"residual > {p['nsig_major']:g} sigma"
        else:
            w = f"chi2_pm / max(chi2_np, 1) > {p['rel_warn']:g}"
            m = f"chi2_pm / max(chi2_np, 1) > {p['rel_major']:g}"
        f = f"{p['frac_thresh']:.0%}"
        return {
            "GOOD": f"<= {f} of detected bright-line fits with {w}",
            "WARNING": f"> {f} of detected bright-line fits with {w}, but <= {f} with {m}",
            "MAJOR ISSUE": f"> {f} of detected bright-line fits with {m}",
            "UNDEFINED": f"No DAP file / chi^2 table, or < {p['min_points']} bright-line points with NP S/N > {p['min_snr']:g}",
        }

    def compute(self, ctx):
        p = self.params
        chi = ctx.dap_table("ELINES_SIGMA_CHI")
        cols = []
        for line in p["lines"]:
            cols += [f"flux_{line}", f"e_flux_{line}"]
        npt = ctx.dap_np_columns(cols)
        index = {k: i for i, k in enumerate(np.asarray(chi["id"]).astype(str))}
        rows = np.array([index.get(k, -1) for k in npt["id"]], dtype=int)
        has = rows >= 0
        declared = self.empty_values()

        def aligned(col):
            out = np.full(len(npt), np.nan)
            if col in chi.colnames:
                out[has] = np.asarray(chi[col], dtype=float)[rows[has]]
            return out

        vals, pm_all, np_all, per_line = {}, [], [], {}
        for line in p["lines"]:
            x_pm = aligned(f"X_sq_pek_{line.rsplit('_', 1)[1]}")
            x_np = aligned(f"X_sq_{line}")
            f = np.asarray(npt[f"flux_{line}"], dtype=float)
            e = np.asarray(npt[f"e_flux_{line}"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                use = (np.isfinite(f) & np.isfinite(e) & (e > 0) & (f / e > p["min_snr"])
                       & np.isfinite(x_pm) & np.isfinite(x_np))
            x_pm, x_np = x_pm[use], x_np[use]
            pm_all.append(x_pm)
            np_all.append(x_np)
            ratio = x_pm / np.maximum(x_np, 1.0)
            k = f"ratio_{line_key(line)}"
            if k in declared and ratio.size:
                vals[k] = float(np.median(ratio))
            if ratio.size >= 10:
                warn = ratio > p["rel_warn"] if p["mode"] != "absolute" else np.sqrt(np.clip(x_pm, 0, None)) > p["nsig_warn"]
                per_line[line] = float(np.mean(warn))

        x_pm = np.concatenate(pm_all) if pm_all else np.array([])
        x_np = np.concatenate(np_all) if np_all else np.array([])
        vals["n_points"] = int(x_pm.size)
        if per_line:
            w = max(per_line, key=per_line.get)
            vals.update(worst_line=w, worst_line_frac=per_line[w])
        if x_pm.size < p["min_points"]:
            return self.result(QAFlag.UNDEFINED, note=f"only {x_pm.size} detected bright-line points", **vals)

        ratio = x_pm / np.maximum(x_np, 1.0)
        resid = np.sqrt(np.clip(x_pm, 0, None))
        vals.update(median_chi2_pm=float(np.median(x_pm)), median_chi2_np=float(np.median(x_np)),
                    median_chi2_ratio=float(np.median(ratio)),
                    frac_rel_warn=float(np.mean(ratio > p["rel_warn"])),
                    frac_rel_major=float(np.mean(ratio > p["rel_major"])),
                    frac_abs_warn=float(np.mean(resid > p["nsig_warn"])),
                    frac_abs_major=float(np.mean(resid > p["nsig_major"])))
        key = "abs" if p["mode"] == "absolute" else "rel"
        fw, fm = vals[f"frac_{key}_warn"], vals[f"frac_{key}_major"]
        if fm > p["frac_thresh"]:
            return self.result(QAFlag.MAJOR, note=f"{fm:.0%} fits above major threshold ({p['mode']})", **vals)
        if fw > p["frac_thresh"]:
            return self.result(QAFlag.WARNING, note=f"{fw:.0%} fits above warn threshold ({p['mode']})", **vals)
        return self.result(QAFlag.GOOD, **vals)
