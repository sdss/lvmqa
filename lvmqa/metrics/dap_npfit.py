"""
qa_dap_npfit - consistency of parametric and non-parametric DAP fluxes of bright lines.

Points = (fiber, line) pairs for the bright lines with non-parametric S/N > ``min_snr``.
For each point the deviation is measured in units of the NON-PARAMETRIC error only:

    z = (F_pm - F_np) / e_np        (F_pm from PM_KEL, fixed kinematics, as in DR20)

A point disagrees if |z| > ``nsigma`` or if the parametric fit diverged / is missing
(|F_pm| or e_pm > ``max_abs_flux``, or not finite). The flag is based on the fraction of
disagreeing points.
"""

from __future__ import annotations

import re

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register

BRIGHT_LINES = (
    "[OII]_3726.03", "[OII]_3728.82", "Hbeta_4861.36", "[OIII]_5006.84",
    "Halpha_6562.85", "[NII]_6583.45", "[SII]_6716.44", "[SII]_6730.82",
)


def line_key(line: str) -> str:
    """'[OIII]_5006.84' -> 'OIII5007' (column-name friendly)."""
    name, wl = line.rsplit("_", 1)
    return re.sub(r"[^A-Za-z0-9]", "", name) + f"{float(wl):.0f}"


def _line_values():
    v = []
    for l in BRIGHT_LINES:
        k = line_key(l)
        v += [ValueSpec(f"n_{k}", "i8", "", f"{l}: points with NP S/N > min_snr"),
              ValueSpec(f"fbad_{k}", "f8", "", f"{l}: fraction of disagreeing points"),
              ValueSpec(f"fnp_{k}", "f8", "1e-16 erg / (s cm2)", f"{l}: median non-parametric flux of selected fibers"),
              ValueSpec(f"fpm_{k}", "f8", "1e-16 erg / (s cm2)", f"{l}: median parametric flux of selected fibers (diverged excluded)")]
    return tuple(v)


@register
class DapNpFit(Metric):
    name = "qa_dap_npfit"
    slot = 9
    group = GROUP_DAP
    title = "DAP parametric vs non-parametric fluxes"
    description = ("DAP fit quality: agreement of parametric (PM_KEL) and non-parametric fluxes "
                   "of bright emission lines, in units of the non-parametric errors; diverged "
                   "parametric fits count as disagreement")
    requires = ("dap",)
    defaults = dict(
        lines=BRIGHT_LINES,
        pm_extension="PM_KEL",
        min_snr=10.0,          # non-parametric S/N threshold per fiber and line
        nsigma=3.0,            # |F_pm - F_np| / e_np above this -> disagreement
        max_abs_flux=1e10,     # [1e-16 erg/s/cm2] |F_pm| or e_pm above -> diverged fit (disagreement)
        min_points=20,         # minimum number of (fiber, line) points for a verdict
        warn_frac_bad=0.10,    # fraction of disagreeing points -> WARNING
        major_frac_bad=0.25,   # fraction of disagreeing points -> MAJOR
    )
    values = (
        ValueSpec("n_points", "i8", "", "(fiber, line) points with NP S/N > min_snr"),
        ValueSpec("n_bad", "i8", "", "Disagreeing points (|z| > nsigma or diverged parametric fit)"),
        ValueSpec("frac_bad", "f8", "", "Fraction of disagreeing points"),
        ValueSpec("n_pm_diverged", "i8", "", "Selected points with diverged / missing parametric fit"),
        ValueSpec("median_z", "f8", "", "Median of (F_pm - F_np)/e_np (diverged excluded)"),
        ValueSpec("median_pdiff", "f8", "percent", "Median of 100*(F_pm - F_np)/F_pm (diverged excluded)"),
        ValueSpec("worst_line", "U16", "", "Line with the largest fraction of disagreeing points"),
        ValueSpec("worst_line_fbad", "f8", "", "Fraction of disagreeing points for the worst line"),
    ) + _line_values()
    main_value = "frac_bad"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        cond = (f"|F_pm - F_np| > {p['nsigma']:g} e_np or diverged parametric fit "
                f"(|flux| or error > {p['max_abs_flux']:g})")
        return {
            "GOOD": f"<= {p['warn_frac_bad']:.0%} of bright-line points with {cond}",
            "WARNING": f"> {p['warn_frac_bad']:.0%} of bright-line points with {cond}",
            "MAJOR ISSUE": f"> {p['major_frac_bad']:.0%} of bright-line points with {cond}",
            "UNDEFINED": (f"No DAP file, or < {p['min_points']} (fiber, line) points with "
                          f"non-parametric S/N > {p['min_snr']:g}"),
        }

    def compute(self, ctx):
        p = self.params
        pm = ctx.dap_table(p["pm_extension"])
        pm = pm[np.char.strip(np.asarray(pm["model"]).astype(str)) == "eline"]
        pm_id = np.asarray(pm["id"]).astype(str)
        pm_wl = np.asarray(pm["wl"], dtype=float)
        pm_f_all = np.asarray(pm["flux"], dtype=float)
        pm_e_all = np.asarray(pm["e_flux"], dtype=float)

        cols = []
        for line in p["lines"]:
            cols += [f"flux_{line}", f"e_flux_{line}"]
        npt = ctx.dap_np_columns(cols)
        n_fib = len(npt)
        index = {k: i for i, k in enumerate(npt["id"])}
        declared = self.empty_values()

        vals = {}
        z_all, pd_all, bad_all, div_all = [], [], [], []
        per_line_fbad = {}
        for line in p["lines"]:
            # parametric values aligned on the NP / PT rows (NaN where missing)
            f_pm = np.full(n_fib, np.nan)
            e_pm = np.full(n_fib, np.nan)
            sel = np.abs(pm_wl - float(line.rsplit("_", 1)[1])) < 0.01
            rows = np.array([index.get(k, -1) for k in pm_id[sel]], dtype=int)
            ok = rows >= 0
            f_pm[rows[ok]] = pm_f_all[sel][ok]
            e_pm[rows[ok]] = pm_e_all[sel][ok]
            f_np = np.asarray(npt[f"flux_{line}"], dtype=float)
            e_np = np.asarray(npt[f"e_flux_{line}"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                use = np.isfinite(f_np) & np.isfinite(e_np) & (e_np > 0) & (f_np / e_np > p["min_snr"])
            f_pm, e_pm, f_np, e_np = f_pm[use], e_pm[use], f_np[use], e_np[use]
            diverged = ~(np.abs(f_pm) < p["max_abs_flux"]) | ~(np.abs(e_pm) < p["max_abs_flux"])
            z = (f_pm - f_np) / e_np
            bad = diverged | (np.abs(z) > p["nsigma"])
            z_all.append(z[~diverged])
            with np.errstate(invalid="ignore", divide="ignore"):
                pd_all.append((100.0 * (f_pm - f_np) / f_pm)[~diverged & (f_pm != 0)])
            bad_all.append(bad)
            div_all.append(diverged)
            k = line_key(line)
            if f"n_{k}" in declared:
                vals[f"n_{k}"] = int(use.sum())
                if use.sum():
                    vals[f"fbad_{k}"] = float(bad.mean())
                    vals[f"fnp_{k}"] = float(np.median(f_np))
                    if (~diverged).any():
                        vals[f"fpm_{k}"] = float(np.median(f_pm[~diverged]))
            if use.sum() >= 10:
                per_line_fbad[line] = float(bad.mean())

        bad = np.concatenate(bad_all) if bad_all else np.array([], dtype=bool)
        div = np.concatenate(div_all) if div_all else np.array([], dtype=bool)
        z = np.concatenate(z_all) if z_all else np.array([])
        pdiff = np.concatenate(pd_all) if pd_all else np.array([])
        vals.update(n_points=int(bad.size), n_bad=int(bad.sum()), n_pm_diverged=int(div.sum()))
        if z.size:
            vals["median_z"] = float(np.median(z))
        if pdiff.size:
            vals["median_pdiff"] = float(np.median(pdiff))
        if per_line_fbad:
            worst = max(per_line_fbad, key=per_line_fbad.get)
            vals.update(worst_line=worst, worst_line_fbad=per_line_fbad[worst])
        if bad.size < p["min_points"]:
            return self.result(QAFlag.UNDEFINED, note=f"only {bad.size} points with NP S/N>{p['min_snr']:g}",
                               **vals)
        fb = float(bad.mean())
        vals["frac_bad"] = fb
        note = f"{fb:.0%} points disagree" + (f" ({int(div.sum())} diverged pm fits)" if div.sum() else "")
        if fb > p["major_frac_bad"]:
            return self.result(QAFlag.MAJOR, note=note, **vals)
        if fb > p["warn_frac_bad"]:
            return self.result(QAFlag.WARNING, note=note, **vals)
        return self.result(QAFlag.GOOD, **vals)
