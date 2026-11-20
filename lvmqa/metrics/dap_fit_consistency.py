"""
qa_fit_consistency (formerly qa_dap_npfit) - consistency of parametric and non-parametric DAP fluxes of bright lines.

Points = (fiber, line) pairs for the bright lines with non-parametric S/N > ``min_snr``.
For each point the deviation is measured in units of the non-parametric error, with a
systematic floor added in quadrature:

    z = (F_pm - F_np) / sqrt(e_np^2 + (sys_floor * F_np)^2)     (F_pm from PM_KEL, as in DR20)

The floor (default 2%) absorbs the few-percent model-dependent differences that are expected
between a Gaussian fit and a direct integration (line wings, [OII] doublet deblending, continuum
placement). Without it, at S/N >~ 200 a 1.5% Halpha difference is already > 3 sigma and
bright fields are flagged although both fits are acceptable (DR20 comparison, expnum 7915).

A point disagrees if |z| > ``nsigma`` or if the parametric fit diverged / is missing
(|F_pm| or e_pm > ``max_abs_flux``, or not finite). The flag is based on the fraction of
disagreeing points.

Masked line cores (needs the SFrame): both fits ignore masked pixels, so a fiber whose line core
(|lambda - lambda_obs| < core_half = 1.5 A) contains masked pixels has PM and NP fluxes that agree
with each other but miss part of the flux (DRP 1.2.0, 7959 fiber 1875: three masked pixels at the
Halpha peak, both fluxes 56% of the data integral; 7331 fibers 1554-1577: a masked CCD column at
Halpha). ~0.4-0.9% of the bright-line points in every DRP 1.2.0 exposure. frac_masked_core > 2% ->
WARNING, > 10% -> MAJOR, if >= 5 such points (flag = worse of the two tests).
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
    name = "qa_fit_consistency"
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
        nsigma=3.0,            # |z| above this -> disagreement
        sys_floor=0.02,        # relative systematic floor added in quadrature to e_np
        max_abs_flux=1e10,     # [1e-16 erg/s/cm2] |F_pm| or e_pm above -> diverged fit (disagreement)
        min_points=20,         # minimum number of (fiber, line) points for a verdict
        warn_frac_bad=0.10,    # fraction of disagreeing points -> WARNING
        major_frac_bad=0.25,   # fraction of disagreeing points -> MAJOR
        core_half=1.5,         # [A] half width of the line core checked for masked pixels
        warn_frac_masked=0.02,   # fraction of points with masked line core -> WARNING
        major_frac_masked=0.10,  # ... -> MAJOR
        min_masked=5,          # fewer masked-core points never raise a flag
    )
    values = (
        ValueSpec("n_points", "i8", "", "(fiber, line) points with NP S/N > min_snr"),
        ValueSpec("n_bad", "i8", "", "Disagreeing points (|z| > nsigma or diverged parametric fit)"),
        ValueSpec("frac_bad", "f8", "", "Fraction of disagreeing points"),
        ValueSpec("n_pm_diverged", "i8", "", "Selected points with diverged / missing parametric fit"),
        ValueSpec("median_z", "f8", "", "Median of z = (F_pm - F_np)/sqrt(e_np^2 + (sys_floor F_np)^2) (diverged excluded)"),
        ValueSpec("median_pdiff", "f8", "percent", "Median of 100*(F_pm - F_np)/F_pm (diverged excluded)"),
        ValueSpec("worst_line", "U16", "", "Line with the largest fraction of disagreeing points"),
        ValueSpec("worst_line_fbad", "f8", "", "Fraction of disagreeing points for the worst line"),
        ValueSpec("n_masked_core", "i8", "", "Points with masked SFrame pixels in the line core (flux underestimated)"),
        ValueSpec("frac_masked_core", "f8", "", "Fraction of points with masked line core"),
    ) + _line_values()
    main_value = "frac_bad"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        cond = (f"|F_pm - F_np| > {p['nsigma']:g} sqrt(e_np^2 + ({p['sys_floor']:g} F_np)^2) "
                f"or diverged parametric fit "
                f"(|flux| or error > {p['max_abs_flux']:g})")
        return {
            "GOOD": (f"<= {p['warn_frac_bad']:.0%} of bright-line points with {cond}, and <= "
                     f"{p['warn_frac_masked']:.0%} with masked pixels in the line core"),
            "WARNING": (f"> {p['warn_frac_bad']:.0%} of bright-line points with {cond}, or > "
                        f"{p['warn_frac_masked']:.0%} with masked line core"),
            "MAJOR ISSUE": (f"> {p['major_frac_bad']:.0%} of bright-line points with {cond}, or > "
                            f"{p['major_frac_masked']:.0%} with masked line core"),
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
            cols += [f"flux_{line}", f"e_flux_{line}", f"vel_{line}"]
        n_mask = 0
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
            if ctx.has_sframe and use.any():
                wl = float(line.rsplit("_", 1)[1])
                wm, mk = ctx.dap_sframe_window("MASK", wl - 12.0, wl + 12.0)
                _, fx = ctx.dap_sframe_window("FLUX", wl - 12.0, wl + 12.0)
                v = np.nan_to_num(np.asarray(npt[f"vel_{line}"], dtype=float))
                lc = wl * (1.0 + v / 299792.458)
                core = np.abs(wm[None, :] - lc[:, None]) < p["core_half"]
                masked = (core & ((np.nan_to_num(mk) > 0) | ~np.isfinite(fx))).any(axis=1)
                n_mask += int((masked & use).sum())
            f_pm, e_pm, f_np, e_np = f_pm[use], e_pm[use], f_np[use], e_np[use]
            diverged = ~(np.abs(f_pm) < p["max_abs_flux"]) | ~(np.abs(e_pm) < p["max_abs_flux"])
            z = (f_pm - f_np) / np.hypot(e_np, p["sys_floor"] * np.abs(f_np))
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
        flag = QAFlag.GOOD
        if fb > p["major_frac_bad"]:
            flag = QAFlag.MAJOR
        elif fb > p["warn_frac_bad"]:
            flag = QAFlag.WARNING
        if ctx.has_sframe:
            fm = n_mask / bad.size
            vals.update(n_masked_core=n_mask, frac_masked_core=fm)
            note += f"; {fm:.1%} masked line cores"
            if n_mask < p["min_masked"]:
                pass
            elif fm > p["major_frac_masked"]:
                flag = QAFlag.MAJOR
            elif fm > p["warn_frac_masked"] and flag == QAFlag.GOOD:
                flag = QAFlag.WARNING
        return self.result(flag, note=note, **vals)
