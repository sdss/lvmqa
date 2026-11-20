"""
qa_dap_pmfit - quality of the parametric emission-line fits of bright lines from the actual
residuals in the DAP output (model RSS) file.

For every (fiber, line) with non-parametric S/N > ``min_snr`` the residual of the parametric
joint model

    r = org_spec - mod_joint_spec_PEK      (= data - stellar - parametric lines - smooth term)

(v1.15: minus a straight line through its medians in the line-free pixels of the +-30 A window,
i.e. the residual continuum of an imperfect DAP stellar / smooth model is removed: otherwise it
inflates fx in fibers with a bright continuum)
is evaluated in a window centred on the line at its observed (NP) velocity,
+- max(half_width, 3 x NP dispersion) A ([OII]3726,3729 as one window). The statistic is
the unmodelled flux fraction (S/N independent; noise expectation removed):

    fx = (sum|r| - sqrt(2/pi) sum sigma) / sum(parametric line model),
    sigma from the SFrame IVAR; e_fx = sqrt(1-2/pi) sqrt(sum sigma^2) / sum(model)

A point is 'poor' if fx > fx_warn (10%) and 'bad' if fx > fx_major (20%), in both cases
only if fx > nsig x e_fx (significant).
GOOD:    <= 10% of the points poor
WARNING: > frac_thresh (10%) of the points poor
MAJOR:   > frac_thresh of the points bad

Why not chi^2: for bright lines chi^2 ~ (S/N)^2 x (profile mismatch)^2 grows without limit
(e.g. median chi^2 = 16 for Halpha at S/N~320 in expnum 7915 although only 3% of the flux is
unmodelled), whereas fx measures the mismatch in units of the line flux. The median reduced
chi^2 of the parametric and NP models are still stored. Replaces the chi^2-ratio version
based on ELINES_SIGMA_CHI.
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from ._dapout import C_KMS, GAS_NP, GAS_PM, JOINT_PM, ORG, RES_NP, excess_residual, local_baseline, \
    np_column, shifted_mask, sigma_on

# key: (NP line names summed into one window)
BRIGHT = {
    "OII3727": ("[OII]_3726.03", "[OII]_3728.82"),
    "Hbeta": ("Hbeta_4861.36",),
    "OIII5007": ("[OIII]_5006.84",),
    "Halpha": ("Halpha_6562.85",),
    "NII6583": ("[NII]_6583.45",),
    "SII6716": ("[SII]_6716.44",),
    "SII6731": ("[SII]_6730.82",),
}
MARGIN = 30.0   # A read around the rest wavelengths (velocities up to ~ +-1300 km/s)


def _wl(n):
    return float(n.rsplit("_", 1)[1])


def _values():
    v = [ValueSpec("n_points", "i8", "", "(fiber, line) points with NP S/N > min_snr"),
         ValueSpec("frac_poor", "f8", "", "Fraction of points with significant fx > fx_warn"),
         ValueSpec("frac_bad", "f8", "", "Fraction of points with significant fx > fx_major"),
         ValueSpec("median_fx", "f8", "", "Median unmodelled flux fraction (parametric)"),
         ValueSpec("median_fx_np", "f8", "", "Median unmodelled flux fraction of the NP model (reference)"),
         ValueSpec("median_chi2_pm", "f8", "", "Median reduced chi^2 of the parametric joint model in the windows"),
         ValueSpec("median_chi2_np", "f8", "", "Median reduced chi^2 of the NP joint model in the windows"),
         ValueSpec("worst_line", "U16", "", "Line with the largest fraction of poor points"),
         ValueSpec("worst_line_frac", "f8", "", "Fraction of poor points of the worst line")]
    for k in BRIGHT:
        v += [ValueSpec(f"n_{k}", "i8", "", f"{k}: points used"),
              ValueSpec(f"fx_{k}", "f8", "", f"{k}: median unmodelled flux fraction (parametric)"),
              ValueSpec(f"fpoor_{k}", "f8", "", f"{k}: fraction of poor points")]
    return tuple(v)


@register
class DapPmFit(Metric):
    name = "qa_dap_pmfit"
    slot = 10
    group = GROUP_DAP
    title = "DAP parametric fit residuals"
    description = ("DAP fit quality: unmodelled flux in the residuals of the parametric fit "
                   "(data - stellar - parametric lines) around bright emission lines, in units "
                   "of the modelled line flux (DAP output file)")
    requires = ("dap", "dap_output", "sframe_data")
    defaults = dict(
        lines=tuple(BRIGHT),
        min_snr=10.0,        # NP S/N (summed for [OII]) required per fiber and line
        half_width=3.5,      # [A] minimum half width of the window
        fx_warn=0.10,        # point 'poor' above this unmodelled flux fraction
        fx_major=0.20,       # point 'bad' above this
        nsig=3.0,            # ... and only if fx > nsig * e_fx
        frac_thresh=0.10,    # fraction of poor / bad points above which the exposure is flagged
        min_points=20,
        subtract_baseline=True,   # remove the residual continuum (line-free pixels of the window)
    )
    values = _values()
    main_value = "frac_poor"

    def output_windows(self):
        out = {}
        for k in self.params["lines"]:
            wl = [_wl(n) for n in BRIGHT[k]]
            out[f"pm_{k}"] = (min(wl) - MARGIN, max(wl) + MARGIN, (ORG, RES_NP, GAS_NP, GAS_PM, JOINT_PM))
        return out

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        f = f"{p['frac_thresh']:.0%}"
        fx = "fx = unmodelled flux / parametric line flux (noise removed)"
        return {
            "GOOD": f"<= {f} of bright-line points with {fx} > {p['fx_warn']:.0%}",
            "WARNING": f"> {f} of bright-line points with {fx} > {p['fx_warn']:.0%} ({p['nsig']:g} sigma)",
            "MAJOR ISSUE": f"> {f} of bright-line points with fx > {p['fx_major']:.0%} ({p['nsig']:g} sigma)",
            "UNDEFINED": (f"No DAP / DAP output file, or < {p['min_points']} points with NP "
                          f"S/N > {p['min_snr']:g}"),
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")
        if not ctx.has_dap_output:
            return self.result(QAFlag.UNDEFINED, note="DAP output file missing")
        vals = {}
        poor_all, bad_all, fx_all, fxnp_all, c2pm, c2np = [], [], [], [], [], []
        per_line = {}
        for k in p["lines"]:
            names = BRIGHT[k]
            F = sum(np.nan_to_num(np_column(ctx, f"flux_{n}")) for n in names)
            EF = np.sqrt(sum(np.nan_to_num(np_column(ctx, f"e_flux_{n}")) ** 2 for n in names))
            vel = np_column(ctx, f"vel_{names[-1]}")
            disp = np_column(ctx, f"disp_{names[-1]}")
            with np.errstate(invalid="ignore", divide="ignore"):
                sel = np.isfinite(F) & (EF > 0) & (F / EF > p["min_snr"]) & np.isfinite(vel)
            vals[f"n_{k}"] = int(sel.sum())
            if not sel.any():
                continue
            wave, planes = ctx.dap_output(f"pm_{k}")
            sig = sigma_on(ctx, wave)
            wl = [_wl(n) for n in names]
            z = vel[sel] / C_KMS
            lo, hi = min(wl) * (1 + z), max(wl) * (1 + z)
            half = np.clip(np.nan_to_num(3.0 * disp[sel], nan=p["half_width"]), p["half_width"], 10.0)
            use = (wave[None, :] >= (lo - half)[:, None]) & (wave[None, :] <= (hi + half)[:, None])
            org = planes[ORG][sel].astype(float)
            r_pm = org - planes[JOINT_PM][sel]
            gpm, gnp = planes[GAS_PM][sel].astype(float), planes[GAS_NP][sel].astype(float)
            r_np = planes[RES_NP][sel].astype(float)
            if p["subtract_baseline"]:
                # v1.15: residual continuum (imperfect DAP stellar / smooth model) removed with a
                # straight line through the line-free pixels of the window
                r_pm, _ = local_baseline(r_pm, wave, (gpm, gnp), exclude=use)
                r_np, _ = local_baseline(r_np, wave, (gpm, gnp), exclude=use)
            fx, efx, chi_pm, _ = excess_residual(r_pm, sig[sel], gpm, use)
            fxn, _, chi_np, _ = excess_residual(r_np, sig[sel], gnp, use)
            ok = np.isfinite(fx)
            signif = fx > p["nsig"] * efx
            poor = ok & signif & (fx > p["fx_warn"])
            bad = ok & signif & (fx > p["fx_major"])
            # points without a usable parametric model (missing / zero) count as bad
            nomodel = ~ok
            poor, bad = poor | nomodel, bad | nomodel
            poor_all.append(poor), bad_all.append(bad)
            fx_all.append(fx[ok]), fxnp_all.append(fxn[np.isfinite(fxn)])
            c2pm.append(chi_pm[np.isfinite(chi_pm)]), c2np.append(chi_np[np.isfinite(chi_np)])
            vals[f"fpoor_{k}"] = float(poor.mean())
            if ok.any():
                vals[f"fx_{k}"] = float(np.median(fx[ok]))
            if sel.sum() >= 10:
                per_line[k] = float(poor.mean())
        poor = np.concatenate(poor_all) if poor_all else np.array([], bool)
        bad = np.concatenate(bad_all) if bad_all else np.array([], bool)
        vals["n_points"] = int(poor.size)
        for name, arr in (("median_fx", fx_all), ("median_fx_np", fxnp_all),
                          ("median_chi2_pm", c2pm), ("median_chi2_np", c2np)):
            a = np.concatenate(arr) if arr else np.array([])
            if a.size:
                vals[name] = float(np.median(a))
        if per_line:
            w = max(per_line, key=per_line.get)
            vals.update(worst_line=w, worst_line_frac=per_line[w])
        if poor.size < p["min_points"]:
            return self.result(QAFlag.UNDEFINED, note=f"only {poor.size} points with NP S/N>{p['min_snr']:g}",
                               **vals)
        fp, fb = float(poor.mean()), float(bad.mean())
        vals.update(frac_poor=fp, frac_bad=fb)
        if fb > p["frac_thresh"]:
            return self.result(QAFlag.MAJOR, note=f"{fb:.0%} points with >{p['fx_major']:.0%} unmodelled flux",
                               **vals)
        if fp > p["frac_thresh"]:
            return self.result(QAFlag.WARNING, note=f"{fp:.0%} points with >{p['fx_warn']:.0%} unmodelled flux",
                               **vals)
        return self.result(QAFlag.GOOD, **vals)
