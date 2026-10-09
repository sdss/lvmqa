"""
qa_skyline_sub - sky emission-line subtraction quality, per spectrograph.

DR20 definition (scripts/drp_qa_sky_aj.ipynb): for each sky line / band, median
spectra of FLUX (sky-subtracted) and SKY (sky model) are built from the good science
fibers; a local continuum (median of the side bands) is subtracted from both and

    r = median_pix( (FLUX_line - cont_FLUX) / (SKY_line - cont_SKY) )

is the residual relative to the sky-line flux. Here this is done separately for each
spectrograph (1, 2, 3); the exposure flag is the worst spectrograph.
[OI]6300 is not used (contaminated by nebular emission).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import FLAG_LABELS, QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

# name: (line window, continuum window) [Angstrom], DR20 values
SKY_LINES = {
    "OI5577": ((5575.0, 5580.0), (5571.0, 5583.0)),
    "OH6865": ((6861.0, 6866.0), (6853.0, 6868.0)),
    "NaD": ((5887.0, 5898.0), (5880.0, 5901.0)),
    "O2": ((8610.0, 8710.0), (8600.0, 8724.0)),
}
READ_WINDOWS = ((5565.0, 5910.0), (6845.0, 6875.0), (8590.0, 8730.0))
SPECS = (1, 2, 3)
_SEV = {QAFlag.GOOD: 0, QAFlag.UNDEFINED: 1, QAFlag.WARNING: 2, QAFlag.MAJOR: 3}


def line_ratio(wave, flux, sky, line, cont):
    """DR20 line_stats -> (resid_med, sky_med, resid/sky)."""
    wl = (wave >= line[0]) & (wave <= line[1])
    wc = ((wave >= cont[0]) & (wave < line[0])) | ((wave > line[1]) & (wave <= cont[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nf = flux[wl] - np.nanmedian(flux[wc])
        ns = sky[wl] - np.nanmedian(sky[wc])
        return float(np.nanmedian(nf)), float(np.nanmedian(ns)), float(np.nanmedian(nf / ns))


def _values():
    v = []
    for name in SKY_LINES:
        for s in SPECS:
            v.append(ValueSpec(f"{name}_sp{s}", "f8", "", f"{name}: residual / sky-line flux, spectrograph {s}"))
        v.append(ValueSpec(f"{name}_all", "f8", "", f"{name}: residual / sky-line flux, all spectrographs"))
    for s in SPECS:
        v.append(ValueSpec(f"flag_sp{s}", "U11", "", f"Flag of spectrograph {s}"))
    v += [ValueSpec("worst_spec", "i8", "", "Spectrograph with the worst flag / largest residual"),
          ValueSpec("max_abs_resid", "f8", "", "Largest |residual/sky| over spectrographs and lines except 5577")]
    return tuple(v)


@register
class SkyLineSub(Metric):
    name = "qa_skyline_sub"
    slot = 6
    group = GROUP_DRP
    title = "Sky emission-line subtraction"
    description = ("Sky emission line subtraction quality per spectrograph: residual / sky-line "
                   "flux for [OI]5577, OH6865, NaD, O2 band (median of good science fibers)")
    requires = ("sframe_data",)
    defaults = dict(
        bad_all_lines=0.10,      # |r| > this in ALL lines -> MAJOR
        bad_any=0.30,            # |r| > this in any line except 5577 -> MAJOR
        warn_any=0.10,           # |r| > this in any line except 5577 -> WARNING
        min_fibers=20,           # good science fibers per spectrograph
    )
    values = _values()
    main_value = "max_abs_resid"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": "No spectrograph triggers WARNING / MAJOR ISSUE",
            "WARNING": (f"In any spectrograph: |residual/sky| > {p['warn_any']:g} in any of OH6865, "
                        f"NaD, O2"),
            "MAJOR ISSUE": (f"In any spectrograph: |residual/sky| > {p['bad_all_lines']:g} in all of "
                            f"OI5577, OH6865, NaD, O2, OR > {p['bad_any']:g} in any of OH6865, NaD, O2"),
            "UNDEFINED": "SFrame missing or too few good science fibers",
        }

    def classify(self, r):
        p = self.params
        if any(not np.isfinite(r[k]) for k in SKY_LINES):
            return QAFlag.UNDEFINED
        a = {k: abs(v) for k, v in r.items()}
        others = [a[k] for k in SKY_LINES if k != "OI5577"]
        if all(x > p["bad_all_lines"] for x in a.values()) or any(x > p["bad_any"] for x in others):
            return QAFlag.MAJOR
        if any(x > p["warn_any"] for x in others):
            return QAFlag.WARNING
        return QAFlag.GOOD

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        rows = ctx.good_sci_index
        specid = np.asarray(ctx.slitmap["spectrographid"])[rows]
        data = [(ctx.sframe_window("FLUX", *w, rows=rows), ctx.sframe_window("SKY", *w, rows=rows))
                for w in READ_WINDOWS]

        def ratios(sel):
            out = {}
            for name, (line, cont) in SKY_LINES.items():
                for (wave, f), (_, sk) in data:
                    if wave[0] <= cont[0] and wave[-1] >= cont[1]:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            mf = np.nanmedian(f[sel], axis=0)
                            ms = np.nanmedian(sk[sel], axis=0)
                        out[name] = line_ratio(wave, mf, ms, line, cont)[2]
                        break
            return out

        vals = {}
        flags = {}
        r_all = ratios(np.ones(len(rows), dtype=bool))
        for name, v in r_all.items():
            vals[f"{name}_all"] = v
        maxabs = {}
        for s in SPECS:
            sel = specid == s
            if sel.sum() < p["min_fibers"]:
                flags[s] = QAFlag.UNDEFINED
                continue
            r = ratios(sel)
            for name, v in r.items():
                vals[f"{name}_sp{s}"] = v
            flags[s] = self.classify(r)
            maxabs[s] = max(abs(r[k]) for k in SKY_LINES if k != "OI5577")
        for s in SPECS:
            vals[f"flag_sp{s}"] = FLAG_LABELS[flags[s]]
        defined = {s: f for s, f in flags.items() if f != QAFlag.UNDEFINED}
        if not defined:
            return self.result(QAFlag.UNDEFINED, note="no spectrograph evaluable", **vals)
        worst = max(defined, key=lambda s: (_SEV[defined[s]], maxabs.get(s, 0)))
        vals.update(worst_spec=int(worst), max_abs_resid=float(max(maxabs.values())))
        flag = defined[worst]
        note = "" if flag == QAFlag.GOOD else f"sp{worst}: " + ", ".join(
            f"{k}={vals.get(f'{k}_sp{worst}', np.nan):+.2f}" for k in SKY_LINES)
        return self.result(flag, note=note, **vals)
