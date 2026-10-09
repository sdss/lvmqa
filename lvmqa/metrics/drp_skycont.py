"""
qa_skycont_sub - sky continuum subtraction: over-subtraction test on science fibers.

Astrophysical continuum cannot be negative, so science fibers whose sky-subtracted
continuum is significantly below zero indicate over-subtracted sky. For each good science
fiber and each window (DR20 windows: b 5400-5500, r 6000-6100, z 8800-8900 A):

    snr = median(FLUX) / median(error)        (per-pixel continuum S/N, DR20 definition)

and the fraction of fibers with snr < -1 and < -3 is computed per channel.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

WINDOWS = {"b": (5400.0, 5500.0), "r": (6000.0, 6100.0), "z": (8800.0, 8900.0)}


def _values():
    v = [ValueSpec("n_fibers", "i8", "", "Good science fibers used")]
    for ch in WINDOWS:
        v += [ValueSpec(f"snr_med_{ch}", "f8", "", f"Median per-pixel continuum S/N of science fibers ({ch})"),
              ValueSpec(f"frac_neg1_{ch}", "f8", "", f"Fraction of science fibers with continuum S/N < -snr_warn ({ch})"),
              ValueSpec(f"frac_neg3_{ch}", "f8", "", f"Fraction of science fibers with continuum S/N < -snr_major ({ch})")]
    v.append(ValueSpec("max_frac_neg1", "f8", "", "Largest frac_neg1 over channels"))
    return tuple(v)


@register
class SkyContSub(Metric):
    name = "qa_skycont_sub"
    slot = 7
    group = GROUP_DRP
    title = "Sky continuum subtraction"
    description = ("Sky continuum subtraction quality: fraction of science fibers with "
                   "significantly negative continuum (over-subtracted sky) per channel")
    requires = ("sframe_data",)
    defaults = dict(
        snr_warn=1.0,      # fiber 'negative' if continuum S/N < -snr_warn
        snr_major=3.0,     # fiber 'strongly negative' if continuum S/N < -snr_major
        frac_thresh=0.05,  # exposure flag if fraction exceeds this in any channel
        min_fibers=50,
    )
    values = _values()
    main_value = "max_frac_neg1"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        f = f"{p['frac_thresh']:.0%}"
        return {
            "GOOD": f"<= {f} of science fibers with continuum S/N < -{p['snr_warn']:g} in all channels",
            "WARNING": f"> {f} of science fibers with continuum S/N < -{p['snr_warn']:g} in any channel",
            "MAJOR ISSUE": f"> {f} of science fibers with continuum S/N < -{p['snr_major']:g} in any channel",
            "UNDEFINED": "SFrame missing or too few good science fibers",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        rows = ctx.good_sci_index
        vals = {}
        fw, fm = {}, {}
        n_used = 0
        for ch, (w0, w1) in WINDOWS.items():
            _, f = ctx.sframe_window("FLUX", w0, w1, rows=rows)
            _, iv = ctx.sframe_window("IVAR", w0, w1, rows=rows)
            _, m = ctx.sframe_window("MASK", w0, w1, rows=rows)
            good = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (m == 0)
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                sig = np.nanmedian(np.where(good, 1.0 / np.sqrt(np.where(good, iv, 1.0)), np.nan), axis=1)
                med = np.nanmedian(np.where(good, f, np.nan), axis=1)
                snr = med / sig
            ok = np.isfinite(snr) & (good.mean(axis=1) > 0.5)
            n_used = max(n_used, int(ok.sum()))
            if ok.sum() < p["min_fibers"]:
                continue
            fw[ch] = float(np.mean(snr[ok] < -p["snr_warn"]))
            fm[ch] = float(np.mean(snr[ok] < -p["snr_major"]))
            vals.update({f"snr_med_{ch}": float(np.median(snr[ok])),
                         f"frac_neg1_{ch}": fw[ch], f"frac_neg3_{ch}": fm[ch]})
        vals["n_fibers"] = n_used
        if not fw:
            return self.result(QAFlag.UNDEFINED, note="too few good science fibers", **vals)
        vals["max_frac_neg1"] = max(fw.values())
        if max(fm.values()) > p["frac_thresh"]:
            ch = max(fm, key=fm.get)
            return self.result(QAFlag.MAJOR, note=f"{fm[ch]:.0%} fibers S/N<-{p['snr_major']:g} in {ch}", **vals)
        if max(fw.values()) > p["frac_thresh"]:
            ch = max(fw, key=fw.get)
            return self.result(QAFlag.WARNING, note=f"{fw[ch]:.0%} fibers S/N<-{p['snr_warn']:g} in {ch}", **vals)
        return self.result(QAFlag.GOOD, **vals)
