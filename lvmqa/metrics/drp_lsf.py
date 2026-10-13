"""
qa_lsf - consistency of the measured sky-line widths with the LSF of the SFrame.

For each good science fiber the bright sky lines (see _skylines.SKY_LINES, shared fits with
qa_wavecal_*) are fitted with a Gaussian in FLUX + SKY. The measured FWHM is compared with the
LSF (FWHM) of the fiber at the line wavelength (SFrame extension LSF):

    q_line = (FWHM_measured / LSF) / ref_line
    Q_fiber,channel = median of q_line over the lines of the channel

ref_line is the typical FWHM/LSF ratio of the line: most OH "lines" are unresolved
Lambda-doublets / blends, and the LSF is not exactly Gaussian, so even with a perfect LSF the
Gaussian FWHM of an OH line differs from the LSF by -5% ... +9% in a line-dependent but stable
way. The reference values were measured on the DRP 1.2.0 (21 exposures) and 1.3.2 (16
exposures) test sets (median over fibers and exposures, average of both versions; the
versions differ by <= 0.05). [OI]5577 is a single line: ref = 1.

A fiber deviates if |Q - 1| > tol in a channel.
GOOD:    <= 5% of the measured fibers deviate by more than warn_tol (10%) in every channel
WARNING: >  5% deviate by more than warn_tol in any channel
MAJOR:   >  5% deviate by more than major_tol (20%) in any channel
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register
from ._skylines import CHANNELS, SKY_LINES, lines_of, skylines

# typical Gaussian-FWHM / LSF ratio of each sky line (see module docstring)
REF_RATIO = {
    "sky5577": 1.00,
    "sky6533": 1.06, "sky6923": 1.07, "sky6939": 1.09, "sky7392": 1.09,
    "sky7914": 1.09, "sky8344": 1.07, "sky8399": 0.97, "sky8827": 1.07,
    "sky8988": 0.97, "sky9552": 1.00, "sky9719": 0.98,
}


def _values():
    v = [ValueSpec("n_fibers", "i8", "", "Good science fibers with at least one accepted line fit")]
    for ch in CHANNELS:
        v += [ValueSpec(f"q_med_{ch}", "f8", "", f"Median over fibers of Q = (FWHM/LSF)/ref ({ch})"),
              ValueSpec(f"frac_warn_{ch}", "f8", "", f"Fraction of fibers with |Q-1| > warn_tol ({ch})"),
              ValueSpec(f"frac_major_{ch}", "f8", "", f"Fraction of fibers with |Q-1| > major_tol ({ch})"),
              ValueSpec(f"lsf_med_{ch}", "f8", "Angstrom", f"Median LSF FWHM at the sky lines ({ch})")]
    for n in SKY_LINES:
        v.append(ValueSpec(f"ratio_{n}", "f8", "", f"Median measured FWHM / LSF of {n} (not normalised)"))
    v += [ValueSpec("worst_channel", "U1", "", "Channel with the largest fraction of deviating fibers"),
          ValueSpec("worst_spec", "i8", "", "Spectrograph with the largest |median Q - 1| (worst channel)"),
          ValueSpec("max_frac_warn", "f8", "", "Largest frac_warn over channels")]
    return tuple(v)


@register
class LSF(Metric):
    name = "qa_lsf"
    slot = 14
    group = GROUP_DRP
    title = "Line-spread function"
    description = ("LSF accuracy: Gaussian FWHM of bright sky lines vs. the LSF of the SFrame, "
                   "per fiber and channel, normalised by the typical ratio of each line")
    requires = ("sframe_data",)
    defaults = dict(
        warn_tol=0.10,      # |Q - 1| above -> fiber deviates (WARNING level)
        major_tol=0.20,     # |Q - 1| above -> fiber deviates (MAJOR level)
        frac_thresh=0.05,   # fraction of deviating fibers above which the exposure is flagged
        min_fibers=50,
        ref_ratio=REF_RATIO,
    )
    values = _values()
    main_value = "max_frac_warn"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        f = f"{p['frac_thresh']:.0%}"
        q = "Q = (FWHM_sky/LSF)/ref"
        return {
            "GOOD": f"<= {f} of fibers with |{q} - 1| > {p['warn_tol']:.0%} in every channel",
            "WARNING": f"> {f} of fibers with |{q} - 1| > {p['warn_tol']:.0%} in any channel",
            "MAJOR ISSUE": f"> {f} of fibers with |{q} - 1| > {p['major_tol']:.0%} in any channel",
            "UNDEFINED": "SFrame missing or too few fibers with sky-line fits",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        sk = skylines(ctx)
        specid = sk["specid"]
        vals, fw, fm, spdev = {}, {}, {}, {}
        any_meas = np.zeros(len(specid), dtype=bool)
        for n, l in sk["lines"].items():
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(l["ok"], l["fwhm"] / l["lsf"], np.nan)
            if np.isfinite(r).any():
                vals[f"ratio_{n}"] = float(np.nanmedian(r))
        for ch in CHANNELS:
            names = [n for n in lines_of(ch) if n in sk["lines"]]
            if not names:
                continue
            q = []
            for n in names:
                l = sk["lines"][n]
                with np.errstate(invalid="ignore", divide="ignore"):
                    q.append(np.where(l["ok"], l["fwhm"] / l["lsf"], np.nan) / p["ref_ratio"].get(n, 1.0))
            q = np.array(q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                Q = np.nanmedian(q, axis=0)
            ok = np.isfinite(Q)
            any_meas |= ok
            vals[f"lsf_med_{ch}"] = float(np.nanmedian([sk["lines"][n]["lsf"] for n in names]))
            if ok.sum() < p["min_fibers"]:
                continue
            dev = np.abs(Q[ok] - 1)
            fw[ch], fm[ch] = float(np.mean(dev > p["warn_tol"])), float(np.mean(dev > p["major_tol"]))
            vals.update({f"q_med_{ch}": float(np.median(Q[ok])),
                         f"frac_warn_{ch}": fw[ch], f"frac_major_{ch}": fm[ch]})
            spdev[ch] = {s: float(np.median(Q[ok & (specid == s)]) - 1)
                         for s in (1, 2, 3) if (ok & (specid == s)).any()}
        vals["n_fibers"] = int(any_meas.sum())
        if not fw:
            return self.result(QAFlag.UNDEFINED, note="too few fibers with sky-line fits", **vals)
        wch = max(fw, key=lambda c: (fm[c], fw[c]))
        vals.update(worst_channel=wch, max_frac_warn=max(fw.values()))
        if spdev.get(wch):
            vals["worst_spec"] = int(max(spdev[wch], key=lambda s: abs(spdev[wch][s])))
        if fm[wch] > p["frac_thresh"]:
            return self.result(QAFlag.MAJOR, note=f"{fm[wch]:.0%} fibers |Q-1|>{p['major_tol']:.0%} in {wch}",
                               **vals)
        if max(fw.values()) > p["frac_thresh"]:
            c = max(fw, key=fw.get)
            return self.result(QAFlag.WARNING, note=f"{fw[c]:.0%} fibers |Q-1|>{p['warn_tol']:.0%} in {c}",
                               **vals)
        return self.result(QAFlag.GOOD, **vals)
