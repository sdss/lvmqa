"""
qa_wavecal_b / qa_wavecal_r / qa_wavecal_z - wavelength calibration accuracy per channel
from the centroids of bright sky lines (DR20 definition, measured here on the SFrame).

For each good science fiber the sky lines of the channel (see _skylines.SKY_LINES) are fitted
with a Gaussian in FLUX + SKY and the offset d = centroid - rest wavelength is compared with
the per-fiber LSF (FWHM) at the line:

    fiber 'off by LSF/5' : |d| > LSF/5 for any line of the channel
    fiber 'off by LSF'   : |d| > LSF   for any line of the channel

GOOD:    <= 5% of the fibers off by LSF/5
WARNING: >  5% off by LSF/5 (but <= 5% off by LSF or failed)
MAJOR:   >  5% off by LSF or failed

A fiber 'failed' if fewer than half of the channel's sky lines could be fitted with S/N >= 10
(the bright sky lines normally reach S/N 20-40 in every fiber; e.g. in expnum 7333, DRP 1.2.0,
half of the spectrograph-1 fibers fail in z, which DR20 reported as MAJOR ISSUE). Without
counting failures, a corrupted spectrograph would be invisible to this test.

Differences to DR20: DR20 used per-fiber Gaussian fits of the CFrame (Knox tables) and one
mean LSF value per line; here the fit is done on FLUX + SKY of the SFrame (identical sky
spectrum) with the per-fiber LSF from the SFrame LSF extension. The b channel has a single
usable line ([OI]5577), which the DRP itself uses to correct the zero point, so qa_wavecal_b
verifies that correction only.
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register
from ._skylines import lines_of, skylines


def _values(channel):
    v = [ValueSpec("n_fibers", "i8", "", "Good science fibers with at least one accepted line fit"),
         ValueSpec("frac_measured", "f8", "", "Fraction of good science fibers with an accepted fit"),
         ValueSpec("frac_failed", "f8", "", "Fraction of good science fibers with < half of the lines fitted"),
         ValueSpec("frac_warn", "f8", "", "Fraction of fibers with |offset| > LSF/5 in any line, or failed"),
         ValueSpec("frac_major", "f8", "", "Fraction of fibers with |offset| > LSF in any line, or failed"),
         ValueSpec("p95_rel_offset", "f8", "", "95th percentile over fibers of max_lines |offset|/LSF"),
         ValueSpec("max_spec_offset", "f8", "Angstrom", "Largest |median offset| of a spectrograph (over lines)"),
         ValueSpec("worst_spec", "i8", "", "Spectrograph with the largest |median offset|")]
    for name in lines_of(channel):
        v.append(ValueSpec(f"off_{name}", "f8", "Angstrom", f"Median centroid - rest wavelength of {name}"))
    for s in (1, 2, 3):
        v.append(ValueSpec(f"drp_skyoff_sp{s}", "f8", "Angstrom",
                           f"DRP sky-line offset before its correction ({channel.upper()}{s} WAVE SKYOFF)"))
    return tuple(v)


class _WaveCal(Metric):
    channel = "b"
    group = GROUP_DRP
    requires = ("sframe_data",)
    defaults = dict(
        warn_lsf_frac=0.2,     # |offset| > warn_lsf_frac * LSF -> fiber off (WARNING level)
        major_lsf_frac=1.0,    # |offset| > major_lsf_frac * LSF -> fiber off (MAJOR level)
        frac_thresh=0.05,      # fraction of fibers above which the exposure is flagged
        min_frac_measured=0.2, # fewer measured fibers -> UNDEFINED
        exclude=(),            # sky lines measured and reported but not used for the flag
    )
    main_value = "frac_warn"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        f = f"{p['frac_thresh']:.0%}"
        w = f"LSF/{1 / p['warn_lsf_frac']:g}" if p["warn_lsf_frac"] < 1 else f"{p['warn_lsf_frac']:g} LSF"
        m = "LSF" if p["major_lsf_frac"] == 1 else f"{p['major_lsf_frac']:g} LSF"
        return {
            "GOOD": f"<= {f} of fibers with a sky-line offset > {w}",
            "WARNING": f"> {f} of fibers with a sky-line offset > {w}, <= {f} with offset > {m}",
            "MAJOR ISSUE": (f"> {f} of fibers with a sky-line offset > {m} or with < half of the "
                            "sky lines measurable (S/N < 10)"),
            "UNDEFINED": (f"SFrame missing, channel not covered, or < {p['min_frac_measured']:.0%} "
                          "of the fibers with an accepted sky-line fit"),
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        sk = skylines(ctx)
        allnames = [n for n in lines_of(self.channel) if n in sk["lines"]]
        names = [n for n in allnames if n not in tuple(p["exclude"])]
        vals = {}
        for n in allnames:          # reported for all lines (also the excluded ones)
            l_ = sk["lines"][n]
            d = np.where(l_["ok"], l_["mu"] - l_["rest"], np.nan)
            if np.isfinite(d).any():
                vals[f"off_{n}"] = float(np.nanmedian(d))
        for s in (1, 2, 3):
            v = ctx.header.get(f"{self.channel.upper()}{s} WAVE SKYOFF")
            vals[f"drp_skyoff_sp{s}"] = float(v) if v is not None else float("nan")
        if not names:
            return self.result(QAFlag.UNDEFINED, note="no sky line in the channel", **vals)
        L = [sk["lines"][n] for n in names]
        off = np.array([np.where(l["ok"], l["mu"] - l["rest"], np.nan) for l in L])
        lsf = np.array([l["lsf"] for l in L])
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(off) / lsf
        measured = np.isfinite(rel).any(axis=0)
        failed = np.isfinite(rel).sum(axis=0) < 0.5 * len(names)
        specid = sk["specid"]
        n_meas = int(measured.sum())
        vals.update(n_fibers=n_meas, frac_measured=n_meas / max(len(measured), 1),
                    frac_failed=float(failed.mean()))
        if vals["frac_measured"] < p["min_frac_measured"]:
            return self.result(QAFlag.UNDEFINED, note=f"only {n_meas} fibers measured", **vals)
        relmax = np.max(np.where(np.isfinite(rel), rel, -np.inf), axis=0)
        fw = float(np.mean((relmax > p["warn_lsf_frac"]) | failed))
        fm = float(np.mean((relmax > p["major_lsf_frac"]) | failed))
        relmax = relmax[measured]
        sp_off = {}
        for s in (1, 2, 3):
            sel = specid == s
            if sel.any() and np.isfinite(off[:, sel]).any():
                sp_off[s] = float(np.nanmedian(off[:, sel]))
        if sp_off:
            ws = max(sp_off, key=lambda s: abs(sp_off[s]))
            vals.update(max_spec_offset=abs(sp_off[ws]), worst_spec=int(ws))
        vals.update(frac_warn=fw, frac_major=fm, p95_rel_offset=float(np.percentile(relmax, 95)))
        if fm > p["frac_thresh"]:
            return self.result(QAFlag.MAJOR, note=f"{fm:.0%} fibers off by > LSF or failed "
                                                  f"({vals['frac_failed']:.0%} failed)", **vals)
        if fw > p["frac_thresh"]:
            return self.result(QAFlag.WARNING, note=f"{fw:.0%} fibers off by > LSF/5", **vals)
        return self.result(QAFlag.GOOD, **vals)


# z: sky8988 is not used for the flag (complex, blended line profile; v1.15)
EXCLUDE = {"z": ("sky8988",)}


def _make(channel, slot):
    excl = EXCLUDE.get(channel, ())
    lines = ", ".join(f"{n[3:]}" for n in lines_of(channel) if n not in excl)
    cls = type(f"WaveCal_{channel}", (_WaveCal,), dict(
        name=f"qa_wavecal_{channel}", slot=slot, channel=channel,
        defaults={**_WaveCal.defaults, "exclude": excl},
        title=f"Wavelength calibration ({channel})",
        description=(f"Wavelength calibration in the {channel} channel: centroids of bright sky lines "
                     f"({lines} A) vs. rest wavelengths, per fiber, in units of the LSF FWHM"),
        values=_values(channel)))
    return register(cls)


WaveCalB = _make("b", 11)
WaveCalR = _make("r", 12)
WaveCalZ = _make("z", 13)
