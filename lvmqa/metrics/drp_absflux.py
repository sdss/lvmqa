"""
qa_absflux - absolute flux calibration: agreement of the sensitivity curves derived from
standard stars with template matching (MOD) and from field stars in the science IFU (SCI).

Same logic as DR20: per channel z = |MOD - SCI| / sqrt(rms_MOD^2 + rms_SCI^2) using the
mean sensitivities and their rms (drpall columns ``mod_mean_sen*``, ``sci_mean_sen*``,
``*_rms_sen*``; fallback: SFrame header MODSENM*/MODSENR*/SCISENM*/SCISENR*).
"""

from __future__ import annotations

import math

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

CHANNELS = ("b", "r", "z")


def _valid(v) -> bool:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > -999.0


def _values():
    v = [ValueSpec("fluxcal", "U4", "", "Flux calibration method used by the DRP"),
         ValueSpec("mod_found", "bool", "", "MOD sensitivities available in all channels"),
         ValueSpec("n_outliers", "i8", "", "Channels with z > z_thresh")]
    for ch in CHANNELS:
        v += [ValueSpec(f"mod_sen_{ch}", "f8", "", f"Mean MOD sensitivity, {ch} channel"),
              ValueSpec(f"mod_rms_{ch}", "f8", "", f"RMS of MOD sensitivity, {ch} channel"),
              ValueSpec(f"sci_sen_{ch}", "f8", "", f"Mean SCI (field stars) sensitivity, {ch} channel"),
              ValueSpec(f"sci_rms_{ch}", "f8", "", f"RMS of SCI sensitivity, {ch} channel"),
              ValueSpec(f"z_{ch}", "f8", "", f"|MOD-SCI|/sqrt(rms_MOD^2+rms_SCI^2), {ch} channel")]
    return tuple(v)


@register
class AbsFlux(Metric):
    name = "qa_absflux"
    slot = 3
    group = GROUP_DRP
    title = "Absolute flux calibration (MOD vs SCI)"
    description = "Absolute flux calibration: MOD vs SCI sensitivity agreement across all channels"
    requires = ("drpall", "sframe_header")
    defaults = dict(
        z_thresh=1.0,              # channel is an outlier if z > z_thresh
        n_major=2,                 # >= n_major outlier channels -> MAJOR (1 -> WARNING)
        major_if_no_fluxcal=True,  # fluxcal == NONE -> MAJOR
    )
    values = _values()
    main_value = "n_outliers"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": f"MOD and SCI sensitivities agree within {p['z_thresh']:g} sigma in all channels",
            "WARNING": f"MOD and SCI differ by > {p['z_thresh']:g} sigma in one channel",
            "MAJOR ISSUE": (f"MOD and SCI differ by > {p['z_thresh']:g} sigma in >= {p['n_major']} "
                            f"channels, OR MOD absent" +
                            (", OR no flux calibration (fluxcal = NONE)" if p["major_if_no_fluxcal"] else "")),
            "UNDEFINED": "SCI sensitivities or rms unavailable",
        }

    def compute(self, ctx):
        p = self.params
        fluxcal = str(ctx.value("fluxcal", ("FLUXCAL",), "") or "").strip()
        vals = {"fluxcal": fluxcal}
        data = {}
        for ch in CHANNELS:
            C = ch.upper()
            for kind, drp, hdr in (("mod_sen", f"mod_mean_sen{ch}", f"MODSENM{C}"),
                                   ("mod_rms", f"mod_rms_sen{ch}", f"MODSENR{C}"),
                                   ("sci_sen", f"sci_mean_sen{ch}", f"SCISENM{C}"),
                                   ("sci_rms", f"sci_rms_sen{ch}", f"SCISENR{C}")):
                v = ctx.drp(drp)
                if not _valid(v) and ctx.has_sframe:
                    v = ctx.header.get(hdr)
                data[f"{kind}_{ch}"] = float(v) if _valid(v) else float("nan")
        vals.update(data)
        mod_found = all(math.isfinite(data[f"mod_sen_{ch}"]) and data[f"mod_sen_{ch}"] != 0
                        for ch in CHANNELS)
        vals["mod_found"] = mod_found

        if p["major_if_no_fluxcal"] and fluxcal.upper() == "NONE":
            return self.result(QAFlag.MAJOR, note="no flux calibration (fluxcal=NONE)", **vals)
        if not mod_found:
            return self.result(QAFlag.MAJOR, note="MOD sensitivities absent", **vals)

        n_out = 0
        for ch in CHANNELS:
            m, s = data[f"mod_sen_{ch}"], data[f"sci_sen_{ch}"]
            rm, rs = data[f"mod_rms_{ch}"], data[f"sci_rms_{ch}"]
            if not all(math.isfinite(x) for x in (m, s, rm, rs)) or (rm == 0 and rs == 0):
                return self.result(QAFlag.UNDEFINED, note=f"SCI sensitivity unavailable ({ch})", **vals)
            z = abs(m - s) / math.hypot(rm, rs)
            vals[f"z_{ch}"] = z
            n_out += z > p["z_thresh"]
        vals["n_outliers"] = int(n_out)
        if n_out >= p["n_major"]:
            return self.result(QAFlag.MAJOR, note=f"MOD/SCI disagree in {n_out} channels", **vals)
        if n_out:
            return self.result(QAFlag.WARNING, note="MOD/SCI disagree in 1 channel", **vals)
        return self.result(QAFlag.GOOD, **vals)
