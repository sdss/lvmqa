"""
qa_status - tile status (information, NOT a QA flag): what the tile contains.

    status_auroral  auroral lines [OIII]4363, [NII]5755, [SIII]6312 detected (NP S/N > 5) in
                    >= 1% of the fibers (any of the lines; v1.8: [OII]7320,7330 and [SII]4068,4076
                    no longer used)
    status_bright   bright lines ([OII]3727, Hgamma, Hbeta, [OIII]4959,5007, [NII]6548,6583,
                    Halpha, [SII]6716,6731, [SIII]9069,9532) detected in >= 1% of the fibers
    status_faint    faint lines (incl. auroral; see _lines.FAINT_GROUPS, without [OI]) detected
                    in >= 1% of the fibers
    status_sp       an extended stellar population exists (> 15 adjacent fibers with stellar
                    continuum S/N > 10; same definition as qa_starkinematics)
A line counts as detected in a fiber if NP S/N > min_snr AND it is kinematically consistent with
Halpha in the same fiber (Halpha S/N > min_snr; |v_line - v_Halpha| < dv_max = 80 km/s after the
rest-wavelength corrections of qa_gaskinematics; 0.4 < disp_line / disp_Halpha < 2.5 [A]).
Without this check the NP fit "detects" sky-subtraction residuals: e.g. [SIII]6312 in IC1613
(5351, 3.4% of the fibers, at -141 km/s = sky [OI]6300 wing) and [OII]7320 in 61411 (5%, at
-90 km/s, disp 0.4 x Halpha = OH 7316.3). Genuine lines on the DRP 1.2.0 set have |dv| < 25 km/s ([OII]7320 -52: blend centroid)
and disp ratios 0.9-1.6 (HeI4471 0.53: NP deblending with [FeII]).
OI8446 is excluded from status_faint: it is a bright airglow line, and in Milky-Way fields its sky
residuals have the gas velocity (DRP 1.2.0: "detected" in 99% of the fibers of 7331-7334).
All False when the DAP file is missing (the flag of qa_incomplete says so).
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_STATUS, Metric, ValueSpec, register
from ._lines import AURORAL, BRIGHT_GROUPS, FAINT_GROUPS
from .dap_gaskinematics import REST

C_KMS = 299792.458
HA = "Halpha_6562.85"
from ._stars import cached_regions


def _members(groups):
    return tuple(prim for prim, _ in groups.values())


@register
class TileStatus(Metric):
    name = "qa_status"
    slot = None
    status = True
    group = GROUP_STATUS
    title = "Tile status (information)"
    description = ("Information per tile (not a QA flag): auroral, bright and faint emission "
                   "lines detected, extended stellar population present")
    requires = ("dap",)
    defaults = dict(min_snr=5.0, min_frac=0.01, dv_max=80.0, disp_ratio=(0.4, 2.5),
                    exclude=("OI_8446.0",),  # airglow line: sky residuals at the MW gas velocity
                    sp_min_snr=10.0, sp_min_region=15, sp_link_pitch=1.5)
    values = (
        ValueSpec("auroral", "bool", "", "Auroral lines detected (S/N > 5) in >= 1% of the fibers"),
        ValueSpec("bright", "bool", "", "Bright lines detected (S/N > 5) in >= 1% of the fibers"),
        ValueSpec("faint", "bool", "", "Faint lines (incl. auroral) detected (S/N > 5) in >= 1% of the fibers"),
        ValueSpec("sp", "bool", "", "Extended stellar population present"),
        ValueSpec("frac_auroral", "f8", "", "Largest fraction of fibers with an auroral line detected"),
        ValueSpec("best_auroral", "U16", "", "Auroral line detected in most fibers"),
        ValueSpec("frac_bright", "f8", "", "Largest fraction of fibers with a bright line detected"),
        ValueSpec("frac_faint", "f8", "", "Largest fraction of fibers with a faint line detected"),
        ValueSpec("best_faint", "U16", "", "Faint line detected in most fibers"),
        ValueSpec("n_sp_regions", "i8", "", "Extended stellar regions"),
    )
    criteria = {"GOOD": "Information only (always GOOD when the DAP file exists)",
                "UNDEFINED": "No DAP file"}

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")

        def vel(t, l):
            wl = float(l.rsplit("_", 1)[1])
            return np.asarray(t[f"vel_{l}"], float) + C_KMS * (wl - REST.get(l, wl)) / REST.get(l, wl)

        def frac(lines):
            lines = tuple(lines)
            cols = [f"{q}_{l}" for l in dict.fromkeys(lines + (HA,)) for q in ("flux", "e_flux", "vel", "disp")]
            t = ctx.dap_np_columns(cols)
            n = len(t)
            with np.errstate(invalid="ignore", divide="ignore"):
                sa = np.asarray(t[f"flux_{HA}"], float) / np.asarray(t[f"e_flux_{HA}"], float)
                va, da = vel(t, HA), np.asarray(t[f"disp_{HA}"], float)
                out = {}
                for l in lines:
                    f, e = np.asarray(t[f"flux_{l}"], float), np.asarray(t[f"e_flux_{l}"], float)
                    det = (e > 0) & (f / e > p["min_snr"])
                    if l != HA:
                        r = np.asarray(t[f"disp_{l}"], float) / da
                        det &= (sa > p["min_snr"]) & (np.abs(vel(t, l) - va) < p["dv_max"]) & \
                               (r > p["disp_ratio"][0]) & (r < p["disp_ratio"][1])
                    out[l] = float(np.sum(det)) / max(n, 1)
            return out

        fa = frac(AURORAL)
        fb = frac(_members(BRIGHT_GROUPS))
        ff = frac(tuple(l for l in dict.fromkeys(_members(FAINT_GROUPS) + AURORAL) if l not in p["exclude"]))
        reg = cached_regions(ctx, min_snr=p["sp_min_snr"], min_region=p["sp_min_region"],
                             link_pitch=p["sp_link_pitch"])
        ba, bf = max(fa, key=fa.get), max(ff, key=ff.get)
        vals = dict(auroral=fa[ba] >= p["min_frac"], bright=max(fb.values()) >= p["min_frac"],
                    faint=ff[bf] >= p["min_frac"], sp=len(reg["regions"]) > 0,
                    frac_auroral=fa[ba], best_auroral=ba, frac_bright=max(fb.values()),
                    frac_faint=ff[bf], best_faint=bf, n_sp_regions=len(reg["regions"]))
        return self.result(QAFlag.GOOD, **vals)
