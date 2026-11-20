"""
qa_starkinematics - prototype: consistency of the stellar kinematics (DAP RSP fit) within
extended stellar populations.

Extended stellar population (see _stars.py): a connected group of > 15 adjacent fibers with
stellar-continuum S/N (RSP med_flux / e_med_flux) > 10. Milky-Way fields usually contain only
isolated stars with unrelated kinematics -> UNDEFINED. Inside each region, a fiber is an
outlier if its stellar velocity (RSP redshift x c) deviates from the region median by more than
max(nsig x NMAD, dv_tol = 50 km/s), or its stellar dispersion (RSP disp) by more than
max(nsig x NMAD, disp_rel x median). Lenient thresholds (prototype):
GOOD     <= 25% outliers in every region
WARNING  > 25% outliers in a region, or a degenerate region (all fibers with the same stellar
         dispersion: identical fits, e.g. 7333 on DRP 1.2.0 where 22 fibers of the corrupted
         spectrograph have v = 27 km/s, disp = 1.8), or |median stellar - Halpha NP velocity|
         > dv_gas (100 km/s) in the largest region
MAJOR    > 50% outliers in a region
Note: in DAP 1.1 the stellar velocity is searched on a coarse grid (auto_z_del = 1e-4,
i.e. 30 km/s steps; e.g. 3 distinct values over all fibers of expnum 7915) and has no
error estimate, so only gross inconsistencies can be detected.
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from ._stars import C_KMS, cached_regions


@register
class StarKinematics(Metric):
    name = "qa_starkinematics"
    slot = 20
    group = GROUP_DAP
    title = "Stellar kinematics (extended populations)"
    description = ("Prototype: consistency of DAP stellar velocities and dispersions within "
                   "extended stellar populations (> 15 adjacent fibers with continuum S/N > 10)")
    requires = ("dap",)
    defaults = dict(min_snr=10.0, min_region=15, link_pitch=1.5, nsig=3.0, dv_tol=50.0,
                    disp_rel=0.5, frac_warn=0.25, frac_major=0.5, dv_gas=100.0)
    values = (
        ValueSpec("n_regions", "i8", "", "Extended stellar regions"),
        ValueSpec("n_fibers", "i8", "", "Fibers in extended stellar regions"),
        ValueSpec("n_snr", "i8", "", "Fibers with continuum S/N > min_snr"),
        ValueSpec("largest_region", "i8", "", "Fibers in the largest region"),
        ValueSpec("v_med", "f8", "km/s", "Median stellar velocity of the largest region"),
        ValueSpec("v_nmad", "f8", "km/s", "NMAD of the stellar velocity in the largest region"),
        ValueSpec("disp_med", "f8", "", "Median stellar dispersion (RSP disp) of the largest region"),
        ValueSpec("dv_star_gas", "f8", "km/s", "Median stellar - Halpha NP velocity in the largest region"),
        ValueSpec("max_frac_out", "f8", "", "Largest fraction of kinematic outliers over regions"),
        ValueSpec("n_degenerate", "i8", "", "Regions in which all fibers have the same stellar dispersion"),
    )
    main_value = "max_frac_out"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": f"<= {p['frac_warn']:.0%} kinematic outliers in every extended stellar region",
            "WARNING": (f"> {p['frac_warn']:.0%} of the fibers of a region deviate by > max({p['nsig']:g} NMAD, "
                        f"{p['dv_tol']:g} km/s) in velocity or > max({p['nsig']:g} NMAD, {p['disp_rel']:.0%}) in dispersion; "
                        f"or a degenerate region (identical stellar dispersion in all fibers); or |v_star - v_Halpha| "
                        f"> {p['dv_gas']:g} km/s in the largest region"),
            "MAJOR ISSUE": f"> {p['frac_major']:.0%} kinematic outliers in a region",
            "UNDEFINED": (f"No DAP file, or no region of > {p['min_region']} adjacent fibers with continuum "
                          f"S/N > {p['min_snr']:g}"),
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")
        reg = cached_regions(ctx, min_snr=p["min_snr"], min_region=p["min_region"], link_pitch=p["link_pitch"])
        rsp = ctx.dap_table("RSP")
        v = np.asarray(rsp["redshift"], float) * C_KMS
        d = np.asarray(rsp["disp"], float)
        vals = dict(n_regions=len(reg["regions"]), n_snr=reg["n_snr"],
                    n_fibers=int(sum(r.size for r in reg["regions"])))
        if not reg["regions"]:
            return self.result(QAFlag.UNDEFINED, note="no extended stellar population", **vals)
        fracs, ndeg = [], 0
        for r in reg["regions"]:
            vr, dr = v[r], d[r]
            mv, md = np.nanmedian(vr), np.nanmedian(dr)
            sv = 1.4826 * np.nanmedian(np.abs(vr - mv))
            sd = 1.4826 * np.nanmedian(np.abs(dr - md))
            out = (np.abs(vr - mv) > max(p["nsig"] * sv, p["dv_tol"])) | \
                  (np.abs(dr - md) > max(p["nsig"] * sd, p["disp_rel"] * abs(md)))
            fracs.append(float(np.mean(out)))
            ndeg += int(np.unique(dr[np.isfinite(dr)]).size <= 1)
        big = reg["regions"][0]
        vha = np.asarray(ctx.dap_np_columns(["vel_Halpha_6562.85"])["vel_Halpha_6562.85"], float)
        with np.errstate(invalid="ignore"):
            dsg = float(np.nanmedian(v[big] - vha[big])) if np.isfinite(vha[big]).any() else np.nan
        vals.update(largest_region=int(big.size), v_med=float(np.nanmedian(v[big])),
                    v_nmad=float(1.4826 * np.nanmedian(np.abs(v[big] - np.nanmedian(v[big])))),
                    disp_med=float(np.nanmedian(d[big])), dv_star_gas=dsg, max_frac_out=max(fracs),
                    n_degenerate=ndeg)
        f = max(fracs)
        note = f"{len(reg['regions'])} region(s), largest {big.size} fibers, outliers {f:.0%}"
        if f > p["frac_major"]:
            return self.result(QAFlag.MAJOR, note=note, **vals)
        if f > p["frac_warn"]:
            return self.result(QAFlag.WARNING, note=note, **vals)
        if ndeg:
            return self.result(QAFlag.WARNING, note=note + f"; {ndeg} degenerate region(s)", **vals)
        if np.isfinite(dsg) and abs(dsg) > p["dv_gas"]:
            return self.result(QAFlag.WARNING, note=note + f"; v_star - v_Halpha = {dsg:+.0f} km/s", **vals)
        return self.result(QAFlag.GOOD, note=note, **vals)
