"""
qa_incomplete - completeness of the data products of one exposure.

Checks that the SFrame and DAP files exist, and how many science-IFU fibers carry no
usable data (NaN / ivar<=0 / masked) in each spectrograph channel (b, r, z) and for
each spectrograph.
"""

from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..context import CHANNEL_WINDOWS
from ..registry import GROUP_GEN, Metric, ValueSpec, register


@register
class IncompleteData(Metric):
    name = "qa_incomplete"
    slot = 1
    group = GROUP_GEN
    title = "Incomplete data"
    description = ("Completeness of the exposure: presence of SFrame and DAP files, and "
                   "fraction of science fibers without usable data per channel and "
                   "spectrograph")
    requires = ("sframe_data", "dap")
    defaults = dict(
        min_good_pix_frac=0.5,     # fiber is 'lost' in a channel below this good-pixel fraction
        warn_frac_lost=0.05,       # WARNING: > 5% of science fibers lost (any channel)
        major_frac_lost_channel=0.25,  # MAJOR: > 25% of science fibers lost in one channel
        major_frac_lost_spec=0.5,  # MAJOR: > 50% of the science fibers of one spectrograph lost
        min_dap_coverage=0.95,     # WARNING: DAP analysed < 95% of the usable science fibers
    )
    values = (
        ValueSpec("sframe_found", "bool", "", "SFrame file found"),
        ValueSpec("dap_found", "bool", "", "DAP file found"),
        ValueSpec("n_sci", "i8", "", "Number of science-IFU fibers"),
        ValueSpec("n_lost", "i8", "", "Science fibers without usable data in any channel"),
        ValueSpec("frac_lost", "f8", "", "Fraction of science fibers lost (any channel)"),
        ValueSpec("frac_lost_b", "f8", "", "Fraction of science fibers lost in b channel"),
        ValueSpec("frac_lost_r", "f8", "", "Fraction of science fibers lost in r channel"),
        ValueSpec("frac_lost_z", "f8", "", "Fraction of science fibers lost in z channel"),
        ValueSpec("max_frac_lost_spec", "f8", "", "Largest fraction of lost science fibers in one spectrograph"),
        ValueSpec("n_fibstatus_bad", "i8", "", "Science fibers with SLITMAP fibstatus != 0"),
        ValueSpec("dap_coverage", "f8", "", "DAP-analysed / usable science fibers"),
    )
    main_value = "frac_lost"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "WARNING": (f"DAP file missing, or >{p['warn_frac_lost']:.0%} of science fibers "
                        f"without usable data, or DAP covers <{p['min_dap_coverage']:.0%} of "
                        f"usable science fibers"),
            "MAJOR ISSUE": (f"SFrame missing, or >{p['major_frac_lost_channel']:.0%} of science "
                            f"fibers lost in one channel, or >{p['major_frac_lost_spec']:.0%} "
                            f"lost in one spectrograph"),
            "UNDEFINED": "SFrame could not be read",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.MAJOR, note="SFrame missing",
                               sframe_found=False, dap_found=ctx.has_dap)

        sci = ctx.sci_fiber_index
        sm = ctx.slitmap[sci]
        gf = ctx.fiber_good_fraction
        lost_ch = {ch: gf[ch][sci] < p["min_good_pix_frac"] for ch in CHANNEL_WINDOWS}
        lost = np.zeros(len(sci), dtype=bool)
        for v in lost_ch.values():
            lost |= v
        n_sci = int(len(sci))
        frac = {ch: float(v.mean()) if n_sci else np.nan for ch, v in lost_ch.items()}
        frac_lost = float(lost.mean()) if n_sci else np.nan

        specid = np.asarray(sm["spectrographid"])
        spec_frac = [float(lost[specid == s].mean()) for s in np.unique(specid)]
        max_spec = max(spec_frac) if spec_frac else np.nan
        n_fibstat = int(np.sum(np.asarray(sm["fibstatus"]) != 0))

        dap_cov = np.nan
        if ctx.has_dap:
            pt = ctx.dap_table("PT")
            good_ids = set(np.asarray(sm["fiberid"])[~lost].tolist())
            dap_ids = set(np.asarray(pt["fiberid"]).astype(int).tolist())
            if good_ids:
                dap_cov = len(dap_ids & good_ids) / len(good_ids)

        vals = dict(sframe_found=True, dap_found=ctx.has_dap, n_sci=n_sci,
                    n_lost=int(lost.sum()), frac_lost=frac_lost,
                    frac_lost_b=frac["b"], frac_lost_r=frac["r"], frac_lost_z=frac["z"],
                    max_frac_lost_spec=max_spec, n_fibstatus_bad=n_fibstat,
                    dap_coverage=dap_cov)

        notes = []
        flag = QAFlag.GOOD
        worst_ch = max(frac, key=frac.get)
        if frac[worst_ch] > p["major_frac_lost_channel"] or max_spec > p["major_frac_lost_spec"]:
            flag = QAFlag.MAJOR
            notes.append(f"{frac[worst_ch]:.0%} fibers lost in {worst_ch}, worst spectrograph {max_spec:.0%}")
        else:
            if not ctx.has_dap:
                flag = QAFlag.WARNING
                notes.append("DAP missing")
            if frac_lost > p["warn_frac_lost"]:
                flag = QAFlag.WARNING
                notes.append(f"{frac_lost:.1%} science fibers lost")
            if np.isfinite(dap_cov) and dap_cov < p["min_dap_coverage"]:
                flag = QAFlag.WARNING
                notes.append(f"DAP coverage {dap_cov:.1%}")
        return self.result(flag, note=", ".join(notes), **vals)
