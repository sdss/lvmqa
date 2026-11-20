"""
qa_gradient - flux-scale pattern across the IFU in the Halpha map that real (astrophysical)
flux variations cannot explain. Two components:

1. Spectrograph sectors (per exposure, every exposure). The 1801 science fibers are split
   between the three spectrographs in three fixed 120-deg sectors of the IFU. A relative
   flux-scale error of one spectrograph makes log F(Halpha) jump across the sector boundaries.
   Adjacent fiber pairs across each boundary are compared with "control" pairs of the same
   sector, same step direction and nearby position, which measure the real change of the
   Halpha surface brightness along that step (multitile/sectors.py). Result: offsets o1, o2, o3
   (log10 flux scale, sum 0) and their spread max(o) - min(o).
2. Smooth IFU gradient (mosaics only, from qa_fluxmap). In a mosaic of >= 30 overlapping
   tiles, qa_fluxmap fits a common "dipole" of the overlap edge offsets: the flux scale of one
   side of the IFU relative to the opposite side, which the same sky seen by two tiles cannot
   produce. ifu_dipole_dex = difference between opposite overlap regions (+-0.21 deg from the
   centre). It is a property of the whole mosaic (run value), attached to every exposure of it.
   (Inside one exposure a smooth instrumental gradient cannot be told from a real sky gradient.)

GOOD     sector spread and IFU dipole <= 0.04 dex (10%), or not significant (< 3 sigma)
WARNING  sector spread or IFU dipole > 0.04 dex (significant)
MAJOR    sector spread or IFU dipole > 0.10 dex (26%, significant)
UNDEFINED  no DAP file / SLITMAP, or < 2 sector boundaries with >= 10 usable cross pairs (faint
           fields: Halpha S/N > 5 needed in both fibers) and no mosaic dipole

Validation: DRP 1.2.0 7333 (corrupted sp1): o1 = +1.10 dex -> MAJOR; all other 1.2.0 exposures
<= 0.014 dex; Gum (1.3.2, 545 exposures): sector offsets +0.0025 / -0.0006 / -0.0020 dex
(+-0.001), i.e. no spectrograph pattern, but IFU dipole 0.058 dex towards IFU PA 153 deg
-> WARNING for the whole mosaic.
"""
from __future__ import annotations

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

LINE = "Halpha_6562.85"


def classify(p, spread, e_spread, dipole, e_dipole):
    """(flag, note) from the sector spread and the mosaic IFU dipole (shared with lvmqa.recalc)."""
    def level(x, ex):
        if not np.isfinite(x):
            return -1
        if np.isfinite(ex) and abs(x) <= p["nsig"] * ex:
            return 0
        return 2 if abs(x) > p["major_dex"] else (1 if abs(x) > p["warn_dex"] else 0)

    ls, ld = level(spread, e_spread), level(dipole, e_dipole) if p["use_dipole"] else -1
    if ls < 0 and ld < 0:
        return QAFlag.UNDEFINED, "spectrograph sectors not measurable (too few Halpha fibers with S/N > %g)" % p["min_snr"]
    parts = []
    if ls >= 0:
        parts.append(f"spectrograph spread {spread:.3f} dex ({10 ** spread:.2f}x)")
    if ld >= 0:
        parts.append(f"mosaic IFU dipole {dipole:.3f} dex")
    lvl = max(ls, ld)
    flag = (QAFlag.GOOD, QAFlag.WARNING, QAFlag.MAJOR)[lvl]
    return flag, ("" if flag == QAFlag.GOOD else ", ".join(parts))


@register
class Gradient(Metric):
    name = "qa_gradient"
    slot = 26
    multi = True
    multi_after = ("qa_fluxmap",)
    group = GROUP_DRP
    title = "Flux-scale pattern across the IFU (spectrographs / gradient)"
    description = ("Halpha flux-scale jumps between the spectrograph sectors of the IFU (adjacent fiber pairs "
                   "across the sector boundaries vs nearby same-sector pairs) and, in mosaics, the common "
                   "IFU dipole of the tile overlaps (from qa_fluxmap)")
    requires = ("dap", "sframe_data")
    defaults = dict(min_snr=5.0, pitch=37.0, ctrl_radius=2.2, min_pairs=10, min_ctrl=2,
                    warn_dex=0.04, major_dex=0.10, nsig=3.0, use_dipole=True)
    values = (
        ValueSpec("sp1_dex", "f8", "", "Halpha flux-scale offset of spectrograph 1 (log10; sum of sp1-3 = 0)"),
        ValueSpec("sp2_dex", "f8", "", "Offset of spectrograph 2"),
        ValueSpec("sp3_dex", "f8", "", "Offset of spectrograph 3"),
        ValueSpec("e_sp_dex", "f8", "", "Typical error of the spectrograph offsets"),
        ValueSpec("spread_dex", "f8", "", "max - min of the spectrograph offsets"),
        ValueSpec("e_spread_dex", "f8", "", "Error of spread_dex"),
        ValueSpec("d12_dex", "f8", "", "Jump sp1 -> sp2 across their boundary (log10 F2/F1, physics removed)"),
        ValueSpec("d23_dex", "f8", "", "Jump sp2 -> sp3"),
        ValueSpec("d31_dex", "f8", "", "Jump sp3 -> sp1"),
        ValueSpec("closure_dex", "f8", "", "d12 + d23 + d31 (0 if consistent)"),
        ValueSpec("e_closure_dex", "f8", "", "Error of closure_dex"),
        ValueSpec("n_cross", "i8", "", "Cross-boundary fiber pairs used"),
        ValueSpec("ifu_dipole_dex", "f8", "", "Mosaic IFU dipole: flux difference of opposite overlap regions (run value)"),
        ValueSpec("e_ifu_dipole_dex", "f8", "", "Error of ifu_dipole_dex"),
        ValueSpec("ifu_dipole_pa", "f8", "", "Brighter IFU side, IFU frame (deg E of N + sci_pa)"),
        ValueSpec("mosaic_nodes", "i8", "", "Exposures in the qa_fluxmap graph (dipole fitted if member)"),
    )
    main_value = "spread_dex"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"Spread of the spectrograph Halpha flux scales (jumps across the IFU sector boundaries, "
                     f"physical gradients removed with nearby same-sector fiber pairs) and, in mosaics, the "
                     f"common IFU dipole of the tile overlaps <= {p['warn_dex']:g} dex or < {p['nsig']:g} sigma"),
            "WARNING": f"spread or IFU dipole > {p['warn_dex']:g} dex",
            "MAJOR ISSUE": f"spread or IFU dipole > {p['major_dex']:g} dex",
            "UNDEFINED": f"too few fibers with Halpha S/N > {p['min_snr']:g} along the sector boundaries, no mosaic",
        }

    def collect(self, ctx):
        """Worker: spectrograph sector offsets of this exposure (small dict)."""
        from ..multitile.sectors import sector_offsets
        from ..utils.ifu_layout import assign_fibers
        p = self.params
        if not ctx.has_dap:
            return None
        pt = ctx.dap_table("PT")
        npc = ctx.dap_np_columns([f"flux_{LINE}", f"e_flux_{LINE}"])
        ra, dec = np.asarray(pt["ra"], float), np.asarray(pt["dec"], float)
        f, ef = np.asarray(npc[f"flux_{LINE}"], float), np.asarray(npc[f"e_flux_{LINE}"], float)
        lay = assign_fibers(ra, dec, float(np.nan_to_num(ctx.row.get("sci_pa", 0.0))))
        spec = lay["spec"]
        try:   # exact spectrograph ids from the SLITMAP (the layout is only the fallback)
            sm = ctx.slitmap
            spm = dict(zip(np.asarray(sm["fiberid"]).astype(int), np.asarray(sm["spectrographid"]).astype(int)))
            spec = np.array([spm.get(int(i), 0) for i in pt["fiberid"]])
        except Exception:  # noqa: BLE001
            pass
        with np.errstate(all="ignore"):
            ok = (f > 0) & (f / ef > p["min_snr"])
            r = sector_offsets(lay["x"], lay["y"], spec, np.log10(f), ok, pitch=p["pitch"],
                               ctrl_radius=p["ctrl_radius"], min_pairs=p["min_pairs"], min_ctrl=p["min_ctrl"])
        o, eo = r["o"], r["e_o"]
        vals = dict(n_cross=int(r["n_cross"]), closure_dex=r["closure"], e_closure_dex=r["e_closure"])
        for (a, b), key in (((1, 2), "d12_dex"), ((2, 3), "d23_dex"), ((3, 1), "d31_dex")):
            if (a, b) in r["d"]:
                vals[key] = r["d"][(a, b)][0]
        if np.isfinite(o).all():
            i_max, i_min = int(np.argmax(o)), int(np.argmin(o))
            vals.update(sp1_dex=float(o[0]), sp2_dex=float(o[1]), sp3_dex=float(o[2]),
                        e_sp_dex=float(np.median(eo)), spread_dex=float(o[i_max] - o[i_min]),
                        e_spread_dex=float(np.hypot(eo[i_max], eo[i_min])))
        return vals

    def compute_multi(self, payloads, cfg, drpall, shared=None):
        p = self.params
        fm = (shared or {}).get("qa_fluxmap", {})
        out = {}
        for e, v in payloads.items():
            if v is None:
                out[e] = self.result(QAFlag.UNDEFINED, note="no DAP file")
                continue
            vals = dict(v)
            dip = fm.get(e)
            if dip is not None:
                vals["mosaic_nodes"] = int(dip["n_nodes"])
                if dip["member"]:
                    vals.update(ifu_dipole_dex=dip["dipole"], e_ifu_dipole_dex=dip["e_dipole"],
                                ifu_dipole_pa=dip["dipole_pa"])
            flag, note = classify(p, vals.get("spread_dex", np.nan), vals.get("e_spread_dex", np.nan),
                                  vals.get("ifu_dipole_dex", np.nan), vals.get("e_ifu_dipole_dex", np.nan))
            out[e] = self.result(flag, note=note, **vals)
        return out
