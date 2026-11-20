"""
qa_absorption - prototype: how well the DAP stellar-continuum model reproduces key stellar
absorption features, in fibers with a significant stellar component (DAP output file).

Fibers: stellar S/N = RSP med_flux / e_med_flux >= ``min_snr``.
For each feature a Lick-style index (pseudo-continuum = straight line through the means of
the blue and red side bands, EW in A) is measured on

    data  = no_gas_spec              (observed - NP emission-line model)
    model = mod_joint_spec - gas_model_NP   (stellar model + smooth term)

with the bands shifted by the stellar redshift of the fiber (RSP 'redshift'); pixels at
bright sky lines (SFrame SKY > sky_mask x stellar model, median over the fibers) are
excluded from both; bands with < 50% usable pixels are not measured.
dI = I(data) - I(model) > 0 means the observed absorption is deeper than in the model.

Features (Worthey+94 / Worthey & Ottaviani 97 Lick bands; CaT bands after Cenarro+01):
    HdA, HgA, Hb, Mgb, Fe5270, Fe5335      inside the DAP fit range
    CaT1 (8498), CaT2 (8542), CaT3 (8662)  inside w_range but in the masked range 8250-10000 A
                                           of mask_bands_LVM.txt (unchanged since the first
                                           DAP version 2023-11-06, while w_range was later
                                           extended to 8700 A): the model CaT is a prediction
                                           from the bluer spectrum -> dI tests the stellar
                                           population fit, not only the fit quality. Flagged.
    NaD                                    masked (5500-6000 A) and dominated by ISM absorption
                                           -> reported only
Per feature: median dI over the fibers and its uncertainty (1.25 NMAD / sqrt(n)).
'poor' if |median dI| > tol_warn (0.5 A) and > 3 sigma, 'bad' if > tol_major (1 A).
GOOD:      no flagged feature bad and <= 25% poor
WARNING:   >= 1 bad, or > 25% of the flagged features poor
MAJOR:     >= 2 and >= 50% of the flagged features bad
UNDEFINED: < min_fibers fibers with significant stellar continuum
Prototype thresholds (typical |dI| in the DRP 1.2.0 test set: 0.1-0.9 A).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from ._dapout import GAS_NP, JOINT, NOGAS

# name: (blue band, feature band, red band) [A, air]
FEATURES = {
    "HdA": ((4041.60, 4079.75), (4083.50, 4122.25), (4128.50, 4161.00)),
    "HgA": ((4283.50, 4319.75), (4319.75, 4363.50), (4367.25, 4419.75)),
    "Hb": ((4827.875, 4847.875), (4847.875, 4876.625), (4876.625, 4891.625)),
    "Mgb": ((5142.625, 5161.375), (5160.125, 5192.625), (5191.375, 5206.375)),
    "Fe5270": ((5233.150, 5248.150), (5245.650, 5285.650), (5285.650, 5318.150)),
    "Fe5335": ((5304.625, 5315.875), (5312.125, 5352.125), (5353.375, 5363.375)),
    "NaD": ((5862.375, 5877.375), (5876.875, 5909.375), (5922.125, 5948.125)),
    "CaT1": ((8474.0, 8484.0), (8484.0, 8513.0), (8563.0, 8577.0)),
    "CaT2": ((8474.0, 8484.0), (8522.0, 8562.0), (8563.0, 8577.0)),
    "CaT3": ((8619.0, 8642.0), (8642.0, 8682.0), (8700.0, 8725.0)),
}
FLAG_FEATURES = ("HdA", "HgA", "Hb", "Mgb", "Fe5270", "Fe5335", "CaT1", "CaT2", "CaT3")
MARGIN = 15.0


def _values():
    v = [ValueSpec("n_fibers", "i8", "", "Fibers with stellar S/N >= min_snr"),
         ValueSpec("n_poor", "i8", "", "Flag features 'poor' (or bad)"),
         ValueSpec("n_bad", "i8", "", "Flag features 'bad'"),
         ValueSpec("worst_feature", "U8", "", "Flag feature with the largest |median dI|"),
         ValueSpec("max_abs_dI", "f8", "Angstrom", "Largest |median dI| over the flag features")]
    for k in FEATURES:
        v += [ValueSpec(f"dI_{k}", "f8", "Angstrom", f"{k}: median I(data) - I(model)"),
              ValueSpec(f"e_dI_{k}", "f8", "Angstrom", f"{k}: uncertainty of the median dI"),
              ValueSpec(f"I_{k}", "f8", "Angstrom", f"{k}: median index of the data"),
              ValueSpec(f"status_{k}", "i8", "", f"{k}: -1 n/a, 0 good, 1 poor, 2 bad")]
    return tuple(v)


def _index(w, f, bands, z, keep, min_frac=0.5):
    """Lick-style index (A) per fiber; f (n, npix); bands shifted by (1+z) per fiber."""
    (b0, b1), (c0, c1), (r0, r1) = bands
    zz = (1.0 + z)[:, None]
    W = w[None, :]
    mb = (W >= b0 * zz) & (W <= b1 * zz) & keep
    mr = (W >= r0 * zz) & (W <= r1 * zz) & keep
    mc = (W >= c0 * zz) & (W <= c1 * zz) & keep
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        fb = np.nanmean(np.where(mb, f, np.nan), axis=1)
        fr = np.nanmean(np.where(mr, f, np.nan), axis=1)
        wb = 0.5 * (b0 + b1) * (1 + z)
        wr = 0.5 * (r0 + r1) * (1 + z)
        cont = fb[:, None] + (fr - fb)[:, None] * (W - wb[:, None]) / (wr - wb)[:, None]
        depth = np.where(mc, 1.0 - f / cont, np.nan)
        frac = mc.sum(axis=1) / np.maximum(((W >= c0 * zz) & (W <= c1 * zz)).sum(axis=1), 1)
        idx = np.nanmean(depth, axis=1) * (c1 - c0) * (1 + z)
    idx[frac < 0.7] = np.nan
    return idx


@register
class Absorption(Metric):
    name = "qa_absorption"
    slot = 18
    group = GROUP_DAP
    title = "DAP stellar absorption features"
    description = ("Prototype: Lick-style indices of key stellar absorption features in the data "
                   "(emission lines removed) vs. the DAP stellar model, in fibers with a "
                   "significant stellar continuum (DAP output file)")
    requires = ("dap", "dap_output", "sframe_data")
    defaults = dict(
        features=tuple(FEATURES),
        flag_features=FLAG_FEATURES,
        min_snr=10.0,        # stellar S/N (RSP med_flux / e_med_flux)
        min_fibers=5,
        tol_warn=0.5,        # [A] |median dI| above (and > nsig sigma) -> poor
        tol_major=1.0,       # [A] -> bad
        nsig=3.0,
        sky_mask=0.5,        # mask pixels where SKY > sky_mask x stellar model (median over fibers)
        frac_poor_warn=0.25, frac_bad_major=0.5,
    )
    values = _values()
    main_value = "max_abs_dI"

    def output_windows(self):
        return {f"abs_{k}": (FEATURES[k][0][0] - MARGIN, FEATURES[k][2][1] + MARGIN, (NOGAS, JOINT, GAS_NP))
                for k in self.params["features"]}

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        f = ", ".join(p["flag_features"])
        return {
            "GOOD": (f"Index differences data - DAP stellar model |median dI| <= {p['tol_warn']:g} A "
                     f"(or not significant) for most of {f}"),
            "WARNING": (f">= 1 feature with |median dI| > {p['tol_major']:g} A, or > {p['frac_poor_warn']:.0%} "
                        f"of the features with > {p['tol_warn']:g} A ({p['nsig']:g} sigma)"),
            "MAJOR ISSUE": f">= 2 and >= {p['frac_bad_major']:.0%} of the features with |median dI| > {p['tol_major']:g} A",
            "UNDEFINED": f"No DAP / DAP output, or < {p['min_fibers']} fibers with stellar S/N >= {p['min_snr']:g}",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")
        if not ctx.has_dap_output:
            return self.result(QAFlag.UNDEFINED, note="DAP output file missing")
        rsp = ctx.dap_table("RSP")
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.asarray(rsp["med_flux"], float) / np.asarray(rsp["e_med_flux"], float)
        z = np.nan_to_num(np.asarray(rsp["redshift"], float))
        sel = np.where(np.isfinite(snr) & (snr >= p["min_snr"]))[0]
        vals = {"n_fibers": int(sel.size)}
        if sel.size < p["min_fibers"]:
            return self.result(QAFlag.UNDEFINED, note=f"only {sel.size} fibers with stellar S/N>={p['min_snr']:g}",
                               **vals)
        status, dabs = {}, {}
        for k in p["features"]:
            wave, planes = ctx.dap_output(f"abs_{k}")
            from ._dapout import sframe_on
            sky = sframe_on(ctx, "SKY", wave)
            data = planes[NOGAS][sel].astype(float)
            model = planes[JOINT][sel].astype(float) - planes[GAS_NP][sel]
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                skyrel = np.nanmedian(sky[sel] / model, axis=0)
                keep = ~(skyrel > p["sky_mask"])
                # OH-dense regions (CaT): if a band would lose > half of its pixels, mask only
                # the brightest sky lines
                zz = np.nanmedian(z[sel])
                for lo, hi in FEATURES[k]:
                    mb = (wave >= lo * (1 + zz)) & (wave <= hi * (1 + zz))
                    if mb.any() and keep[mb].mean() < 0.5:
                        keep = ~(skyrel > 4 * p["sky_mask"])
                        break
            Id = _index(wave, data, FEATURES[k], z[sel], keep[None, :])
            Im = _index(wave, model, FEATURES[k], z[sel], keep[None, :])
            d = Id - Im
            ok = np.isfinite(d)
            status[k] = -1
            if ok.sum() < p["min_fibers"]:
                vals[f"status_{k}"] = -1
                continue
            med = float(np.median(d[ok]))
            e = 1.2533 * 1.4826 * float(np.median(np.abs(d[ok] - med))) / np.sqrt(ok.sum())
            st = 0
            if abs(med) > p["nsig"] * e:
                st = 2 if abs(med) > p["tol_major"] else 1 if abs(med) > p["tol_warn"] else 0
            status[k] = st
            vals.update({f"dI_{k}": med, f"e_dI_{k}": e, f"I_{k}": float(np.median(Id[ok])),
                         f"status_{k}": st})
            dabs[k] = abs(med)
        fl = [k for k in p["flag_features"] if status.get(k, -1) >= 0]
        n_poor = sum(status[k] >= 1 for k in fl)
        n_bad = sum(status[k] == 2 for k in fl)
        vals.update(n_poor=n_poor, n_bad=n_bad)
        if not fl:
            return self.result(QAFlag.UNDEFINED, note="no feature measurable", **vals)
        w = max(fl, key=lambda k: dabs[k])
        vals.update(worst_feature=w, max_abs_dI=dabs[w])
        note = ", ".join(f"{k} {vals[f'dI_{k}']:+.2f}A" for k in fl if status[k] > 0)
        if n_bad >= 2 and n_bad / len(fl) >= p["frac_bad_major"]:
            return self.result(QAFlag.MAJOR, note=note, **vals)
        if n_bad >= 1 or n_poor / len(fl) > p["frac_poor_warn"]:
            return self.result(QAFlag.WARNING, note=note, **vals)
        return self.result(QAFlag.GOOD, note=note, **vals)
