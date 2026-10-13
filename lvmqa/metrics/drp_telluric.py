"""
qa_telluric - residual telluric absorption in the science spectra (v1.8).

Bands (continuum windows from lvmdrp fluxCalMethod.TELLURIC_FREE_REGIONS, avoiding [ArIII]7136):

    band       range [A]     continuum windows [A]          absorber
    O2B        6860-6940     6790-6855, 7080-7130          O2 B band
    H2O72      7160-7230     7080-7130, 7400-7460          H2O
    H2O82      8120-8350     8050-8085, 8440-8480          H2O

Method per band:
1. DAP-normalised (default when the DAP output file exists): fibers with stellar S/N (RSP
   med_flux/e_med_flux) >= 10 (at most 100 brightest); ratio = (data - NP emission lines) /
   (DAP stellar model) [no_gas_spec / model_spec; v1.9: the smooth term (5 Legendre terms of a
   21-px median-filtered residual) is not used: it is ~0 in O2B / H2O72 (identical EW and rms)
   but adds curvature across H2O82 (EW -1.2...-3.4 A with it, -0.9...-1.9 A without)], divided by a
   straight line through its medians in the two continuum windows (removes a smooth model
   offset, e.g. where the DAP fit is masked: 8250-10000 A). The stellar templates contain no
   tellurics, so stellar features (TiO bands of cool stars, Paschen lines) cancel.
2. If (1) is not available or gives |EW| > ew_warn, the local interpolation is also computed
   (FLUX / straight line through the continuum windows in the brightest-continuum fibers, the
   v1.6 method) and adopted if it gives the smaller |EW| (the DAP model may be wrong). Both EWs
   are reported (ew_dap_*, ew_interp_*); ``method`` gives the adopted one (n / s).
The per-fiber normalised spectra are median-stacked; pixels at bright sky lines
(sky > 0.3 x continuum) are excluded.

    EW  = integral (1 - stack) over the band (> 0 residual absorption, < 0 over-correction)
    rms = sqrt(rms_band^2 - rms_continuum^2) of (stack - 1): residual structure that cancels in
          the EW (e.g. a wavelength / resolution mismatch of the correction at a band head)

GOOD:      |EW| < 1 A and rms < 0.04 in all measured bands
WARNING:   1 < |EW| < 3 A, or 0.04 < rms < 0.10 in at least one band
MAJOR:     |EW| > 3 A, or rms > 0.10 in at least one band
UNDEFINED: not inspected (no fiber with sufficient continuum S/N in any band)

Test sets: DRP 1.2.0 (de facto telluric-corrected through the FLUXCAL_MOD sensitivity) shows a
sharp residual at the O2 B-band head in every exposure (data/model -20% at 6868.5 A, +10% at
6881 A): EW(O2B) only +0.2 A (cancels) but rms 0.045-0.057 -> WARNING. DRP 1.3.2: rms(O2B)
0.013-0.021. H2O82 with the DAP-normalised method: EW -1.2 ... -3.4 A (data ~1% above the model
inside the band in all 1.2.0 exposures; scatter 0.6 A vs 1.5 A for the interpolation, which is
biased by TiO bands of cool stars: see dev/telluric_diagnostics.py).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

BANDS = {
    "O2B": ((6860.0, 6940.0), (6790.0, 6855.0), (7080.0, 7130.0)),
    "H2O72": ((7160.0, 7230.0), (7080.0, 7130.0), (7400.0, 7460.0)),
    "H2O82": ((8120.0, 8350.0), (8050.0, 8085.0), (8440.0, 8480.0)),
}


def _values():
    v = []
    for b in BANDS:
        v += [ValueSpec(f"ew_{b}", "f8", "Angstrom", f"EW of the {b} telluric band (adopted method)"),
              ValueSpec(f"e_ew_{b}", "f8", "Angstrom", f"Uncertainty of ew_{b} (1.25 NMAD / sqrt(n))"),
              ValueSpec(f"ew_dap_{b}", "f8", "Angstrom", f"EW of {b} with the DAP-normalised method"),
              ValueSpec(f"ew_interp_{b}", "f8", "Angstrom", f"EW of {b} with the local interpolation (if computed)"),
              ValueSpec(f"rms_{b}", "f8", "", f"Excess rms of the stacked normalised spectrum in {b} "
                                               f"(sqrt(rms_band^2 - rms_continuum^2))"),
              ValueSpec(f"n_{b}", "i8", "", f"Fibers used for {b}"),
              ValueSpec(f"fmask_{b}", "f8", "", f"Fraction of {b} pixels masked at bright sky lines")]
    v += [ValueSpec("max_abs_ew", "f8", "Angstrom", "Largest |EW| over the measured bands"),
          ValueSpec("worst_band", "U8", "", "Band with the largest |EW|"),
          ValueSpec("max_rms", "f8", "", "Largest excess rms over the measured bands"),
          ValueSpec("worst_rms_band", "U8", "", "Band with the largest excess rms"),
          ValueSpec("snr_med", "f8", "", "Median continuum S/N of the selected fibers"),
          ValueSpec("pwv_med", "f8", "mm", "Median precipitable water vapour from the DRP (PWV_MED)"),
          ValueSpec("method", "U32", "", "Adopted method per band (n = DAP-normalised, s = local interpolation)")]
    return tuple(v)


@register
class Telluric(Metric):
    name = "qa_telluric"
    slot = 15
    group = GROUP_DRP
    title = "Telluric correction"
    description = ("Residual telluric absorption: equivalent widths and residual rms of the O2 B "
                   "(6860-6940 A) and H2O (7160-7230, 8120-8350 A) bands in the bright-continuum fibers")
    requires = ("sframe_data", "dap", "dap_output")
    defaults = dict(
        ew_warn=1.0,       # |EW| [A] above -> WARNING
        ew_major=3.0,      # |EW| [A] above -> MAJOR
        rms_warn=0.04,     # excess rms of the stacked normalised spectrum -> WARNING
        rms_major=0.10,    # ... -> MAJOR
        min_snr=10.0,      # continuum S/N per pixel required in both windows (interpolation)
        min_snr_dap=10.0,  # stellar S/N (RSP med_flux / e_med_flux) for the DAP-normalised method
        max_fibers=100,    # use at most this many brightest fibers
        min_fibers=3,      # fewer -> band not measured
        sky_mask=0.3,      # mask pixels where sky > sky_mask x continuum of the selected fibers
        bands=BANDS,       # {name: ((band), (left continuum window), (right continuum window))}
        use_dap=True,      # DAP-normalised method first (falls back to interpolation without DAP output)
        dap_model="stellar",  # 'stellar' (model_spec, default since v1.9) or 'stellar+smooth' (mod_joint_spec - gas_model_NP)
        nsig=0.0,          # >0: a band counts only if also |EW| > nsig * e_ew
    )
    values = _values()
    main_value = "max_abs_ew"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": f"|EW| < {p['ew_warn']:g} A and excess rms < {p['rms_warn']:g} in all bands",
            "WARNING": (f"{p['ew_warn']:g} < |EW| < {p['ew_major']:g} A, or {p['rms_warn']:g} < rms < "
                        f"{p['rms_major']:g} in at least one band"),
            "MAJOR ISSUE": f"|EW| > {p['ew_major']:g} A or rms > {p['rms_major']:g} in at least one band",
            "UNDEFINED": "Not inspected: SFrame missing or too few bright-continuum fibers in every band",
        }

    def output_windows(self):
        from ._dapout import GAS_NP, JOINT, MODEL, NOGAS
        if not self.params["use_dap"]:
            return {}
        return {f"tell_{b}": (cl[0] - 5.0, cr[1] + 5.0, (NOGAS, JOINT, GAS_NP, MODEL))
                for b, (band, cl, cr) in self.params["bands"].items()}

    # ------------------------------------------------------------------ helpers
    def _measure(self, x, norm, skyrel, band, cl, cr, snr, sel):
        """EW, error, rms excess from per-fiber continuum-normalised spectra ``norm``."""
        p = self.params
        mb = (x >= band[0]) & (x <= band[1])
        mc = ((x >= cl[0]) & (x <= cl[1])) | ((x >= cr[0]) & (x <= cr[1]))
        clean = np.isfinite(skyrel) & (skyrel < p["sky_mask"])
        use = mb & clean
        if use.sum() < 0.5 * mb.sum():
            return None
        width = band[1] - band[0]
        stack = np.nanmedian(norm, axis=0)
        ew = float(np.nanmean(1.0 - stack[use]) * width)
        ew_fib = np.nanmean(1.0 - norm[:, use], axis=1) * width
        ew_fib = ew_fib[np.isfinite(ew_fib)]
        e = 1.2533 * 1.4826 * float(np.median(np.abs(ew_fib - np.median(ew_fib)))) / np.sqrt(max(ew_fib.size, 1))
        rb = np.sqrt(np.nanmean((stack[use] - 1.0) ** 2))
        rc = np.sqrt(np.nanmean((stack[mc & clean] - 1.0) ** 2)) if (mc & clean).sum() > 5 else 0.0
        rms = float(np.sqrt(max(rb ** 2 - rc ** 2, 0.0)))
        return dict(ew=ew, e=e, n=int(len(sel)), snr=snr, fmask=float(1.0 - use.sum() / mb.sum()), rms=rms,
                    x=x, stack=stack)

    @staticmethod
    def _line(x, y, cl, cr):
        ml, mr = (x >= cl[0]) & (x <= cl[1]), (x >= cr[0]) & (x <= cr[1])
        yl, yr = np.nanmedian(y[:, ml], axis=1), np.nanmedian(y[:, mr], axis=1)
        xl, xr = np.median(x[ml]), np.median(x[mr])
        return yl[:, None] + (yr - yl)[:, None] * (x - xl)[None, :] / (xr - xl), yl, yr

    def _band_dap(self, ctx, b, band, cl, cr):
        """DAP-normalised method: (data - NP lines) / (stellar + smooth DAP model), divided by a
        straight line through its medians in the two continuum windows."""
        from ._dapout import GAS_NP, JOINT, MODEL, NOGAS
        p = self.params
        rsp = ctx.dap_table("RSP")
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.asarray(rsp["med_flux"], float) / np.asarray(rsp["e_med_flux"], float)
        sel = np.where(np.isfinite(snr) & (snr >= p["min_snr_dap"]))[0]
        if sel.size < p["min_fibers"]:
            return None
        sel = sel[np.argsort(snr[sel])[::-1][:p["max_fibers"]]]
        wave, planes = ctx.dap_output(f"tell_{b}")
        _, sky = ctx.dap_sframe_window("SKY", wave[0], wave[-1])
        sky = sky[sel][:, :wave.size]
        data = planes[NOGAS][sel].astype(float)
        if p["dap_model"] == "stellar":
            cont = planes[MODEL][sel].astype(float)
        else:
            cont = planes[JOINT][sel].astype(float) - planes[GAS_NP][sel]
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            good_c = np.nanmedian(cont, axis=1) > 0
            sel, data, cont, sky = sel[good_c], data[good_c], cont[good_c], sky[good_c]
            if sel.size < p["min_fibers"]:
                return None
            ratio = data / cont
            lin, _, _ = self._line(wave, ratio, cl, cr)
            norm = ratio / lin
            skyrel = np.nanmedian(sky / cont, axis=0)
            return self._measure(wave, norm, skyrel, band, cl, cr, snr[sel], sel)

    def _band(self, ctx, rows, band, cl, cr):
        """Local interpolation (v1.6 method): FLUX / straight line through the continuum windows."""
        p = self.params
        x, f = ctx.sframe_window("FLUX", cl[0], cr[1], rows=rows)
        _, iv = ctx.sframe_window("IVAR", cl[0], cr[1], rows=rows)
        _, sk = ctx.sframe_window("SKY", cl[0], cr[1], rows=rows)
        good = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
        f = np.where(good, f, np.nan)
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            err = np.where(good, 1.0 / np.sqrt(np.where(good, iv, 1.0)), np.nan)
            ml, mr = (x >= cl[0]) & (x <= cl[1]), (x >= cr[0]) & (x <= cr[1])
            fl, fr = np.nanmedian(f[:, ml], axis=1), np.nanmedian(f[:, mr], axis=1)
            snr = np.minimum(fl / np.nanmedian(err[:, ml], axis=1), fr / np.nanmedian(err[:, mr], axis=1))
            sel = np.where(np.isfinite(snr) & (snr >= p["min_snr"]) & (fl > 0) & (fr > 0))[0]
            if sel.size < p["min_fibers"]:
                return None
            sel = sel[np.argsort(snr[sel])[::-1][:p["max_fibers"]]]
            cont, _, _ = self._line(x, f[sel], cl, cr)
            norm = f[sel] / cont
            skyrel = np.nanmedian(sk[sel] / cont, axis=0)
            return self._measure(x, norm, skyrel, band, cl, cr, snr[sel], sel)

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        rows = ctx.good_sci_index
        pwv = ctx.header.get("PWV_MED")
        vals = {"pwv_med": float(pwv) if pwv is not None else float("nan")}
        ews, rmss, snrs, sig = {}, {}, [], {}
        have_out = p["use_dap"] and ctx.has_dap and ctx.has_dap_output
        used = []
        for b, (band, cl, cr) in p["bands"].items():
            rd = self._band_dap(ctx, b, band, cl, cr) if have_out else None
            r, m = rd, "n"
            if rd is not None:
                vals[f"ew_dap_{b}"] = rd["ew"]
            if rd is None or abs(rd["ew"]) > p["ew_warn"]:
                # the DAP fit may be wrong (e.g. masked fit range): check the local interpolation
                # and adopt it if it gives the smaller |EW|
                rs = self._band(ctx, rows, band, cl, cr)
                if rs is not None:
                    vals[f"ew_interp_{b}"] = rs["ew"]
                    if rd is None or abs(rs["ew"]) < abs(rd["ew"]):
                        r, m = rs, "s"
            used.append(f"{b}:{m}" if r is not None else f"{b}:-")
            if r is None:
                vals[f"n_{b}"] = 0
                continue
            ews[b] = r["ew"]
            rmss[b] = r["rms"]
            sig[b] = abs(r["ew"]) > p["nsig"] * r["e"]
            snrs.append(r["snr"])
            vals.update({f"ew_{b}": r["ew"], f"e_ew_{b}": r["e"], f"n_{b}": r["n"], f"fmask_{b}": r["fmask"],
                         f"rms_{b}": r["rms"]})
        vals["method"] = " ".join(used)
        if not ews:
            return self.result(QAFlag.UNDEFINED, note="not inspected: no bright-continuum fibers", **vals)
        wb = max(ews, key=lambda k: abs(ews[k]))
        wr = max(rmss, key=rmss.get)
        vals.update(max_abs_ew=abs(ews[wb]), worst_band=wb, max_rms=rmss[wr], worst_rms_band=wr,
                    snr_med=float(np.median(np.concatenate(snrs))))
        flagged = {k: abs(v) for k, v in ews.items() if sig[k]}
        flag, notes = QAFlag.GOOD, []
        if flagged:
            fb = max(flagged, key=flagged.get)
            if flagged[fb] > p["ew_major"]:
                flag = QAFlag.MAJOR
            elif flagged[fb] > p["ew_warn"]:
                flag = QAFlag.WARNING
            if flagged[fb] > p["ew_warn"]:
                notes.append(f"EW({fb}) = {ews[fb]:+.2f} A")
        if rmss[wr] > p["rms_major"]:
            flag = QAFlag.MAJOR
            notes.append(f"rms({wr}) = {rmss[wr]:.3f}")
        elif rmss[wr] > p["rms_warn"]:
            flag = max(flag, QAFlag.WARNING, key=lambda f: {QAFlag.GOOD: 0, QAFlag.WARNING: 1, QAFlag.MAJOR: 2}[f])
            notes.append(f"rms({wr}) = {rmss[wr]:.3f}")
        return self.result(flag, note="; ".join(notes), **vals)
