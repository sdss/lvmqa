"""
qa_flatfield - fiber flat-field / relative throughput consistency between spectrographs,
measured on sky lines.

Sky-line fluxes must be identical in all science fibers (the sky is uniform across the
field), so after flat-fielding they trace the relative throughput calibration. For each
bright sky line the flux is measured per fiber in FLUX + SKY (i.e. before sky
subtraction): sum over +-``half_width`` A minus a local continuum (median of side bands
``cont_inner``..``cont_outer`` A from the line).

Modes (parameter ``mode``):

relative (default)
    d_s = median_s / median(median_1, median_2, median_3) - 1 per line and spectrograph;
    D[s, channel] = median of d_s over the lines of that channel. A spectrograph/channel is
    WARNING if |D| > warn_rel (3%) and MAJOR if |D| > major_rel (6%), provided |D| exceeds
    nsig x its error (error of the medians / line-to-line scatter). Exposure = worst.
scatter
    DR20 rules with sigma = median per-fiber NMAD of the good science fibers: a line is MAJOR
    if one spectrograph deviates by > 2 sigma or two by > 1 sigma, WARNING if one by > 1 sigma;
    exposure MAJOR if >= 30% of lines MAJOR, WARNING if >= 30% WARNING or MAJOR.
dr20
    as 'scatter', but medians and sigma (np.std) over ALL fibers of each spectrograph,
    including SkyE/SkyW/standard-star and bad fibers, as in the DR20 Knox tables.

Why not the DR20 rule by default: its sigma is the fiber-to-fiber scatter, not the error
of the spectrograph median, so the test becomes stricter as the sky lines get brighter.
In DR20 the std over all fibers (incl. sky-telescope fibers that see a different sky and
outliers) inflated sigma by 2-4x, which made ~5-9% throughput offsets pass; with a robust
scatter of science fibers only (~1.5-3%) offsets of 2% are already flagged.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

# DR20 sky-line list (rest air wavelengths, A) minus 9552 (blended with Paschen-epsilon 9546)
SKY_LINES = {
    "sky5577": (5577.34, "b"), "sky6533": (6533.01, "r"), "sky6923": (6923.15, "r"),
    "sky6939": (6939.55, "r"), "sky7392": (7392.18, "r"), "sky7914": (7913.66, "z"),
    "sky8344": (8344.57, "z"), "sky8399": (8399.23, "z"), "sky8827": (8827.07, "z"),
    "sky8988": (8988.42, "z"), "sky9719": (9719.78, "z"),
}
SPECS = (1, 2, 3)
CHANNELS = ("b", "r", "z")
_SEV = {QAFlag.GOOD: 0, QAFlag.WARNING: 1, QAFlag.MAJOR: 2}


def _nmad(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x, axis=axis, keepdims=True)
        return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def _values():
    v = [ValueSpec("mode", "U8", "", "Flagging mode used"),
         ValueSpec("n_lines", "i8", "", "Sky lines evaluated"),
         ValueSpec("max_rel_offset", "f8", "percent", "Largest |D| = |median over lines of median_s/ref - 1| over spectrographs and channels"),
         ValueSpec("worst_sp_ch", "U8", "", "Spectrograph-channel with the largest |D|"),
         ValueSpec("frac_lines_warn", "f8", "", "Fraction of lines flagged WARNING or MAJOR by the scatter-based (DR20) line rule"),
         ValueSpec("frac_lines_bad", "f8", "", "Fraction of lines flagged MAJOR by the scatter-based (DR20) line rule"),
         ValueSpec("max_offset", "f8", "", "Largest spectrograph offset in units of the fiber scatter"),
         ValueSpec("fiber_scatter", "f8", "percent", "Median relative per-fiber scatter of sky-line fluxes")]
    for s in SPECS:
        for ch in CHANNELS:
            v.append(ValueSpec(f"off_sp{s}_{ch}", "f8", "percent",
                               f"D for spectrograph {s}, {ch} channel: median over lines of "
                               f"median flux / median of the three spectrographs - 1"))
    v += [ValueSpec("n_blocks", "i8", "", "Fiber blocks evaluated (diagnostic)"),
          ValueSpec("max_block_offset", "f8", "", "Largest block offset (median over lines, units of fiber scatter)"),
          ValueSpec("max_block_rel", "f8", "percent", "Largest block offset relative to the reference (median over lines)"),
          ValueSpec("n_blocks_gt1", "i8", "", "Blocks with median offset > 1 fiber scatter (diagnostic)"),
          ValueSpec("worst_block", "U4", "", "Block with the largest offset")]
    return tuple(v)


@register
class FlatField(Metric):
    name = "qa_flatfield"
    slot = 8
    group = GROUP_DRP
    title = "Flat-field consistency (sky lines)"
    description = ("Flat-field quality: consistency of bright sky-line fluxes (relative "
                   "throughput) between spectrographs, per channel")
    requires = ("sframe_data",)
    defaults = dict(
        mode="relative",         # 'relative' (default), 'scatter' or 'dr20'
        lines=tuple(SKY_LINES),
        half_width=3.0,          # line integration half width [A]
        cont_inner=6.0,          # side bands from +-cont_inner to +-cont_outer [A]
        cont_outer=12.0,
        warn_rel=0.03,           # relative mode: |D| above -> WARNING
        major_rel=0.06,          # relative mode: |D| above -> MAJOR
        nsig=3.0,                # relative mode: |D| must also exceed nsig x error
        warn_sigma=1.0, bad_sigma=2.0,   # scatter / dr20 modes
        frac_lines=0.30,                 # scatter / dr20 modes
        min_fibers=20,           # per spectrograph / block
    )
    values = _values()
    main_value = "max_rel_offset"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        if p["mode"] == "relative":
            return {
                "GOOD": (f"Sky-line flux of every spectrograph within {p['warn_rel']:.0%} of the "
                         f"reference in all channels (or not significant)"),
                "WARNING": (f"Median relative sky-line flux offset of a spectrograph in a channel "
                            f"> {p['warn_rel']:.0%} (and > {p['nsig']:g} sigma)"),
                "MAJOR ISSUE": (f"Median relative sky-line flux offset of a spectrograph in a "
                                f"channel > {p['major_rel']:.0%} (and > {p['nsig']:g} sigma)"),
                "UNDEFINED": "SFrame missing or too few good science fibers",
            }
        f = f"{p['frac_lines']:.0%}"
        return {
            "GOOD": f"< {f} of sky lines with a spectrograph offset > {p['warn_sigma']:g} sigma",
            "WARNING": (f">= {f} of sky lines where the median of one spectrograph deviates by > "
                        f"{p['warn_sigma']:g} sigma (sigma = per-fiber scatter, mode {p['mode']})"),
            "MAJOR ISSUE": (f">= {f} of sky lines where one spectrograph deviates by > "
                            f"{p['bad_sigma']:g} sigma or two by > {p['warn_sigma']:g} sigma"),
            "UNDEFINED": "SFrame missing or too few good science fibers",
        }

    def _line_fluxes(self, ctx, rows, wl):
        p = self.params
        w0, w1 = wl - p["cont_outer"] - 1, wl + p["cont_outer"] + 1
        wave, f = ctx.sframe_window("FLUX", w0, w1, rows=rows)
        _, sk = ctx.sframe_window("SKY", w0, w1, rows=rows)
        y = f + sk
        d = wave - wl
        inl = np.abs(d) <= p["half_width"]
        cont = (np.abs(d) >= p["cont_inner"]) & (np.abs(d) <= p["cont_outer"])
        dw = float(np.median(np.diff(wave)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(y[:, cont], axis=1)
            yl = y[:, inl]
            nbad = np.sum(~np.isfinite(yl), axis=1)
            flux = np.nansum(yl - c[:, None], axis=1) * dw
        flux[(nbad > 0) | ~np.isfinite(c)] = np.nan
        return flux

    def compute(self, ctx):
        p = self.params
        mode = p["mode"]
        if mode not in ("relative", "scatter", "dr20"):
            raise ValueError(f"unknown mode {mode!r}")
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        sm = ctx.slitmap
        rows = np.arange(len(sm)) if mode == "dr20" else ctx.good_sci_index
        specid = np.asarray(sm["spectrographid"])[rows]
        blocks = np.char.strip(np.asarray(sm["blockid"]).astype(str))[rows]
        sci = np.isin(rows, ctx.good_sci_index)
        wmin, wmax = ctx.wave[0], ctx.wave[-1]
        spread = np.std if mode == "dr20" else _nmad

        line_flags, max_off, scat = [], 0.0, []
        rel = {(s, ch): [] for s in SPECS for ch in CHANNELS}
        rel_err = {(s, ch): [] for s in SPECS for ch in CHANNELS}
        block_dev, block_rel = {}, {}
        for name in p["lines"]:
            wl, ch = SKY_LINES[name]
            if wl - p["cont_outer"] < wmin or wl + p["cont_outer"] > wmax:
                continue
            flux = self._line_fluxes(ctx, rows, wl)
            meds, sigs, errs = {}, {}, {}
            for s in SPECS:
                v = flux[(specid == s) & np.isfinite(flux)]
                if v.size >= p["min_fibers"]:
                    meds[s] = float(np.median(v))
                    sigs[s] = float(spread(v))
                    errs[s] = 1.2533 * float(_nmad(v)) / np.sqrt(v.size)
            if len(meds) < 3:
                continue
            sigma = float(np.median(list(sigs.values())))
            ref = float(np.median(list(meds.values())))
            if not (sigma > 0 and np.isfinite(sigma)) or ref <= 0:
                continue
            scat.append(sigma / ref)
            offs = {s: abs(meds[s] - ref) / sigma for s in SPECS}
            n1 = sum(o > p["warn_sigma"] for o in offs.values())
            n2 = sum(o > p["bad_sigma"] for o in offs.values())
            line_flags.append(QAFlag.MAJOR if (n2 >= 1 or n1 >= 2) else
                              QAFlag.WARNING if n1 == 1 else QAFlag.GOOD)
            max_off = max(max_off, max(offs.values()))
            for s in SPECS:
                rel[(s, ch)].append(meds[s] / ref - 1.0)
                rel_err[(s, ch)].append(errs[s] / ref)
            for b in np.unique(blocks[sci]):
                v = flux[(blocks == b) & sci & np.isfinite(flux)]
                if v.size >= p["min_fibers"] // 2:
                    block_dev.setdefault(b, []).append(abs(float(np.median(v)) - ref) / sigma)
                    block_rel.setdefault(b, []).append(float(np.median(v)) / ref - 1.0)

        vals = {"mode": mode}
        D, Derr = {}, {}
        for key, v in rel.items():
            if not v:
                continue
            v = np.asarray(v)
            D[key] = float(np.median(v))
            e_med = float(np.median(rel_err[key]))
            e_scatter = 1.2533 * float(_nmad(v)) / np.sqrt(v.size) if v.size >= 3 else e_med
            Derr[key] = max(e_med, e_scatter)
            vals[f"off_sp{key[0]}_{key[1]}"] = 100.0 * D[key]
        if block_dev:
            bmed = {b: float(np.median(v)) for b, v in block_dev.items()}
            wb = max(bmed, key=bmed.get)
            vals.update(n_blocks=len(bmed), max_block_offset=bmed[wb],
                        max_block_rel=100.0 * float(np.median(block_rel[wb])),
                        n_blocks_gt1=int(sum(v > 1 for v in bmed.values())), worst_block=wb)
        n = len(line_flags)
        vals["n_lines"] = n
        if n == 0 or not D:
            return self.result(QAFlag.UNDEFINED, note="no sky line evaluable", **vals)
        fb = sum(f == QAFlag.MAJOR for f in line_flags) / n
        fw = sum(f != QAFlag.GOOD for f in line_flags) / n
        worst = max(D, key=lambda k: abs(D[k]))
        vals.update(frac_lines_warn=fw, frac_lines_bad=fb, max_offset=max_off,
                    fiber_scatter=100.0 * float(np.median(scat)),
                    max_rel_offset=100.0 * abs(D[worst]), worst_sp_ch=f"sp{worst[0]}-{worst[1]}")

        if mode == "relative":
            flag = QAFlag.GOOD
            for key, d in D.items():
                if abs(d) <= p["nsig"] * Derr[key]:
                    continue
                if abs(d) > p["major_rel"]:
                    flag = QAFlag.MAJOR
                elif abs(d) > p["warn_rel"] and flag != QAFlag.MAJOR:
                    flag = QAFlag.WARNING
            note = "" if flag == QAFlag.GOOD else \
                f"sp{worst[0]} {worst[1]}: {100 * D[worst]:+.1f}% sky-line flux offset"
            return self.result(flag, note=note, **vals)

        if fb >= p["frac_lines"]:
            return self.result(QAFlag.MAJOR, note=f"{fb:.0%} of sky lines MAJOR", **vals)
        if fw >= p["frac_lines"]:
            return self.result(QAFlag.WARNING, note=f"{fw:.0%} of sky lines deviate", **vals)
        return self.result(QAFlag.GOOD, **vals)
