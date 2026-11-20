"""
qa_skycont_sub - sky continuum subtraction quality from the continuum floor of the faintest
science fibers (over- and under-subtraction).

A sky-continuum subtraction error is an additive offset common to (nearly) all fibers,
while the astrophysical continuum differs from fiber to fiber and cannot be negative. The
fibers with the faintest continuum therefore measure the residual sky.

Per channel window (b 5400-5500, r 6000-6100 A as in DR20; z: eight inter-OH gaps in
8400-8740 A, 72 A in total, instead of the DR20 8800-8900 A which contains the OH lines 8827,
8836, 8867, 8886 A). The z gaps are the stretches >= 6 A where the worst-case median sky of 8
exposures (DRP 1.2.0 and 1.3.2) stays within 6% of its running 10th-percentile envelope
(dev/find_skycont_window.py logic); gaps at stellar CaT / Paschen and nebular lines ([ClII]8579,
[CI]8727, P12-P16) are not used, nor the H2O band 8120-8350 A. Longer clean windows do not exist
in z (the OH 6-2 / 7-3 bands and the O2 (0-1) band fill 7700-9300 A).
and good science fiber the per-pixel continuum S/N (DR20 definition) is

    snr = median(FLUX) / median(error)

1. faint fibers: the ``faint_frac`` (20%) of the fibers with the faintest continuum. They
   are ranked by the continuum S/N in the *other two* channels: ranking in the measured
   channel itself would preferentially pick negative noise fluctuations and bias the floor
   low (by ~ -1.3 x the noise of snr for the lowest 20%).
2. floor = median snr of these fibers (per channel); uncertainty 1.25 NMAD / sqrt(n).
3. the same per spectrograph (faintest 20% of its own fibers), and the spread of the three
   spectrograph floors per channel.
4. DR20-like criteria on all floors (combined and per spectrograph):
     GOOD     -1 <= floor <= +1 everywhere, spectrograph spread <= 1.0
     WARNING  -3 <= floor < -1 (over-subtraction), or floor > +1, or spread > 1.0
     MAJOR    floor < -3 (over-subtraction), or spread > 3 (a spectrograph-specific residual
              cannot be astrophysical)
   A positive floor (possible under-subtraction) raises at most a WARNING, however large:
   it can also be real, spatially uniform astrophysical light (diffuse starlight, a galaxy
   filling the field, bright nebular continuum), whereas a negative floor is unphysical.
The SKY extension (the subtracted sky model) is not used.
Spectrograph spreads: the per-spectrograph floor has an error of ~0.035 (117 faint fibers), so
the typical spreads of 0.1-0.6 (DRP 1.2.0 and 1.3.2) are significant (5-20 sigma): real
spectrograph-dependent sky residuals, but small w.r.t. the noise per pixel. The WARNING threshold
(1.0, as for the floor itself) flags spreads that equal the per-pixel noise (v1.8; 1.5 before).

Diagnostics: per-spectrograph floors (a spectrograph-specific problem is diluted in the
combined floor), the median snr of all fibers and the fractions of fibers with snr < -1 / -3
(the previous version of this metric).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DRP, Metric, ValueSpec, register

# channel: (w0, w1) or tuple of (w0, w1) sub-windows
WINDOWS = {"b": (5400.0, 5500.0), "r": (6000.0, 6100.0), "z": ((8403.5, 8412.5), (8436.5, 8443.5), (8477.0, 8484.5), (8508.0, 8519.0),
                  (8522.5, 8530.0), (8559.5, 8570.5), (8717.5, 8724.0), (8729.0, 8735.5))}


def ranges(win):
    """Sub-windows of a WINDOWS entry as a tuple of (w0, w1)."""
    return (tuple(win),) if np.isscalar(win[0]) else tuple(tuple(r) for r in win)


def span(win):
    r = ranges(win)
    return min(a for a, _ in r), max(b for _, b in r)


def in_window(w, win):
    w = np.asarray(w)
    return np.any([(w >= a) & (w <= b) for a, b in ranges(win)], axis=0)


def _values():
    v = [ValueSpec("n_fibers", "i8", "", "Good science fibers used"),
         ValueSpec("n_faint", "i8", "", "Faint fibers used for the floor (per channel)")]
    for ch in WINDOWS:
        v += [ValueSpec(f"floor_{ch}", "f8", "", f"Continuum floor: median S/N per pixel of the faintest fibers ({ch})"),
              ValueSpec(f"e_floor_{ch}", "f8", "", f"Uncertainty of floor_{ch}"),
              ValueSpec(f"floor_sp1_{ch}", "f8", "", f"Floor of spectrograph 1 (its own faintest fibers, {ch})"),
              ValueSpec(f"floor_sp2_{ch}", "f8", "", f"Floor of spectrograph 2 (its own faintest fibers, {ch})"),
              ValueSpec(f"floor_sp3_{ch}", "f8", "", f"Floor of spectrograph 3 (its own faintest fibers, {ch})"),
              ValueSpec(f"snr_med_{ch}", "f8", "", f"Median continuum S/N of all science fibers ({ch})"),
              ValueSpec(f"frac_neg1_{ch}", "f8", "", f"Fraction of science fibers with continuum S/N < -1 ({ch})"),
              ValueSpec(f"frac_neg3_{ch}", "f8", "", f"Fraction of science fibers with continuum S/N < -3 ({ch})")]
    v += [ValueSpec("floor_min", "f8", "", "Lowest floor over channels and spectrographs"),
          ValueSpec("floor_max", "f8", "", "Highest floor over channels and spectrographs"),
          ValueSpec("spec_spread", "f8", "", "Largest difference between spectrograph floors (worst channel)")]
    return tuple(v)


@register
class SkyContSub(Metric):
    name = "qa_skycont_sub"
    slot = 7
    group = GROUP_DRP
    title = "Sky continuum subtraction"
    description = ("Sky continuum subtraction quality: continuum S/N floor of the faintest science "
                   "fibers per channel (negative: over-subtraction; positive: possible "
                   "under-subtraction, at most WARNING)")
    requires = ("sframe_data",)
    defaults = dict(
        faint_frac=0.2,     # fraction of the faintest fibers defining the floor
        snr_warn=1.0,       # |floor| above -> WARNING
        snr_major=3.0,      # floor below -snr_major -> MAJOR (over-subtraction only)
        spread_warn=1.0,    # floors of the spectrographs differ by more -> WARNING (1.5 before v1.8)
        spread_major=3.0,   # ... -> MAJOR (spectrograph-specific residual cannot be astrophysical)
        min_fibers=50,
    )
    values = _values()
    main_value = "floor_min"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        fl = (f"floor = median continuum S/N per pixel of the faintest {p['faint_frac']:.0%} "
              "science fibers")
        return {
            "GOOD": f"-{p['snr_warn']:g} <= {fl} <= +{p['snr_warn']:g} in all channels (b, r, z)",
            "WARNING": (f"floor < -{p['snr_warn']:g} (over-subtraction) or floor > +{p['snr_warn']:g} "
                        "(possible under-subtraction, any value) in any channel"),
            "MAJOR ISSUE": f"floor < -{p['snr_major']:g} in any channel (over-subtraction)",
            "UNDEFINED": "SFrame missing or too few good science fibers",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        rows = ctx.good_sci_index
        specid = np.asarray(ctx.slitmap["spectrographid"])[rows]
        snr = {}
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            for ch, win in WINDOWS.items():
                w0, w1 = span(win)
                wv, f = ctx.sframe_window("FLUX", w0, w1, rows=rows)
                _, iv = ctx.sframe_window("IVAR", w0, w1, rows=rows)
                _, m = ctx.sframe_window("MASK", w0, w1, rows=rows)
                good = np.isfinite(f) & np.isfinite(iv) & (iv > 0) & (m == 0) & in_window(wv, win)[None, :]
                sig = np.nanmedian(np.where(good, 1.0 / np.sqrt(np.where(good, iv, 1.0)), np.nan), axis=1)
                med = np.nanmedian(np.where(good, f, np.nan), axis=1)
                s = med / sig
                s[good.sum(axis=1) <= 0.5 * in_window(wv, win).sum()] = np.nan
                snr[ch] = s
        vals = {"n_fibers": int(rows.size)}
        floors, efl, spfl = {}, {}, {}
        for ch in WINDOWS:
            s = snr[ch]
            ok = np.isfinite(s)
            if ok.sum() < p["min_fibers"]:
                continue
            vals.update({f"snr_med_{ch}": float(np.median(s[ok])),
                         f"frac_neg1_{ch}": float(np.mean(s[ok] < -1)),
                         f"frac_neg3_{ch}": float(np.mean(s[ok] < -3))})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rank = np.nanmean(np.vstack([snr[o] for o in WINDOWS if o != ch]), axis=0)
            okr = ok & np.isfinite(rank)
            if okr.sum() < p["min_fibers"]:
                continue
            q = okr & (rank <= np.percentile(rank[okr], 100 * p["faint_frac"]))
            fl = float(np.median(s[q]))
            floors[ch] = fl
            efl[ch] = 1.2533 * 1.4826 * float(np.median(np.abs(s[q] - fl))) / np.sqrt(q.sum())
            vals.update({f"floor_{ch}": fl, f"e_floor_{ch}": efl[ch], "n_faint": int(q.sum())})
            for sp in (1, 2, 3):
                # faintest fibers selected within each spectrograph
                oks = okr & (specid == sp)
                if oks.sum() >= p["min_fibers"] // 2:
                    qs = oks & (rank <= np.percentile(rank[oks], 100 * p["faint_frac"]))
                    spfl.setdefault(ch, {})[sp] = float(np.median(s[qs]))
                    vals[f"floor_sp{sp}_{ch}"] = spfl[ch][sp]
        if not floors:
            return self.result(QAFlag.UNDEFINED, note="too few good science fibers", **vals)
        # all floors: combined and per spectrograph (a spectrograph-specific residual is
        # diluted in the combined floor, e.g. expnum 7333: sp1 z +7.0 vs combined +0.26)
        allf = {f"{c}": v for c, v in floors.items()}
        allf.update({f"sp{sp}-{c}": v for c, d in spfl.items() for sp, v in d.items()})
        lo = min(allf, key=allf.get)
        hi = max(allf, key=allf.get)
        spread = {c: max(d.values()) - min(d.values()) for c, d in spfl.items() if len(d) >= 2}
        ws = max(spread, key=spread.get) if spread else None
        vals.update(floor_min=allf[lo], floor_max=allf[hi],
                    spec_spread=spread[ws] if ws else np.nan)
        fstr = " ".join(f"{c}{floors[c]:+.2f}" for c in floors)
        if allf[lo] < -p["snr_major"]:
            return self.result(QAFlag.MAJOR, note=f"over-subtracted, {lo} {allf[lo]:+.2f} ({fstr})", **vals)
        if ws and spread[ws] > p["spread_major"]:
            return self.result(QAFlag.MAJOR, note=f"spectrograph floors differ by {spread[ws]:.1f} in {ws} "
                                                  f"(worst {hi} {allf[hi]:+.2f})", **vals)
        if allf[lo] < -p["snr_warn"]:
            return self.result(QAFlag.WARNING, note=f"over-subtracted, {lo} {allf[lo]:+.2f} ({fstr})", **vals)
        if ws and spread[ws] > p["spread_warn"]:
            return self.result(QAFlag.WARNING, note=f"spectrograph floors differ by {spread[ws]:.1f} in {ws}", **vals)
        if allf[hi] > p["snr_warn"]:
            return self.result(QAFlag.WARNING, note=f"positive floor {hi} {allf[hi]:+.2f}: under-subtracted or "
                                                    f"uniform astrophysical continuum ({fstr})", **vals)
        return self.result(QAFlag.GOOD, note=f"floor {fstr}", **vals)
