"""
qa_gaskinematics - prototype: consistency of the ionised-gas kinematics measured by the DAP
(non-parametric velocities and dispersions, NP_ELINES).

A) Line-to-line velocity consistency (fibers with S/N > min_snr in both lines; median
   difference over fibers, uncertainty 1.25 NMAD / sqrt(n)):
     H     Hbeta, Hgamma and the 2 best Paschen lines (P9-P12, S/N criterion) vs Halpha
     SIII  [SIII]9531 vs [SIII]9069
     HeI   HeI 6678, 7065, 4471, 7281 vs HeI 5876
     OIIb  [OII]3729 vs 3726       (report only: the NP moments of the blended doublet are
                                    pulled towards each other, ~ -10 km/s in all exposures)
     OIIr  [OII]7330 vs 7320       (report only: blends of two doublets + [CaII]7324)
   Velocities are re-referenced to accurate air rest wavelengths: the DAP line list uses e.g.
   Halpha 6562.85 (6562.80), Hbeta 4861.36 (4861.33), HeI 5876.0 (5875.62: -19 km/s),
   [SIII] 9069.0/9531.1 (9068.6/9530.6: -13/-16 km/s), [OII] 7318.92/7329.66 (blends at
   ~7320.1/7330.2). These pairs compare b, r and z channels and different parts of each
   channel, i.e. they test the relative wavelength calibration with science lines,
   independently of the sky lines used by the DRP.
   Pair 'poor' if |dv| > dv_warn (15 km/s ~ LSF/5) and > 3 sigma; 'bad' if > dv_major (30).
B) Spatial consistency of the brightest line (``line``, Halpha): for fibers with S/N > min_snr
   the velocity and dispersion are compared with the median of their neighbours within
   ``radius_pitch`` x fiber pitch (2.6 x 37" ~ 96"; >= 4 neighbours). Outlier: |residual|
   > max(nsig x NMAD of all residuals, floor 15 km/s; 10 km/s before v1.8). In addition, the fibers of each
   spectrograph are compared with adjacent fibers (< 60") of the other spectrographs
   (spectrograph-boundary offsets in v and sigma).
GOOD:    no flag pair poor, outliers <= 1%, |boundary offsets| <= 3 km/s
WARNING: a flag pair poor, or outliers > 1%, or a boundary offset > 3 km/s
MAJOR:   a flag pair bad, or outliers > 3%, or a boundary offset > 10 km/s
Note: the NP velocity errors (e_vel_*) of DAP 1.1 are unusable (median ~3000 km/s).
Prototype thresholds (DRP 1.2.0 test set: neighbour residual NMAD 0.4-0.9 km/s, outliers
0-0.4%, boundary offsets < 1.2 km/s; Hbeta-Halpha -6...-8 km/s in all exposures).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from ._dapout import C_KMS

# DAP line name -> accurate air rest wavelength [A]
REST = {
    "Halpha_6562.85": 6562.80, "Hbeta_4861.36": 4861.33, "Hgamma_4340.49": 4340.47,
    "HI_9229.02": 9229.01, "HI_9014.91": 9014.91, "HI_8862.78": 8862.78, "HI_8750.47": 8750.47,
    "[SIII]_9069.0": 9068.6, "[SIII]_9531.1": 9530.6,
    "[OII]_3726.03": 3726.03, "[OII]_3728.82": 3728.82,
    "[OII]_7318.92": 7320.1, "[OII]_7329.66": 7330.2,
    "HeI_5876.0": 5875.62, "HeI_6678.15": 6678.15, "HeI_7065.19": 7065.19,
    "HeI_4471.48": 4471.48, "HeI_7281.35": 7281.35,
    "[NII]_6583.45": 6583.45, "[OIII]_5006.84": 5006.84,
}
PASCHEN = ("HI_9229.02", "HI_9014.91", "HI_8862.78", "HI_8750.47")
GROUPS = {
    "H": ("Halpha_6562.85", ("Hbeta_4861.36", "Hgamma_4340.49")),
    "SIII": ("[SIII]_9069.0", ("[SIII]_9531.1",)),
    "HeI": ("HeI_5876.0", ("HeI_6678.15", "HeI_7065.19", "HeI_4471.48", "HeI_7281.35")),
    "OIIb": ("[OII]_3726.03", ("[OII]_3728.82",)),
    "OIIr": ("[OII]_7318.92", ("[OII]_7329.66",)),
}
REPORT_ONLY = ("OIIb", "OIIr")


def pkey(line):
    import re
    n, w = line.rsplit("_", 1)
    return re.sub(r"[^A-Za-z0-9]", "", n) + f"{float(w):.0f}"


def _pairs():
    out = []
    for g, (ref, others) in GROUPS.items():
        for o in others + (PASCHEN if g == "H" else ()):
            out.append((g, ref, o))
    return out


def _values():
    v = [ValueSpec("n_pairs", "i8", "", "Line pairs evaluated (flag groups)"),
         ValueSpec("n_pairs_poor", "i8", "", "Flag pairs with |dv| > dv_warn"),
         ValueSpec("n_pairs_bad", "i8", "", "Flag pairs with |dv| > dv_major"),
         ValueSpec("worst_pair", "U24", "", "Flag pair with the largest |dv|"),
         ValueSpec("max_abs_dv", "f8", "km/s", "Largest |dv| over the flag pairs"),
         ValueSpec("line", "U16", "", "Line used for the spatial test"),
         ValueSpec("n_spatial", "i8", "", "Fibers in the spatial test"),
         ValueSpec("sig_res_v", "f8", "km/s", "NMAD of velocity - neighbour median"),
         ValueSpec("sig_res_disp", "f8", "km/s", "NMAD of dispersion - neighbour median"),
         ValueSpec("frac_out_v", "f8", "", "Fraction of velocity outliers"),
         ValueSpec("frac_out_disp", "f8", "", "Fraction of dispersion outliers"),
         ValueSpec("n_out_v", "i8", "", "Number of velocity outliers"),
         ValueSpec("n_out_disp", "i8", "", "Number of dispersion outliers"),
         ValueSpec("max_boundary_dv", "f8", "km/s", "Largest |velocity offset| of a spectrograph vs adjacent fibers of the others"),
         ValueSpec("max_boundary_ddisp", "f8", "km/s", "Largest |dispersion offset| of a spectrograph vs the others"),
         ValueSpec("worst_spec", "i8", "", "Spectrograph with the largest boundary offset")]
    for s in (1, 2, 3):
        v += [ValueSpec(f"dv_sp{s}", "f8", "km/s", f"Spectrograph {s}: median v - v(adjacent fibers of other spectrographs)"),
              ValueSpec(f"ddisp_sp{s}", "f8", "km/s", f"Spectrograph {s}: median sigma - sigma(adjacent fibers of other spectrographs)")]
    for g, ref, o in _pairs():
        k = f"{g}_{pkey(o)}"
        v += [ValueSpec(f"dv_{k}", "f8", "km/s", f"v({o}) - v({ref}), median over fibers"),
              ValueSpec(f"e_dv_{k}", "f8", "km/s", f"Uncertainty of dv_{k}"),
              ValueSpec(f"n_{k}", "i8", "", f"Fibers with S/N > min_snr in {o} and {ref}")]
    return tuple(v)


def _np_all(ctx):
    cache = ctx.__dict__.setdefault("_np_all_cache", {})
    if "t" not in cache:
        from astropy.table import join
        t = ctx.dap_table("PT")["id", "fiberid", "ra", "dec"]
        for ext in ("NP_ELINES_B", "NP_ELINES_R", "NP_ELINES_I"):
            try:
                x = ctx.dap_table(ext)
            except Exception:  # noqa: BLE001
                continue
            t = join(t, x, keys="id", join_type="left")
        # restore PT order
        order = {k: i for i, k in enumerate(np.asarray(ctx.dap_table("PT")["id"]).astype(str))}
        t = t[np.argsort([order[str(k)] for k in t["id"]])]
        cache["t"] = t
    return cache["t"]


class _Neighbours:
    """Minimal fixed-radius neighbour search (numpy only; ~2000 points -> full distance matrix)."""

    def __init__(self, P):
        self.P = np.asarray(P, float)
        d = np.hypot(self.P[:, None, 0] - self.P[None, :, 0], self.P[:, None, 1] - self.P[None, :, 1])
        self.D = d
        dd = d + np.diag(np.full(len(d), np.inf))
        self.pitch = float(np.median(dd.min(axis=1))) if len(d) > 1 else np.nan

    def query_ball_point(self, X, r):
        X = np.asarray(X, float)
        if X.ndim == 1:
            return list(np.where(np.hypot(*(self.P - X).T) <= r)[0])
        return [list(np.where(row <= r)[0]) for row in self.D]


def _col(t, name):
    return np.asarray(t[name], dtype=float) if name in t.colnames else np.full(len(t), np.nan)


@register
class GasKinematics(Metric):
    name = "qa_gaskinematics"
    slot = 16
    group = GROUP_DAP
    title = "Ionised-gas kinematics"
    description = ("Prototype: consistency of DAP NP gas velocities between lines of the same "
                   "species (b/r/z channels) and spatial consistency of velocity and dispersion "
                   "of the brightest line between neighbouring fibers and spectrographs")
    requires = ("dap",)
    defaults = dict(
        min_snr=10.0,
        min_fibers=10,
        n_paschen=2,
        dv_warn=15.0, dv_major=30.0, nsig=3.0,
        line="Halpha_6562.85",
        min_spatial=100,     # fibers with S/N > min_snr needed for the spatial tests
        min_outliers=3,      # fewer outliers never raise a flag
        min_boundary=20,     # boundary fibers per spectrograph needed
        radius_pitch=2.6, min_neighbours=4, out_nsig=5.0, out_floor=15.0,
        frac_out_warn=0.01, frac_out_major=0.03,
        boundary_radius=60.0, boundary_warn=3.0, boundary_major=10.0,
        report_only=REPORT_ONLY,
    )
    values = _values()
    main_value = "max_abs_dv"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"|dv| <= {p['dv_warn']:g} km/s for all line pairs, <= {p['frac_out_warn']:.0%} "
                     f"spatial outliers, spectrograph boundary offsets <= {p['boundary_warn']:g} km/s"),
            "WARNING": (f"a line pair with |dv| > {p['dv_warn']:g} km/s, or > {p['frac_out_warn']:.0%} outliers "
                        f"in v or sigma, or a boundary offset > {p['boundary_warn']:g} km/s"),
            "MAJOR ISSUE": (f"a line pair with |dv| > {p['dv_major']:g} km/s, or > {p['frac_out_major']:.0%} "
                            f"outliers, or a boundary offset > {p['boundary_major']:g} km/s"),
            "UNDEFINED": "No DAP file or too few fibers with bright lines",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")
        t = _np_all(ctx)
        vals, notes = {}, []

        def vel(line):
            v = _col(t, f"vel_{line}")
            wl = float(line.rsplit("_", 1)[1])
            return v + C_KMS * (wl - REST.get(line, wl)) / REST.get(line, wl)

        def snr(line):
            with np.errstate(invalid="ignore", divide="ignore"):
                return _col(t, f"flux_{line}") / _col(t, f"e_flux_{line}")

        # ---- A) line pairs
        pas = sorted(PASCHEN, key=lambda l: -np.sum(snr(l) > p["min_snr"]))[:p["n_paschen"]]
        pair_res = {}
        for g, ref, o in _pairs():
            if o in PASCHEN and o not in pas:
                continue
            s = (snr(ref) > p["min_snr"]) & (snr(o) > p["min_snr"])
            dv = (vel(o) - vel(ref))[s]
            dv = dv[np.isfinite(dv)]
            k = f"{g}_{pkey(o)}"
            vals[f"n_{k}"] = int(dv.size)
            if dv.size < p["min_fibers"]:
                continue
            m = float(np.median(dv))
            e = 1.2533 * 1.4826 * float(np.median(np.abs(dv - m))) / np.sqrt(dv.size)
            vals.update({f"dv_{k}": m, f"e_dv_{k}": e})
            if g not in p["report_only"]:
                pair_res[k] = (m, e)
        sig = {k: abs(m) for k, (m, e) in pair_res.items() if abs(m) > p["nsig"] * e}
        n_poor = sum(v > p["dv_warn"] for v in sig.values())
        n_bad = sum(v > p["dv_major"] for v in sig.values())
        vals.update(n_pairs=len(pair_res), n_pairs_poor=n_poor, n_pairs_bad=n_bad)
        if pair_res:
            w = max(pair_res, key=lambda k: abs(pair_res[k][0]))
            vals.update(worst_pair=w, max_abs_dv=abs(pair_res[w][0]))
        for k, v in sig.items():
            if v > p["dv_warn"]:
                notes.append(f"{k} {pair_res[k][0]:+.0f} km/s")

        # ---- B) spatial consistency
        line = p["line"]
        wl = float(line.rsplit("_", 1)[1])
        v = vel(line)
        d = _col(t, f"disp_{line}") / wl * C_KMS
        ok = (snr(line) > p["min_snr"]) & np.isfinite(v) & np.isfinite(d)
        vals.update(line=line, n_spatial=int(ok.sum()))
        flag_sp = QAFlag.GOOD
        if ok.sum() >= p["min_spatial"]:
            ra, dec = _col(t, "ra"), _col(t, "dec")
            x = (ra - np.nanmedian(ra)) * np.cos(np.deg2rad(np.nanmedian(dec))) * 3600.0
            y = (dec - np.nanmedian(dec)) * 3600.0
            P = np.c_[x, y][ok]
            tree = _Neighbours(P)
            pitch = tree.pitch
            vv, dd = v[ok], d[ok]
            rv = np.full(vv.size, np.nan)
            rd = np.full(vv.size, np.nan)
            for i, nb in enumerate(tree.query_ball_point(P, p["radius_pitch"] * pitch)):
                nb = [j for j in nb if j != i]
                if len(nb) >= p["min_neighbours"]:
                    rv[i] = vv[i] - np.median(vv[nb])
                    rd[i] = dd[i] - np.median(dd[nb])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sv = 1.4826 * float(np.nanmedian(np.abs(rv)))
                sd = 1.4826 * float(np.nanmedian(np.abs(rd)))
                fin = np.isfinite(rv)
                ov = np.abs(rv[fin]) > max(p["out_nsig"] * sv, p["out_floor"])
                od = np.abs(rd[fin]) > max(p["out_nsig"] * sd, p["out_floor"])
            fo_v, fo_d = float(ov.mean()) if ov.size else np.nan, float(od.mean()) if od.size else np.nan
            vals.update(sig_res_v=sv, sig_res_disp=sd, frac_out_v=fo_v, frac_out_disp=fo_d,
                        n_out_v=int(ov.sum()), n_out_disp=int(od.sum()))
            fo = np.nanmax([fo_v, fo_d])
            if max(ov.sum(), od.sum()) < p["min_outliers"]:
                fo = 0.0
            if fo > p["frac_out_major"]:
                flag_sp = QAFlag.MAJOR
            elif fo > p["frac_out_warn"]:
                flag_sp = QAFlag.WARNING
            if fo > p["frac_out_warn"]:
                notes.append(f"{fo:.1%} kinematic outliers")
            # spectrograph boundaries
            if ctx.has_sframe:
                rows = ctx.dap_sframe_rows
                spec = np.where(rows >= 0, np.asarray(ctx.slitmap["spectrographid"])[np.clip(rows, 0, None)], -1)
            else:
                spec = (np.asarray(t["fiberid"]).astype(int) - 1) // 648 + 1
            sp = spec[ok]
            bdv, bdd = {}, {}
            for s in (1, 2, 3):
                r1, r2 = [], []
                for i in np.where(sp == s)[0]:
                    nb = [j for j in tree.query_ball_point(P[i], p["boundary_radius"]) if sp[j] != s]
                    if len(nb) >= 2:
                        r1.append(vv[i] - np.median(vv[nb]))
                        r2.append(dd[i] - np.median(dd[nb]))
                if len(r1) >= p["min_boundary"]:
                    bdv[s], bdd[s] = float(np.median(r1)), float(np.median(r2))
                    vals.update({f"dv_sp{s}": bdv[s], f"ddisp_sp{s}": bdd[s]})
            if bdv:
                ws = max(bdv, key=lambda s: max(abs(bdv[s]), abs(bdd[s])))
                mv, md = max(abs(x) for x in bdv.values()), max(abs(x) for x in bdd.values())
                vals.update(max_boundary_dv=mv, max_boundary_ddisp=md, worst_spec=int(ws))
                mb = max(mv, md)
                if mb > p["boundary_major"]:
                    flag_sp = QAFlag.MAJOR
                elif mb > p["boundary_warn"] and flag_sp != QAFlag.MAJOR:
                    flag_sp = QAFlag.WARNING
                if mb > p["boundary_warn"]:
                    notes.append(f"sp{ws} offset v {bdv[ws]:+.1f}, sigma {bdd[ws]:+.1f} km/s")
        if not pair_res and ok.sum() < p["min_spatial"]:
            return self.result(QAFlag.UNDEFINED, note="too few fibers with bright lines", **vals)
        flag = QAFlag.MAJOR if (n_bad or flag_sp == QAFlag.MAJOR) else \
            QAFlag.WARNING if (n_poor or flag_sp == QAFlag.WARNING) else QAFlag.GOOD
        return self.result(flag, note="; ".join(notes), **vals)
