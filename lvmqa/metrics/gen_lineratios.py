"""
qa_lineratios - emission-line ratio consistency (General group).

Uses the NON-PARAMETRIC DAP measurements only (the parametric fit fixes the [OIII] and
[NII] doublet ratios, and parametric-vs-non-parametric consistency is the job of
qa_fit_consistency). Only fibers where all lines needed by a sub-test have S/N > ``min_snr``
are used.

Sub-tests
---------
oiii : [OIII] 5007/4959  vs 2.98                (Storey & Zeippen 2000)
nii  : [NII]  6584/6548  vs 2.96                (Storey & Zeippen 2000)
siii : [SIII] 9531/9069  vs 2.47, reddened per fiber with E(B-V) from Halpha/Hbeta
       (intrinsic 2.86, CCM89, R_V=3.1; Halpha/Hbeta < 2.86 -> A_V = 0)
       -> statistic: median(R_obs / R_expected) - 1; flagged if larger than the
          threshold AND significant (> nsig_median x error of the median)
hahb : Halpha/Hbeta vs 2.86  }  fraction of fibers whose ratio is below the intrinsic
hbhg : Hbeta/Hgamma vs 2.14  }  Case-B value by more than ``balmer_nsigma`` sigma

The metric flag is the worst flag over the sub-tests that could be evaluated.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..bitmask import FLAG_LABELS, QAFlag
from ..registry import GROUP_GEN, Metric, ValueSpec, register
from ..utils.extinction import ebv_from_balmer, k_lambda

LINES: Dict[str, Tuple[str, float]] = {
    "oiii4959": ("[OIII]_4958.91", 4958.91), "oiii5007": ("[OIII]_5006.84", 5006.84),
    "nii6548": ("[NII]_6548.05", 6548.05), "nii6584": ("[NII]_6583.45", 6583.45),
    "siii9069": ("[SIII]_9069.0", 9069.0), "siii9531": ("[SIII]_9531.1", 9531.1),
    "ha": ("Halpha_6562.85", 6562.85), "hb": ("Hbeta_4861.36", 4861.36),
    "hg": ("Hgamma_4340.49", 4340.49),
}
DOUBLETS = {"oiii": ("oiii5007", "oiii4959"), "nii": ("nii6584", "nii6548"),
            "siii": ("siii9531", "siii9069")}
BALMER = {"hahb": ("ha", "hb"), "hbhg": ("hb", "hg")}
_SEV = {QAFlag.GOOD: 0, QAFlag.WARNING: 1, QAFlag.MAJOR: 2}


def _nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def _values():
    v = []
    for t in DOUBLETS:
        v.append(ValueSpec(f"{t}_flag", "U11", "", f"Flag of the {t} ratio sub-test"))
        v.append(ValueSpec(f"{t}_n", "i8", "", f"{t}: fibers used"))
        v.append(ValueSpec(f"{t}_dev", "f8", "percent", f"{t}: median(R_obs/R_expected) - 1"))
    v.append(ValueSpec("siii_expratio", "f8", "", "Median expected (reddened) [SIII] 9531/9069"))
    v.append(ValueSpec("av_med", "f8", "mag", "Median A_V from Halpha/Hbeta of fibers in the siii test"))
    for t in BALMER:
        v.append(ValueSpec(f"{t}_flag", "U11", "", f"Flag of the {t} Balmer sub-test"))
        v.append(ValueSpec(f"{t}_n", "i8", "", f"{t}: fibers used"))
        v.append(ValueSpec(f"{t}_med", "f8", "", f"{t}: median observed ratio"))
        v.append(ValueSpec(f"{t}_fbelow", "f8", "",
                           f"{t}: fraction of fibers below the intrinsic ratio by > balmer_nsigma"))
    return tuple(v)


@register
class LineRatios(Metric):
    name = "qa_lineratios"
    slot = 2
    group = GROUP_GEN
    title = "Emission-line ratio consistency"
    description = ("Fixed doublet ratios ([OIII], [NII], [SIII] corrected for extinction) and "
                   "Balmer decrements (Ha/Hb, Hb/Hg) vs theory, from non-parametric DAP fluxes")
    requires = ("dap",)
    defaults = dict(
        min_snr=10.0,                # all lines of a sub-test must have S/N above this
        min_fibers=5,                # minimum number of fibers for a sub-test verdict
        ratio_oiii=2.98, ratio_nii=2.96, ratio_siii=2.47,
        ratio_hahb=2.86, ratio_hbhg=2.137,
        warn_dev_oiii=10.0, major_dev_oiii=20.0,   # |median deviation| [percent]
        warn_dev_nii=10.0, major_dev_nii=20.0,
        warn_dev_siii=15.0, major_dev_siii=30.0,
        nsig_median=3.0,             # deviation must also exceed this x error of the median
        balmer_nsigma=5.0,           # fiber counts as 'below theory' at this significance
        balmer_warn_frac=0.05, balmer_major_frac=0.20,
        rv=3.1, extinction_law="ccm89",
    )
    values = _values()
    main_value = "hahb_fbelow"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": "All evaluable sub-tests consistent with theory",
            "WARNING": (f"Any sub-test: doublet |median dev| > {p['warn_dev_oiii']:g}% ([OIII]), "
                        f"{p['warn_dev_nii']:g}% ([NII]), {p['warn_dev_siii']:g}% ([SIII], "
                        f"extinction-corrected) and > {p['nsig_median']:g} sigma; or > "
                        f"{p['balmer_warn_frac']:.0%} of fibers with Ha/Hb or Hb/Hg below Case B "
                        f"by > {p['balmer_nsigma']:g} sigma. Non-parametric fluxes, fibers with "
                        f"S/N > {p['min_snr']:g} in all lines of the sub-test, >= {p['min_fibers']} fibers"),
            "MAJOR ISSUE": (f"Same, with doublet |median dev| > {p['major_dev_oiii']:g}% / "
                            f"{p['major_dev_nii']:g}% / {p['major_dev_siii']:g}%, or Balmer "
                            f"fraction > {p['balmer_major_frac']:.0%}"),
            "UNDEFINED": "No DAP file, or no sub-test with enough high-S/N fibers",
        }

    def _fluxes(self, ctx):
        """{line_key: (flux, err)} from NP_ELINES_*, aligned on the DAP PT ids."""
        cols = []
        for name, _ in LINES.values():
            cols += [f"flux_{name}", f"e_flux_{name}"]
        npt = ctx.dap_np_columns(cols)
        return {key: (np.asarray(npt[f"flux_{name}"], dtype=float),
                      np.asarray(npt[f"e_flux_{name}"], dtype=float))
                for key, (name, _) in LINES.items()}

    def _good(self, fl, keys):
        ok = np.ones(len(next(iter(fl.values()))[0]), dtype=bool)
        for k in keys:
            f, e = fl[k]
            with np.errstate(invalid="ignore", divide="ignore"):
                ok &= np.isfinite(f) & np.isfinite(e) & (e > 0) & (f / e > self.params["min_snr"])
        return ok

    def _doublet_flag(self, test, y):
        """y = R_obs / R_expected per fiber -> (flag, n, deviation in percent)."""
        p = self.params
        n = len(y)
        if n < p["min_fibers"]:
            return QAFlag.UNDEFINED, n, np.nan
        dev = float(np.median(y) - 1.0)
        err = 1.2533 * _nmad(y) / np.sqrt(n)
        a = abs(dev) * 100.0
        signif = abs(dev) > p["nsig_median"] * err
        if signif and a > p[f"major_dev_{test}"]:
            return QAFlag.MAJOR, n, 100 * dev
        if signif and a > p[f"warn_dev_{test}"]:
            return QAFlag.WARNING, n, 100 * dev
        return QAFlag.GOOD, n, 100 * dev

    def _balmer_flag(self, r, sig, theo):
        p = self.params
        n = len(r)
        med = float(np.median(r)) if n else np.nan
        if n < p["min_fibers"]:
            return QAFlag.UNDEFINED, n, med, np.nan
        fb = float(np.mean((r - theo) / sig < -p["balmer_nsigma"]))
        if fb > p["balmer_major_frac"]:
            return QAFlag.MAJOR, n, med, fb
        if fb > p["balmer_warn_frac"]:
            return QAFlag.WARNING, n, med, fb
        return QAFlag.GOOD, n, med, fb

    def compute(self, ctx):
        p = self.params
        fl = self._fluxes(ctx)
        vals = {}
        sub = {}
        k_9069, k_9531 = k_lambda([9069.0, 9531.1], p["rv"], p["extinction_law"])
        for test, (num, den) in DOUBLETS.items():
            keys = [num, den] + (["ha", "hb"] if test == "siii" else [])
            ok = self._good(fl, keys)
            r = fl[num][0][ok] / fl[den][0][ok]
            expected = np.full(r.shape, p[f"ratio_{test}"])
            if test == "siii":
                ebv = ebv_from_balmer(fl["ha"][0][ok] / fl["hb"][0][ok], p["ratio_hahb"],
                                      p["rv"], p["extinction_law"])
                expected = p["ratio_siii"] * 10 ** (0.4 * ebv * (k_9069 - k_9531))
                if ok.sum():
                    vals["siii_expratio"] = float(np.median(expected))
                    vals["av_med"] = float(np.median(p["rv"] * ebv))
            sub[test], vals[f"{test}_n"], vals[f"{test}_dev"] = self._doublet_flag(test, r / expected)

        for test, (num, den) in BALMER.items():
            ok = self._good(fl, [num, den])
            fn, en = fl[num][0][ok], fl[num][1][ok]
            fd, ed = fl[den][0][ok], fl[den][1][ok]
            r = fn / fd
            sig = r * np.sqrt((en / fn) ** 2 + (ed / fd) ** 2)
            sub[test], n, med, fb = self._balmer_flag(r, sig, p[f"ratio_{test}"])
            vals.update({f"{test}_n": n, f"{test}_med": med, f"{test}_fbelow": fb})

        for t, f in sub.items():
            vals[f"{t}_flag"] = FLAG_LABELS[f]
        defined = {t: f for t, f in sub.items() if f != QAFlag.UNDEFINED}
        if not defined:
            return self.result(QAFlag.UNDEFINED, note=f"no sub-test with >= {p['min_fibers']} "
                               f"fibers at S/N > {p['min_snr']:g}", **vals)
        worst = max(defined.values(), key=lambda f: _SEV[f])
        bad = [t for t, f in defined.items() if f != QAFlag.GOOD]
        note = ("issues: " + ", ".join(f"{t}={defined[t].name}" for t in bad)) if bad else ""
        return self.result(worst, note=note, **vals)
