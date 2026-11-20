"""
Re-derive the QA flags from the stored measurements (no SFrame / DAP access).

    python -m lvmqa.diagnostics DRPVER [DAPVER] [--tag T] [--set METRIC.PARAM=VALUE ...] [--pdf] [--check]
    python -m lvmqa.recalc ...       (same)
    from lvmqa.recalc import recalculate

Reads <base>.fits (flags) and <base>_supl.fits (measured values, notes), re-applies the
*classification* step of every metric with the current code defaults plus the given parameter
overrides, recomputes all summaries (qa_obs, science summaries), qa_overall, qa_bitmask and
the tile status, and writes <base>_recalculated.{fits,xlsx}, _supl.{fits,xlsx}, .html (and
optionally the diagnostics PDF).

Only thresholds applied to *stored* statistics can be changed this way ("flag parameters",
listed in FLAG_PARAMS). Parameters of the measurement itself (windows, masks, S/N cuts,
per-fiber outlier definitions, ...) need a full pipeline run. Metrics without a classifier,
and metrics that were UNDEFINED (missing inputs), keep their stored flag.

``--check`` re-classifies with unchanged parameters and reports every flag that differs from
the stored one (should be none: a consistency test of the classifiers against compute()).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from astropy.table import Table

from .bitmask import FLAG_LABELS, LABEL_TO_FLAG, QAFlag
from .config import PipelineConfig

log = logging.getLogger("lvmqa")
G, W, M, U = QAFlag.GOOD, QAFlag.WARNING, QAFlag.MAJOR, QAFlag.UNDEFINED
_SEV = {G: 0, W: 1, M: 2}

Classifier = Callable[[Any, Dict[str, Any], QAFlag, str], Optional[Tuple[QAFlag, str, Dict[str, Any]]]]
CLASSIFIERS: Dict[str, Classifier] = {}
# parameters that are applied in the classification step (changeable by recalculation)
FLAG_PARAMS: Dict[str, Tuple[str, ...]] = {}


def classifier(*names, params=()):
    def deco(fn):
        for n in names:
            CLASSIFIERS[n] = fn
            FLAG_PARAMS[n] = tuple(params)
        return fn
    return deco


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return math.nan
    return x


def _ok(x) -> bool:
    return math.isfinite(_f(x))


def _worst(*flags):
    fl = [f for f in flags if f in _SEV]
    return max(fl, key=_SEV.get) if fl else U


# ----------------------------------------------------------------------------- observational
@classifier("qa_illumination", params=("twilight_sun_alt", "moon_alt_min", "moon_sep_min", "fli_warn", "fli_major"))
def _illumination(m, v, flag, note):
    p = m.params
    sa, ma, fli, sep = (_f(v[k]) for k in ("sun_alt", "moon_alt", "moon_fli", "moon_sep"))
    if not _ok(sa):
        return None
    if sa > p["twilight_sun_alt"]:
        return M, f"twilight (sun alt {sa:.1f})", {}
    if not (_ok(ma) and _ok(fli) and _ok(sep)):
        return None
    if ma <= p["moon_alt_min"] or sep >= p["moon_sep_min"] or fli < p["fli_warn"]:
        return G, "", {}
    if fli < p["fli_major"]:
        return W, f"moon FLI {fli:.2f}, sep {sep:.0f}", {}
    return M, f"bright moon FLI {fli:.2f}, sep {sep:.0f}", {}


@classifier("qa_brightstars", params=("mag_major", "mag_warn"))
def _brightstars(m, v, flag, note):
    p = m.params
    b = _f(v["brightest_vmag"])
    if not _ok(b):
        return G, "", {"n_major": 0, "n_warn": 0}
    if b < p["mag_major"]:
        return M, f"V={b:.2f} star in field", {}
    if b < p["mag_warn"]:
        return W, f"V={b:.2f} star in field", {}
    return G, "", {}


@classifier("qa_guider", params=("fwhm_warn", "fwhm_major", "transparency_major"))
def _guider(m, v, flag, note):
    p = m.params
    major, warn = [], []
    if not bool(v["solved"]) and "not solved" in note:
        major.append("not solved")
    if bool(v["warn_match"]):
        major.append("no Gaia match")
    if bool(v["warn_tran"]):
        (major if p["transparency_major"] else warn).append("poor transparency")
    fw = _f(v["fwhm_med"])
    if _ok(fw) and fw > p["fwhm_major"]:
        major.append(f"FWHM {fw:.1f}\"")
    elif _ok(fw) and fw > p["fwhm_warn"]:
        warn.append(f"FWHM {fw:.1f}\"")
    if bool(v["warn_fwhm"]):
        warn.append("FWHM above median")
    if major:
        return M, ", ".join(major + warn), {}
    if warn:
        return W, ", ".join(warn), {}
    return G, "", {}


# ----------------------------------------------------------------------------- general
@classifier("qa_incomplete", params=("warn_frac_lost", "major_frac_lost_channel", "major_frac_lost_spec",
                                     "min_dap_coverage"))
def _incomplete(m, v, flag, note):
    p = m.params
    if not bool(v["sframe_found"]):
        return M, "SFrame missing", {}
    frac = {c: _f(v[f"frac_lost_{c}"]) for c in "brz"}
    wc = max(frac, key=lambda c: frac[c] if _ok(frac[c]) else -1)
    ms = _f(v["max_frac_lost_spec"])
    if frac[wc] > p["major_frac_lost_channel"] or ms > p["major_frac_lost_spec"]:
        return M, f"{frac[wc]:.0%} fibers lost in {wc}, worst spectrograph {ms:.0%}", {}
    notes, fl = [], G
    if not bool(v["dap_found"]):
        fl = W
        notes.append("DAP missing")
    elif not bool(v["dap_output_found"]):
        fl = W
        notes.append("DAP output missing")
    if _f(v["frac_lost"]) > p["warn_frac_lost"]:
        fl = W
        notes.append(f"{_f(v['frac_lost']):.1%} science fibers lost")
    cov = _f(v["dap_coverage"])
    if _ok(cov) and cov < p["min_dap_coverage"]:
        fl = W
        notes.append(f"DAP coverage {cov:.1%}")
    return fl, ", ".join(notes), {}


@classifier("qa_lineratios", params=("min_fibers", "siii_soften", "warn_dev_oiii", "major_dev_oiii", "warn_dev_nii",
                                     "major_dev_nii", "warn_dev_siii", "major_dev_siii", "nsig_median", "balmer_warn_frac",
                                     "balmer_major_frac"))
def _lineratios(m, v, flag, note):
    from .metrics.gen_lineratios import BALMER, DOUBLETS
    p = m.params
    sub, upd = {}, {}
    for t in DOUBLETS:
        n, dev, e = int(_f(v.get(f"{t}_n", -1)) if _ok(v.get(f"{t}_n")) else -1), _f(v.get(f"{t}_dev")), \
            _f(v.get(f"{t}_e_dev"))
        if _ok(dev) and not _ok(e):
            return None                      # run without e_dev (before v1.10): cannot re-classify
        sub[t] = m.doublet_classify(t, n, dev, e if _ok(e) else 0.0)
    softened = False
    if "siii" in sub:
        sub["siii"], softened = m.siii_soften(sub["siii"], _f(v.get("siii_n")), _f(v.get("siii_dev")),
                                              _f(v.get("siii_dev_raw")), _f(v.get("siii_e_dev_raw")))
    for t in BALMER:
        n, fb = _f(v.get(f"{t}_n")), _f(v.get(f"{t}_fbelow"))
        if not (n >= p["min_fibers"]) or not _ok(fb):
            sub[t] = U
        elif fb > p["balmer_major_frac"]:
            sub[t] = M
        elif fb > p["balmer_warn_frac"]:
            sub[t] = W
        else:
            sub[t] = G
    for t, f in sub.items():
        upd[f"{t}_flag"] = FLAG_LABELS[f]
    defined = {t: f for t, f in sub.items() if f != U}
    if not defined:
        return U, "no sub-test evaluable", upd
    bad = [t for t, f in defined.items() if f != G]
    nt = ("issues: " + ", ".join(f"{t}={defined[t].name}" for t in bad)) if bad else ""
    if softened:
        nt += " (siii softened: uncorrected ratio consistent, Ha/Hb reddening suspect)"
    return _worst(*defined.values()), nt, upd


# ----------------------------------------------------------------------------- DRP
@classifier("qa_absflux", params=("z_thresh", "n_major", "major_if_no_fluxcal"))
def _absflux(m, v, flag, note):
    p = m.params
    if p["major_if_no_fluxcal"] and str(v["fluxcal"]).strip().upper() == "NONE":
        return M, "no flux calibration (fluxcal=NONE)", {}
    if not bool(v["mod_found"]):
        return M, "MOD sensitivities absent", {}
    z = [_f(v.get(f"z_{c}")) for c in "brz"]
    if not all(_ok(x) for x in z):
        return None
    n = int(sum(x > p["z_thresh"] for x in z))
    if n >= p["n_major"]:
        return M, f"MOD/SCI disagree in {n} channels", {"n_outliers": n}
    if n:
        return W, "MOD/SCI disagree in 1 channel", {"n_outliers": n}
    return G, "", {"n_outliers": 0}


@classifier("qa_relflux_br", "qa_relflux_rz",
            params=("step_warn", "step_major", "kink_warn", "kink_major", "nsig", "min_src", "obs_fallback"))
def _relflux(m, v, flag, note):
    from .metrics.drp_relflux import classify
    p = m.params
    src, obs = _f(v.get("median_step_src")), _f(v.get("median_step_obs"))
    if not (_ok(src) or _ok(obs)):
        return None
    if "e_median_step_src" not in v and "e_median_step_obs" not in v:
        return None  # run made before v1.13 (no errors / slope mismatch): keep the stored flag
    n_src = _f(v.get("n_src"))
    if _ok(src) and n_src >= p["min_src"]:
        sig = "src"
    elif _ok(obs) and p["obs_fallback"]:
        sig = "obs"
    else:
        return U, "too few continuum pixels / fibers", {}
    med, e_med = _f(v.get(f"median_step_{sig}")), _f(v.get(f"e_median_step_{sig}"))
    kin, e_kin = _f(v.get(f"median_kink_{sig}")), _f(v.get(f"e_median_kink_{sig}"))
    upd = {"median_step": med, "e_median_step": e_med, "median_kink": kin, "e_median_kink": e_kin, "signal": sig}
    model = str(v.get("slope_model", "") or "").strip() or "sides"   # stored for the adopted signal
    if "slope_model" in v and sig != str(v.get("signal", sig)).strip():
        model = "sides"                       # signal changed by the new parameters: no stored model
    fl, nt = classify(p, med, e_med, kin, e_kin, sig, model, _f(v.get("median_kink_raw")))
    return fl, nt, upd


@classifier("qa_skyline_sub", params=("bad_all_lines", "bad_any", "warn_any"))
def _skyline(m, v, flag, note):
    from .metrics.drp_skyline import SKY_LINES, SPECS
    flags, maxabs, upd = {}, {}, {}
    for s in SPECS:
        r = {k: _f(v.get(f"{k}_sp{s}")) for k in SKY_LINES}
        if not any(_ok(x) for x in r.values()):
            flags[s] = U
        else:
            flags[s] = m.classify(r)
            maxabs[s] = max(abs(r[k]) for k in SKY_LINES if k != "OI5577" and _ok(r[k])) if any(
                _ok(r[k]) for k in SKY_LINES if k != "OI5577") else 0.0
        upd[f"flag_sp{s}"] = FLAG_LABELS[flags[s]]
    defined = {s: f for s, f in flags.items() if f != U}
    if not defined:
        return U, "no spectrograph evaluable", upd
    worst = max(defined, key=lambda s: (_SEV[defined[s]], maxabs.get(s, 0)))
    f = defined[worst]
    nt = "" if f == G else f"sp{worst}: " + ", ".join(f"{k}={_f(v.get(f'{k}_sp{worst}')):+.2f}" for k in SKY_LINES)
    upd.update(worst_spec=int(worst))
    return f, nt, upd


@classifier("qa_skycont_sub", params=("snr_warn", "snr_major", "spread_warn", "spread_major"))
def _skycont(m, v, flag, note):
    from .metrics.drp_skycont import WINDOWS
    p = m.params
    floors = {c: _f(v.get(f"floor_{c}")) for c in WINDOWS if _ok(v.get(f"floor_{c}"))}
    if not floors:
        return None
    allf = dict(floors)
    spfl = {}
    for c in WINDOWS:
        d = {s: _f(v.get(f"floor_sp{s}_{c}")) for s in (1, 2, 3) if _ok(v.get(f"floor_sp{s}_{c}"))}
        if d:
            spfl[c] = d
            allf.update({f"sp{s}-{c}": x for s, x in d.items()})
    lo, hi = min(allf, key=allf.get), max(allf, key=allf.get)
    spread = {c: max(d.values()) - min(d.values()) for c, d in spfl.items() if len(d) >= 2}
    ws = max(spread, key=spread.get) if spread else None
    upd = dict(floor_min=allf[lo], floor_max=allf[hi], spec_spread=spread[ws] if ws else np.nan)
    fstr = " ".join(f"{c}{floors[c]:+.2f}" for c in floors)
    if allf[lo] < -p["snr_major"]:
        return M, f"over-subtracted, {lo} {allf[lo]:+.2f} ({fstr})", upd
    if ws and spread[ws] > p["spread_major"]:
        return M, f"spectrograph floors differ by {spread[ws]:.1f} in {ws} (worst {hi} {allf[hi]:+.2f})", upd
    if allf[lo] < -p["snr_warn"]:
        return W, f"over-subtracted, {lo} {allf[lo]:+.2f} ({fstr})", upd
    if ws and spread[ws] > p["spread_warn"]:
        return W, f"spectrograph floors differ by {spread[ws]:.1f} in {ws}", upd
    if allf[hi] > p["snr_warn"]:
        return W, f"positive floor {hi} {allf[hi]:+.2f}: under-subtracted or uniform astrophysical continuum ({fstr})", upd
    return G, f"floor {fstr}", upd


@classifier("qa_flatfield", params=("warn_rel", "major_rel", "nsig", "frac_lines"))
def _flatfield(m, v, flag, note):
    from .metrics.drp_flatfield import CHANNELS, SPECS
    p = m.params
    mode = str(v.get("mode", "")).strip() or p["mode"]
    if mode != "relative":
        fb, fw = _f(v.get("frac_lines_bad")), _f(v.get("frac_lines_warn"))
        if not (_ok(fb) and _ok(fw)):
            return None
        if fb >= p["frac_lines"]:
            return M, f"{fb:.0%} of sky lines MAJOR", {}
        if fw >= p["frac_lines"]:
            return W, f"{fw:.0%} of sky lines deviate", {}
        return G, "", {}
    fl, worst, wd = G, None, -1.0
    for s in SPECS:
        for c in CHANNELS:
            d, e = _f(v.get(f"off_sp{s}_{c}")), _f(v.get(f"e_off_sp{s}_{c}"))
            if not _ok(d):
                continue
            if not _ok(e):
                return None                  # run without e_off (before v1.10)
            if abs(d) > wd:
                worst, wd = (s, c), abs(d)
            if abs(d) <= p["nsig"] * e:
                continue
            if abs(d) > 100 * p["major_rel"]:
                fl = M
            elif abs(d) > 100 * p["warn_rel"] and fl != M:
                fl = W
    if worst is None:
        return None
    nt = "" if fl == G else f"sp{worst[0]} {worst[1]}: {_f(v.get(f'off_sp{worst[0]}_{worst[1]}')):+.1f}% sky-line flux offset"
    return fl, nt, {}


@classifier("qa_wavecal_b", "qa_wavecal_r", "qa_wavecal_z", params=("frac_thresh", "min_frac_measured"))
def _wavecal(m, v, flag, note):
    p = m.params
    fw, fm, fmeas = _f(v.get("frac_warn")), _f(v.get("frac_major")), _f(v.get("frac_measured"))
    if not (_ok(fw) and _ok(fm)):
        return None
    if _ok(fmeas) and fmeas < p["min_frac_measured"]:
        return U, f"only {fmeas:.0%} fibers measured", {}
    if fm > p["frac_thresh"]:
        return M, f"{fm:.0%} fibers off by > LSF or failed ({_f(v.get('frac_failed')):.0%} failed)", {}
    if fw > p["frac_thresh"]:
        return W, f"{fw:.0%} fibers off by > LSF/5", {}
    return G, "", {}


@classifier("qa_lsf", params=("frac_thresh",))
def _lsf(m, v, flag, note):
    from .metrics._skylines import CHANNELS
    p = m.params
    fw = {c: _f(v.get(f"frac_warn_{c}")) for c in CHANNELS if _ok(v.get(f"frac_warn_{c}"))}
    fm = {c: _f(v.get(f"frac_major_{c}")) for c in CHANNELS if _ok(v.get(f"frac_major_{c}"))}
    if not fw:
        return None
    wch = max(fw, key=lambda c: (fm.get(c, 0), fw[c]))
    if fm.get(wch, 0) > p["frac_thresh"]:
        return M, f"{fm[wch]:.0%} fibers |Q-1|>{p['major_tol']:.0%} in {wch}", {}
    if max(fw.values()) > p["frac_thresh"]:
        c = max(fw, key=fw.get)
        return W, f"{fw[c]:.0%} fibers |Q-1|>{p['warn_tol']:.0%} in {c}", {}
    return G, "", {}


@classifier("qa_telluric", params=("ew_warn", "ew_major", "rms_warn", "rms_major", "nsig"))
def _telluric(m, v, flag, note):
    from .metrics.drp_telluric import BANDS
    p = m.params
    ews = {b: _f(v.get(f"ew_{b}")) for b in BANDS if _ok(v.get(f"ew_{b}"))}
    if not ews:
        return None
    rms = {b: _f(v.get(f"rms_{b}")) for b in ews}
    fl, notes = G, []
    flagged = {b: abs(e) for b, e in ews.items() if abs(e) > p["nsig"] * _f(v.get(f"e_ew_{b}", 0))}
    if flagged:
        fb = max(flagged, key=flagged.get)
        if flagged[fb] > p["ew_major"]:
            fl = M
        elif flagged[fb] > p["ew_warn"]:
            fl = W
        if flagged[fb] > p["ew_warn"]:
            notes.append(f"EW({fb}) = {ews[fb]:+.2f} A")
    rr = {b: x for b, x in rms.items() if _ok(x)}
    if rr:
        wr = max(rr, key=rr.get)
        if rr[wr] > p["rms_major"]:
            fl = M
            notes.append(f"rms({wr}) = {rr[wr]:.3f}")
        elif rr[wr] > p["rms_warn"]:
            fl = _worst(fl, W)
            notes.append(f"rms({wr}) = {rr[wr]:.3f}")
    return fl, "; ".join(notes), {}


# ----------------------------------------------------------------------------- DAP
@classifier("qa_fit_consistency", params=("min_points", "warn_frac_bad", "major_frac_bad", "warn_frac_masked",
                                          "major_frac_masked", "min_masked"))
def _fitcons(m, v, flag, note):
    p = m.params
    n, fb = _f(v.get("n_points")), _f(v.get("frac_bad"))
    if not _ok(fb) or not (n >= p["min_points"]):
        return None
    fl = M if fb > p["major_frac_bad"] else W if fb > p["warn_frac_bad"] else G
    nt = f"{fb:.0%} points disagree"
    fm, nm = _f(v.get("frac_masked_core")), _f(v.get("n_masked_core"))
    if _ok(fm):
        nt += f"; {fm:.1%} masked line cores"
        if nm >= p["min_masked"]:
            if fm > p["major_frac_masked"]:
                fl = M
            elif fm > p["warn_frac_masked"] and fl == G:
                fl = W
    return fl, nt, {}


@classifier("qa_dap_pmfit", params=("frac_thresh", "min_points"))
def _pmfit(m, v, flag, note):
    p = m.params
    n, fp, fb = _f(v.get("n_points")), _f(v.get("frac_poor")), _f(v.get("frac_bad"))
    if not (_ok(fp) and _ok(fb)) or not (n >= p["min_points"]):
        return None
    if fb > p["frac_thresh"]:
        return M, f"{fb:.0%} points with >{p['fx_major']:.0%} unmodelled flux", {}
    if fp > p["frac_thresh"]:
        return W, f"{fp:.0%} points with >{p['fx_warn']:.0%} unmodelled flux", {}
    return G, "", {}


@classifier("qa_gaskinematics", params=("dv_warn", "dv_major", "nsig", "frac_out_warn", "frac_out_major",
                                        "min_outliers", "boundary_warn", "boundary_major", "report_only"))
def _gaskin(m, v, flag, note):
    p = m.params
    pairs = {}
    for k in [n[3:] for n in v if n.startswith("dv_") and not n.startswith("dv_sp")]:
        g = k.split("_")[0]
        dv, e = _f(v.get(f"dv_{k}")), _f(v.get(f"e_dv_{k}"))
        if _ok(dv) and g not in p["report_only"]:
            pairs[k] = (dv, e)
    notes = []
    sig = {k: abs(d) for k, (d, e) in pairs.items() if abs(d) > p["nsig"] * e}
    n_poor = sum(x > p["dv_warn"] for x in sig.values())
    n_bad = sum(x > p["dv_major"] for x in sig.values())
    notes += [f"{k} {pairs[k][0]:+.0f} km/s" for k, x in sig.items() if x > p["dv_warn"]]
    fl_sp = G
    fov, fod = _f(v.get("frac_out_v")), _f(v.get("frac_out_disp"))
    if _ok(fov) or _ok(fod):
        nov, nod = _f(v.get("n_out_v")), _f(v.get("n_out_disp"))
        if not (_ok(nov) and _ok(nod) and nov >= 0):
            return None                      # run without n_out (before v1.10)
        fo = np.nanmax([fov, fod])
        if max(nov, nod) < p["min_outliers"]:
            fo = 0.0
        if fo > p["frac_out_major"]:
            fl_sp = M
        elif fo > p["frac_out_warn"]:
            fl_sp = W
        if fo > p["frac_out_warn"]:
            notes.append(f"{fo:.1%} kinematic outliers")
    mv, md = _f(v.get("max_boundary_dv")), _f(v.get("max_boundary_ddisp"))
    if _ok(mv):
        mb = max(mv, md if _ok(md) else 0)
        if mb > p["boundary_major"]:
            fl_sp = M
        elif mb > p["boundary_warn"] and fl_sp != M:
            fl_sp = W
        if mb > p["boundary_warn"]:
            notes.append(f"sp{v.get('worst_spec')} boundary offset {mb:.1f} km/s")
    if not pairs and not (_ok(fov) or _ok(fod)):
        return None
    fl = M if (n_bad or fl_sp == M) else W if (n_poor or fl_sp == W) else G
    return fl, "; ".join(notes), {"n_pairs_poor": n_poor, "n_pairs_bad": n_bad}


@classifier("qa_dap_npfluxes", params=("net_warn", "net_major", "nsig", "frac_poor_warn", "frac_bad_major", "sky_max"))
def _npfluxes(m, v, flag, note):
    from .metrics.dap_npfluxes import CATEGORY, GROUPS
    p = m.params
    status, nets, upd = {}, {}, {}
    for k in GROUPS:
        st0 = int(_f(v.get(f"status_{k}", -1))) if _ok(v.get(f"status_{k}")) else -1
        net, e, sky = _f(v.get(f"net_{k}")), _f(v.get(f"e_net_{k}")), _f(v.get(f"sky_{k}"))
        if st0 == -1 or not _ok(net):
            status[k] = st0 if st0 in (-1, -2) else -1
        elif _ok(sky) and sky > p["sky_max"]:
            status[k] = -2
        else:
            st = 0
            if abs(net) > p["nsig"] * e:
                st = 2 if abs(net) > p["net_major"] else 1 if abs(net) > p["net_warn"] else 0
            status[k] = st
            nets[k] = abs(net)
        upd[f"status_{k}"] = status[k]
    ev = [k for k, s in status.items() if s >= 0]
    n_poor = sum(status[k] >= 1 for k in ev)
    n_bad = sum(status[k] == 2 for k in ev)
    upd.update(n_groups=len(ev), n_poor=n_poor, n_bad=n_bad, n_sky=sum(s == -2 for s in status.values()),
               n_bad_bright=sum(status[k] == 2 for k in ev if CATEGORY[k] == "bright"),
               n_bad_faint=sum(status[k] == 2 for k in ev if CATEGORY[k] == "faint"))
    if not ev:
        return U, "no line group evaluable", upd
    lst = ", ".join(f"{k} {_f(v.get(f'net_{k}')):+.2f}" for k in ev if status[k] > 0)
    major = n_bad >= 2 and n_bad / len(ev) >= p["frac_bad_major"]
    for cat in ("bright", "faint"):
        ec = [k for k in ev if CATEGORY[k] == cat]
        nb = sum(status[k] == 2 for k in ec)
        major |= bool(ec) and nb >= 2 and nb / len(ec) >= p["frac_bad_major"]
    if major:
        return M, lst, upd
    if n_bad >= 1 or n_poor / len(ev) > p["frac_poor_warn"]:
        return W, lst, upd
    return G, lst, upd


@classifier("qa_absorption", params=("tol_warn", "tol_major", "nsig", "frac_poor_warn", "frac_bad_major",
                                     "flag_features"))
def _absorption(m, v, flag, note):
    p = m.params
    status, upd = {}, {}
    for k in p["features"]:
        d, e = _f(v.get(f"dI_{k}")), _f(v.get(f"e_dI_{k}"))
        if not _ok(d):
            status[k] = -1
            continue
        st = 0
        if abs(d) > p["nsig"] * e:
            st = 2 if abs(d) > p["tol_major"] else 1 if abs(d) > p["tol_warn"] else 0
        status[k] = st
        upd[f"status_{k}"] = st
    fl = [k for k in p["flag_features"] if status.get(k, -1) >= 0]
    n_poor = sum(status[k] >= 1 for k in fl)
    n_bad = sum(status[k] == 2 for k in fl)
    upd.update(n_poor=n_poor, n_bad=n_bad)
    if not fl:
        return U, "no feature measurable", upd
    nt = ", ".join(f"{k} {_f(v.get(f'dI_{k}')):+.2f}A" for k in fl if status[k] > 0)
    if n_bad >= 2 and n_bad / len(fl) >= p["frac_bad_major"]:
        return M, nt, upd
    if n_bad >= 1 or n_poor / len(fl) > p["frac_poor_warn"]:
        return W, nt, upd
    return G, nt, upd


@classifier("qa_fluxmap", params=("warn_dex", "major_dex", "nsig", "soften_additive", "soften_max_dex"))
def _fluxmap(m, v, flag, note):
    from .metrics.dap_fluxmap import _note
    from .multitile.fluxmap import classify_node
    p = m.params
    s, es = _f(v.get("s_dex")), _f(v.get("e_s_dex"))
    loc, el = _f(v.get("local_dex")), _f(v.get("e_local_dex"))
    ne, nr = _f(v.get("n_edges")), _f(v.get("n_rejected"))
    if not _ok(s) or not (ne > 0):
        return None
    if not _ok(loc):          # run made before v1.13: global offset s
        loc, el = s, es
    mult, add = _f(v.get("mult_dex")), _f(v.get("add_flux"))
    mult_rel = mult - (s - loc) if _ok(mult) else math.nan
    et = _f(v.get("e_local_tot_dex"))
    lab, cause = classify_node(loc, et if _ok(et) else el, int(ne), int(nr), p["warn_dex"], p["major_dex"], p["nsig"], mult_rel,
                               _f(v.get("e_mult_dex")), add, _f(v.get("e_add_flux")),
                               bool(p.get("soften_additive", True)), with_cause=True,
                               soften_max_dex=float(p.get("soften_max_dex", 0.2)))
    fl = {"GOOD": G, "WARNING": W, "MAJOR ISSUE": M}[lab]
    rad = _f(v.get("local_radius_used"))
    nn = _f(v.get("n_night"))
    nt = "" if fl == G else _note(loc, _f(v.get("n_local")) if _ok(_f(v.get("n_local"))) else 0,
                                  rad if _ok(rad) else m.params["local_radius"], s, nr, ne, cause,
                                  _f(v.get("night_dex")) if _ok(nn) and nn >= 3 else math.nan,
                                  _f(v.get("add_frac")), mult_rel)
    return fl, nt, ({"cause": cause} if "cause" in v else {})


@classifier("qa_gradient", params=("warn_dex", "major_dex", "nsig", "use_dipole"))
def _gradient(m, v, flag, note):
    from .metrics.dap_gradient import classify
    sp, dp = _f(v.get("spread_dex")), _f(v.get("ifu_dipole_dex"))
    if not (_ok(sp) or _ok(dp)):
        return None
    return (*classify(m.params, sp, _f(v.get("e_spread_dex")), dp, _f(v.get("e_ifu_dipole_dex"))), {})


@classifier("qa_starkinematics", params=("frac_warn", "frac_major", "dv_gas"))
def _starkin(m, v, flag, note):
    p = m.params
    f = _f(v.get("max_frac_out"))
    if not _ok(f):
        return None
    nt = f"{int(_f(v.get('n_regions')))} region(s), largest {int(_f(v.get('largest_region')))} fibers, outliers {f:.0%}"
    if f > p["frac_major"]:
        return M, nt, {}
    if f > p["frac_warn"]:
        return W, nt, {}
    if _f(v.get("n_degenerate")) > 0:
        return W, nt + f"; {int(_f(v.get('n_degenerate')))} degenerate region(s)", {}
    dsg = _f(v.get("dv_star_gas"))
    if _ok(dsg) and abs(dsg) > p["dv_gas"]:
        return W, nt + f"; v_star - v_Halpha = {dsg:+.0f} km/s", {}
    return G, nt, {}


@classifier("qa_status", params=("min_frac",))
def _status(m, v, flag, note):
    p = m.params
    upd = {}
    for k in ("auroral", "bright", "faint"):
        fr = _f(v.get(f"frac_{k}"))
        if _ok(fr):
            upd[k] = bool(fr >= p["min_frac"])
    return flag, note, upd


# ----------------------------------------------------------------------------- driver
def _row_values(m, row, cols) -> Dict[str, Any]:
    out = {}
    for spec in m.values:
        c = m.column(spec.name)
        if c not in cols:
            out[spec.name] = spec.fill_value()
            continue
        x = row[c]
        if np.ma.is_masked(x):
            out[spec.name] = spec.fill_value()
        elif hasattr(x, "item"):
            out[spec.name] = x.item()
        else:
            out[spec.name] = x
        if isinstance(out[spec.name], bytes):
            out[spec.name] = out[spec.name].decode()
    return out


def _parse_notes(s: str, names) -> Dict[str, str]:
    out, cur = {}, None
    for part in str(s).split("; "):
        k, sep, rest = part.partition(": ")
        if sep and k in names:
            cur = k
            out[k] = rest
        elif cur is not None:              # note containing "; " (e.g. qa_telluric)
            out[cur] += "; " + part
    return out


def recalculate(drp_version: str, dap_version: Optional[str] = None, *, tag: Optional[str] = None,
                metric_options: Optional[Dict[str, Dict[str, Any]]] = None, output_dir=None,
                write: bool = True, html: bool = True, pdf: bool = False, suffix: str = "_recalculated",
                check: bool = False, pdf_n: int = 5, pdf_seed: int = 1, nproc: int = 3, xlsx: bool = False,
                html_detail: Optional[bool] = None, hist_pdf: bool = False, exposure_file=None, **paths):
    """Re-derive all flags from the stored measurements (no SFrame / DAP access); returns
    (QAResult, changes table). Writes <base><suffix>.*; the diagnostics PDF (re-reads the data of
    the plotted exposures) only if ``pdf``."""
    if write and not check and not suffix:
        raise ValueError("suffix must not be empty (the original QA files would be overwritten)")
    paths = {k: v for k, v in paths.items() if v is not None}
    from .output import output_basename, write_outputs
    from .pipeline import ID_COLUMNS, QAResult, build_table, summarize_records
    from .registry import build_metrics

    cfg = PipelineConfig(drp_version=drp_version, dap_version=dap_version, output_dir=output_dir,
                         metric_options=dict(metric_options or {}), **paths)
    base = Path(cfg.output_dir) / output_basename(cfg, tag)
    qa = Table.read(f"{base}.fits", hdu="QA")
    supl = Table.read(f"{base}_supl.fits", hdu="SUPL")
    present = set(qa.colnames) | {c for c in supl.colnames}
    mlist = [m for m in build_metrics(None, cfg.metric_options)
             if m.name in qa.colnames or (m.status and any(c.startswith(m.short + "_") for c in supl.colnames))]
    names = {m.name for m in mlist}
    by_e = {int(r["expnum"]): r for r in qa}
    records, changes = [], []
    tcols = [c for c in supl.colnames if c.startswith("time_")]
    for row in supl:
        e = int(row["expnum"])
        qrow = by_e[e]
        rec = {"expnum": e, "flags": {}, "values": {}, "notes": _parse_notes(row["qa_notes"], names),
               "timing": {c[5:]: float(row[c]) for c in tcols}}
        for m in mlist:
            rec["values"][m.name] = _row_values(m, row, supl.colnames)
            old = LABEL_TO_FLAG.get(str(qrow[m.name]), U) if m.name in qa.colnames else (G if not m.status else G)
            rec["flags"][m.name] = int(old)
            if m.summary:
                continue
            fn = CLASSIFIERS.get(m.name)
            if fn is None or old == U:
                continue
            try:
                res = fn(m, rec["values"][m.name], old, rec["notes"].get(m.name, ""))
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: re-classification failed for %s (%s); keeping stored flag", m.name, e, exc)
                res = None
            if res is None:
                continue
            new, nt, upd = res
            rec["values"][m.name].update(upd)
            rec["flags"][m.name] = int(new)
            rec["notes"].pop(m.name, None)
            if nt:
                rec["notes"][m.name] = nt
        old_summ = {m.name: LABEL_TO_FLAG.get(str(qrow[m.name]), U) for m in mlist if m.summary}
        records.append((rec, qrow, old_summ))
    recs = [r for r, _, _ in records]
    summarize_records(recs, mlist)
    for rec, qrow, _ in records:
        for m in mlist:
            if m.name in qa.colnames:
                o, n = str(qrow[m.name]), FLAG_LABELS[QAFlag(rec["flags"][m.name])]
                if o != n:
                    changes.append((rec["expnum"], m.name, o, n))
    rows = supl[[c for c in ID_COLUMNS if c in supl.colnames]]
    from .pipeline import assign_groups, group_counts_text
    if exposure_file:     # (re)assign the groups; exposures not in the file -> "all" (v1.16)
        from .expfile import read_exposure_file
        spec = read_exposure_file(exposure_file)
        nin = sum(int(e) in spec for e in rows["expnum"])
        log.info("exposure file %s: groups of %d of %d exposures (others: all)", exposure_file, nin, len(rows))
        assign_groups(rows, spec)
    elif "group" not in rows.colnames:
        assign_groups(rows, None)
    table = build_table(rows, recs, mlist, cfg)
    table.meta.update({k: v for k, v in qa.meta.items() if k in ("RUNTIME",)})
    from .diagnostics import _inputs_from_header
    _inputs_from_header(cfg, base)   # carry the non-standard inputs (SFrame pattern etc.) over
    result = QAResult(table=table, metrics=mlist, config=cfg,
                      selection={"recalculated_from": f"{base.name}.fits",
                                 **({"exposure_file": str(exposure_file), "groups": group_counts_text(rows)}
                                    if exposure_file else {}),
                                 "overrides": str(metric_options or {}),
                                 "sframe_pattern": cfg.sframe_pattern or "", "no_dap": cfg.no_dap,
                                 "no_drpall": cfg.no_drpall})
    ch = Table(rows=changes, names=("expnum", "metric", "old", "new"),
               dtype=("i8", "U24", "U12", "U12")) if changes else \
        Table(names=("expnum", "metric", "old", "new"), dtype=("i8", "U24", "U12", "U12"))
    if check:
        return result, ch
    if write:
        result.files.update(write_outputs(result, tag, suffix, xlsx=xlsx))
        out = Path(cfg.output_dir) / output_basename(cfg, tag, suffix)
        if html:
            from .report import write_html
            result.files["html"] = write_html(result, Path(f"{out}.html"), detail=html_detail)
        ch.write(f"{out}_changes.csv", overwrite=True)
        result.files["changes"] = Path(f"{out}_changes.csv")
        if hist_pdf:     # value distributions only (no data access; v1.15)
            from .diagnostics import hist_pdf as _hist_pdf
            result.files["hist_pdf"] = _hist_pdf(Table.read(f"{out}.fits", hdu="QA"),
                                                 Table.read(f"{out}_supl.fits", hdu="SUPL"), cfg,
                                                 Path(f"{out}_diagnostics_hist.pdf"))
        if pdf:
            from .diagnostics import make_pdf
            result.files["pdf"] = make_pdf(drp_version, dap_version, n=pdf_n, seed=pdf_seed, nproc=nproc,
                                           tag=tag, suffix=suffix, metric_options=cfg.metric_options,
                                           output_dir=cfg.output_dir, **paths)
    return result, ch


def main(argv=None, prog=None):
    import ast
    ap = argparse.ArgumentParser(prog=prog, description="Re-derive the QA flags from the measurements stored "
                                 "in <base>_supl.fits (no metric is recomputed) and write <base>_recalculated.* "
                                 "(FITS, XLSX, HTML, changes CSV). The diagnostics PDF only with --pdf.")
    ap.add_argument("drp_version")
    ap.add_argument("dap_version", nargs="?", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="METRIC.PARAM=VALUE")
    ap.add_argument("--suffix", default="_recalculated")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--html-detail", action="store_true", help="detailed per-exposure HTML also for >= 1000 exposures")
    ap.add_argument("--xlsx", action="store_true", help="also write the XLSX copies of the tables")
    ap.add_argument("--pdf", action="store_true",
                    help="also regenerate the diagnostics PDF for the recalculated flags (re-reads the data)")
    ap.add_argument("--exposure-file", metavar="TOML", default=None,
                    help="(re)assign the exposure groups from this TOML file (others: 'all')")
    ap.add_argument("--hist-pdf", action="store_true",
                    help="PDF with only the metric value distributions and limits (no data access)")
    ap.add_argument("--n", type=int, default=5, help="PDF: random exposures per status and metric")
    ap.add_argument("--seed", type=int, default=1, help="PDF: random seed")
    ap.add_argument("-j", "--nproc", type=int, default=3, help="PDF: worker processes")
    ap.add_argument("--check", action="store_true",
                    help="consistency test: re-classify with unchanged parameters, report differences, write nothing")
    ap.add_argument("--list-params", action="store_true", help="list the parameters usable for recalculation")
    for k in ("redux_dir", "analysis_dir", "output_dir", "drpall_file", "cache_dir"):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None)
    ap.add_argument("--drpall", dest="drpall_file", default=None, help="alias of --drpall-file")
    a = ap.parse_args(argv)
    from .cli import check_version_arg
    check_version_arg(ap, "drp_version", a.drp_version)
    check_version_arg(ap, "dap_version", a.dap_version)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.list_params:
        for k, v in FLAG_PARAMS.items():
            print(f"{k:20s} {', '.join(v)}")
        return 0
    opts: Dict[str, Dict[str, Any]] = {}
    for it in a.set:
        key, _, val = it.partition("=")
        mname, _, par = key.partition(".")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        if par not in FLAG_PARAMS.get(mname, ()):
            log.warning("%s.%s is not a classification parameter: it has no effect without a full run",
                        mname, par)
        opts.setdefault(mname, {})[par] = val
    paths = {k: getattr(a, k) for k in ("redux_dir", "analysis_dir", "drpall_file", "cache_dir") if getattr(a, k)}
    res, ch = recalculate(a.drp_version, a.dap_version, tag=a.tag, metric_options=opts, output_dir=a.output_dir,
                          html=not a.no_html, pdf=a.pdf, suffix=a.suffix, check=a.check,
                          html_detail=True if a.html_detail else None, hist_pdf=a.hist_pdf, exposure_file=a.exposure_file,
                          pdf_n=a.n, pdf_seed=a.seed, nproc=a.nproc, xlsx=a.xlsx, **paths)
    from collections import Counter
    print(f"{len(ch)} flag change(s): " + ", ".join(f"{k}: {n}" for k, n in Counter(ch["metric"]).most_common()))
    if a.check:
        for r in ch:
            print(f"  {r['expnum']} {r['metric']}: stored {r['old']} -> re-classified {r['new']}")
    for k, p in res.files.items():
        print(f"  {k:10s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
