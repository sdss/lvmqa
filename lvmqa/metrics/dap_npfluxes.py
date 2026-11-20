"""
qa_dap_npfluxes (formerly qa_faintlines) - flux bias of the DAP emission-line model for bright
and key faint lines, from the stacked residuals in the DAP output file (non-parametric model
by default; the parametric one when available for the line, ``model='auto'``).

Line windows (rest frame, A, see _lines.py). All DAP lines inside a window form one group:
the selection uses their summed NP flux, and the model flux is that of all lines in the
window, so blends (e.g. HeI4471 + [FeII]4470,4475; [OII]7320,7330 + [CaII]7324; the
[OII]3726,3729 doublet) are compared as a whole with the residuals in the same window:

    bright: [OII]3727, Hgamma, Hbeta, [OIII]4959,5007, [NII]6548, Halpha, [NII]6583,
            [SII]6716,6731, [SIII]9069,9531
    faint:  [OIII]4363, HeI4471, HeII4686, [NII]5755, HeI5876, [SIII]6312, HeI6678, HeI7065,
            [ArIII]7136, HeI7281, [OII]7320,7330, [ArIII]7751, OI8446, [ClII]8579,
            [FeII]8617, [CI]8727, P10, P9
    ([OI]6300,6363 are excluded: blended with the sky [OI] lines.)
Groups whose window is dominated by the sky (stacked SKY / stacked line model > sky_max = 5,
SFrame SKY extension) are not evaluated (status -2): sky-subtraction residuals, not the DAP
model, would dominate the net residual (e.g. [FeII]8617 in the OH / O2 (0-1) forest).

For each group the fibers with summed NP S/N >= ``min_snr`` (3) are selected; residual
r = data - joint model (NP: res_joint_spec; PM: org - mod_joint_spec_PEK; v1.15: minus a
straight line through its medians in the line-free pixels of the +-25 A margin, so that an
imperfect DAP continuum under the window does not count as line flux) and the line model
are shifted to the rest frame with the Halpha NP velocity (integer pixels) and summed:

    net = sum(r) / sum(model)   over the window:  > 0 flux missed, < 0 flux over-predicted

A group is 'poor' if |net| > net_warn (20%) and 'bad' if |net| > net_major (50%), both only
if |net| > nsig x its uncertainty; per-fiber residual sums are clipped at 5 NMAD. Groups with
fewer than ``min_fibers`` selected fibers, or whose stacked model is not > nsig sigma, are not
evaluated.
GOOD:      no bad group and <= 25% of the evaluated groups poor
WARNING:   >= 1 bad group, or > 25% of the evaluated groups poor
MAJOR:     >= 2 and >= 50% of the evaluated groups bad, overall or within one category
           (bright / faint; e.g. 7333: 11 of 14 faint groups bad, 12 of 25 overall)
UNDEFINED: no group evaluable
"""

from __future__ import annotations

import re

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register
from ._dapout import C_KMS, GAS_NP, GAS_PM, JOINT_PM, ORG, RES_NP, local_baseline, np_column, sframe_on, sigma_on

from ._lines import BRIGHT_GROUPS, FAINT_GROUPS

GROUPS = {**BRIGHT_GROUPS, **FAINT_GROUPS}
CATEGORY = {**{k: "bright" for k in BRIGHT_GROUPS}, **{k: "faint" for k in FAINT_GROUPS}}
MARGIN = 25.0
STATUS = {-2: "sky", -1: "n/a", 0: "good", 1: "poor", 2: "bad"}


def _wl(n):
    return float(n.rsplit("_", 1)[1])


def _values():
    v = [ValueSpec("model", "U4", "", "Emission-line model analysed (np / pm / auto)"),
         ValueSpec("n_groups", "i8", "", "Line groups evaluated"),
         ValueSpec("n_poor", "i8", "", "Evaluated groups flagged poor (or bad)"),
         ValueSpec("n_bad", "i8", "", "Evaluated groups flagged bad"),
         ValueSpec("n_bad_bright", "i8", "", "Bright-line groups flagged bad"),
         ValueSpec("n_bad_faint", "i8", "", "Faint-line groups flagged bad"),
         ValueSpec("worst_group", "U12", "", "Group with the largest |net|"),
         ValueSpec("max_abs_net", "f8", "", "Largest |net| over the evaluated groups"),
         ValueSpec("n_sky", "i8", "", "Groups not evaluated because the window is sky-dominated")]
    for k, (prim, win) in GROUPS.items():
        v += [ValueSpec(f"n_{k}", "i8", "", f"{k} ({CATEGORY[k]}): fibers with summed NP S/N >= min_snr"),
              ValueSpec(f"net_{k}", "f8", "", f"{k}: stacked residual / model flux in {win[0]:g}-{win[1]:g} A"),
              ValueSpec(f"e_net_{k}", "f8", "", f"{k}: uncertainty of net"),
              ValueSpec(f"nclip_{k}", "i8", "", f"{k}: selected fibers rejected (window, NaN or 5 NMAD residual clip)"),
              ValueSpec(f"sky_{k}", "f8", "", f"{k}: stacked sky / stacked line model in the window"),
              ValueSpec(f"status_{k}", "i8", "", f"{k}: -2 sky-dominated (not evaluated), -1 n/a, 0 good, 1 poor, 2 bad")]
    return tuple(v)


@register
class FaintLines(Metric):
    name = "qa_dap_npfluxes"
    slot = 17
    group = GROUP_DAP
    title = "DAP non-parametric line fluxes"
    description = ("Bias of the DAP emission-line fluxes (NP model by default) for bright and key "
                   "faint lines (b, r, z): stacked residual flux in the line window relative to the "
                   "modelled flux of all lines in the window (DAP output file)")
    requires = ("dap", "dap_output", "sframe_data")
    defaults = dict(
        groups=tuple(GROUPS),
        model="np",          # 'np', 'pm' or 'auto' (pm if the primary line is in PM_KEL)
        min_snr=3.0,         # summed NP S/N of the group for a fiber to enter the stack
        min_fibers=10,
        net_warn=0.2, net_major=0.5, nsig=3.0,
        clip=5.0,            # per-fiber residual-sum clipping (x NMAD)
        frac_poor_warn=0.25, frac_bad_major=0.5,
        sky_max=5.0,         # stacked sky / stacked line model in the window above -> not evaluated
        subtract_baseline=True,  # remove the residual continuum (line-free pixels around the window)
    )
    values = _values()
    main_value = "max_abs_net"

    def output_windows(self):
        return {f"fl_{k}": (GROUPS[k][1][0] - MARGIN, GROUPS[k][1][1] + MARGIN,
                            (ORG, RES_NP, GAS_NP, GAS_PM, JOINT_PM)) for k in self.params["groups"]}

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        net = "net = stacked residual / model flux of all lines in the window"
        return {
            "GOOD": f"No group with |{net}| > {p['net_major']:.0%}, <= {p['frac_poor_warn']:.0%} of the groups > {p['net_warn']:.0%}",
            "WARNING": (f">= 1 group with |net| > {p['net_major']:.0%}, or > {p['frac_poor_warn']:.0%} of the "
                        f"groups with |net| > {p['net_warn']:.0%} ({p['nsig']:g} sigma)"),
            "MAJOR ISSUE": (f">= 2 and >= {p['frac_bad_major']:.0%} of the evaluated groups (overall, or bright or faint "
                            f"alone) with |net| > {p['net_major']:.0%}"),
            "UNDEFINED": (f"No DAP / DAP output, or no line group with >= {p['min_fibers']} fibers of "
                          f"summed NP S/N >= {p['min_snr']:g}"),
        }

    def _np_names(self, ctx):
        names = []
        for ext in ("NP_ELINES_B", "NP_ELINES_R", "NP_ELINES_I"):
            try:
                t = ctx.dap_table(ext)
            except Exception:  # noqa: BLE001
                continue
            names += [c[5:] for c in t.colnames if c.startswith("flux_")]
        return names

    def _use_pm(self, ctx, line):
        mode = self.params["model"]
        if mode in ("np", "pm"):
            return mode == "pm"
        pm = ctx.dap_table("PM_KEL")
        return bool(np.any(np.abs(np.asarray(pm["wl"], float) - _wl(line)) < 0.01))

    def compute(self, ctx):
        p = self.params
        if not ctx.has_dap:
            return self.result(QAFlag.UNDEFINED, note="DAP file missing")
        if not ctx.has_dap_output:
            return self.result(QAFlag.UNDEFINED, note="DAP output file missing")
        vha = np_column(ctx, "vel_Halpha_6562.85")
        allnames = self._np_names(ctx)
        vals = {"model": p["model"]}
        status, nets = {}, {}
        for k in p["groups"]:
            prim, (w0, w1) = GROUPS[k]
            members = [n for n in allnames if w0 <= _wl(n) <= w1] or [prim]
            F = sum(np.nan_to_num(np_column(ctx, f"flux_{n}")) for n in members)
            EF = np.sqrt(sum(np.nan_to_num(np_column(ctx, f"e_flux_{n}")) ** 2 for n in members))
            v = np.where(np.isfinite(vha), vha, np_column(ctx, f"vel_{prim}"))
            with np.errstate(invalid="ignore", divide="ignore"):
                sel = np.where((EF > 0) & (F / EF >= p["min_snr"]) & np.isfinite(v))[0]
            vals[f"n_{k}"] = int(sel.size)
            status[k] = -1
            vals[f"status_{k}"] = -1
            if sel.size < p["min_fibers"]:
                continue
            wave, planes = ctx.dap_output(f"fl_{k}")
            sig = sigma_on(ctx, wave)
            sky = sframe_on(ctx, "SKY", wave)
            if self._use_pm(ctx, prim):
                r = planes[ORG][sel].astype(float) - planes[JOINT_PM][sel]
                g = planes[GAS_PM][sel].astype(float)
            else:
                r = planes[RES_NP][sel].astype(float)
                g = planes[GAS_NP][sel].astype(float)
            dw = float(wave[1] - wave[0])
            x = np.arange(int(np.ceil((w0 - wave[0]) / dw)), int(np.floor((w1 - wave[0]) / dw)) + 1)
            sh = np.round(0.5 * (w0 + w1) * v[sel] / C_KMS / dw).astype(int)
            idx = sh[:, None] + x[None, :]
            if p["subtract_baseline"]:
                # v1.15: residual continuum under the window (imperfect DAP continuum) removed: straight
                # line through the line-free pixels of the +-25 A margin
                win = np.zeros(r.shape, bool)
                np.put_along_axis(win, np.clip(idx, 0, wave.size - 1), True, axis=1)
                r, _ = local_baseline(r, wave, (g, planes[GAS_NP][sel], planes[GAS_PM][sel]), exclude=win)
            inside = (idx.min(axis=1) >= 0) & (idx.max(axis=1) < wave.size)
            idx = np.clip(idx, 0, wave.size - 1)
            rr = np.take_along_axis(r, idx, axis=1)
            gg = np.take_along_axis(g, idx, axis=1)
            ss = np.take_along_axis(sig[sel], idx, axis=1)
            good = inside & np.isfinite(rr).all(axis=1) & np.isfinite(ss).all(axis=1) & np.isfinite(gg).all(axis=1)
            # reject fibers with absurd residuals (corrupted spectra, e.g. a broken spectrograph):
            # 5 x NMAD clipping of the per-fiber residual sums
            Ri = rr.sum(axis=1)
            if good.sum() >= p["min_fibers"]:
                med = np.median(Ri[good])
                mad = 1.4826 * np.median(np.abs(Ri[good] - med))
                if mad > 0:
                    good &= np.abs(Ri - med) < p["clip"] * mad
            vals[f"nclip_{k}"] = int(sel.size - good.sum())
            if good.sum() < p["min_fibers"]:
                continue
            R, G, V = rr[good].sum(), gg[good].sum(), (ss[good] ** 2).sum()
            sk = np.take_along_axis(np.nan_to_num(sky[sel]), idx, axis=1)[good].sum()
            vals[f"sky_{k}"] = float(sk / G) if G > 0 else np.nan
            if G > 0 and sk / G > p["sky_max"]:
                # sky-subtraction residuals dominate the window (e.g. [FeII]8617 in the OH / O2
                # forest): the line flux cannot be verified
                status[k] = -2
                vals[f"status_{k}"] = -2
                continue
            if not G > p["nsig"] * np.sqrt(V):
                continue
            net, enet = float(R / G), float(np.sqrt(V) / G)
            st = 0
            if abs(net) > p["nsig"] * enet:
                st = 2 if abs(net) > p["net_major"] else 1 if abs(net) > p["net_warn"] else 0
            status[k] = st
            nets[k] = abs(net)
            vals.update({f"net_{k}": net, f"e_net_{k}": enet, f"status_{k}": st})
        ev = [k for k, s in status.items() if s >= 0]
        vals["n_sky"] = sum(s == -2 for s in status.values())
        n_poor = sum(status[k] >= 1 for k in ev)
        n_bad = sum(status[k] == 2 for k in ev)
        vals.update(n_groups=len(ev), n_poor=n_poor, n_bad=n_bad,
                    n_bad_bright=sum(status[k] == 2 for k in ev if CATEGORY[k] == "bright"),
                    n_bad_faint=sum(status[k] == 2 for k in ev if CATEGORY[k] == "faint"))
        if not ev:
            return self.result(QAFlag.UNDEFINED, note="no faint-line group detected in enough fibers", **vals)
        w = max(nets, key=nets.get)
        vals.update(worst_group=w, max_abs_net=nets[w])
        lst = ", ".join(f"{k} {vals[f'net_{k}']:+.2f}" for k in ev if status[k] > 0)
        major = n_bad >= 2 and n_bad / len(ev) >= p["frac_bad_major"]
        for cat in ("bright", "faint"):
            ec = [k for k in ev if CATEGORY[k] == cat]
            nb = sum(status[k] == 2 for k in ec)
            major |= bool(ec) and nb >= 2 and nb / len(ec) >= p["frac_bad_major"]
        if major:
            return self.result(QAFlag.MAJOR, note=lst, **vals)
        if n_bad >= 1 or n_poor / len(ev) > p["frac_poor_warn"]:
            return self.result(QAFlag.WARNING, note=lst, **vals)
        return self.result(QAFlag.GOOD, note=lst, **vals)
