"""
qa_fluxmap - prototype multi-exposure metric: consistency of the Halpha flux scale of an
exposure with the overlapping neighbouring tiles (see lvmqa/multitile/fluxmap.py).

Per exposure (workers) the DAP NP Halpha fluxes and fiber positions are collected. In the main
process an overlap graph is built from all processed exposures plus "context" exposures from
drpall within ``rings`` tile rings (centre distance <= (rings + 0.3) x ``spacing_deg``) whose DAP
files exist (they constrain the solution but get no flag here). Edge = robust mean of
log10(F_i/F_j) over fibers matched within 20" with S/N > 5; model d_ij = s_i - s_j (+ a common
IFU-internal dipole, see below), solved by L1 start + Tukey IRLS (no node assumed valid).

v1.13: on the Gum mosaic (545 exposures) the edge offsets depend on the overlap direction
(62% of their variance: +-0.06 dex towards IFU PA ~150 deg, i.e. a ~0.12 dex flux difference
between opposite IFU edges). Summed over the ~25 tiles across the mosaic, this bias turned the
solution into a +-0.7 dex linear ramp (442 MAJOR). Now:
  * pattern='auto': for graphs with >= 50 edges and a component of >= 30 nodes, a common dipole
    a cos(theta) + b sin(theta) of the overlap direction theta (IFU frame) is fitted, together
    with a "no linear trend of s" gauge on the large components (the dipole and a linear ramp of
    s are exactly degenerate). Reported per exposure as dipole_dex / dipole_pa (run-level value).
  * the flag uses the LOCAL offset l = s - median(s of the connected exposures within
    local_radius = 0.97 deg, ~2 tile rings), so residual large-scale structure of s cancels,
    while single tiles and small groups of offset tiles remain.
GOOD     |l| <= 0.04 dex (10%) or not significant (< 3 sigma)
WARNING  |l| > 0.04 dex, or more than half of the node's edges rejected
MAJOR    |l| > 0.10 dex (26%)
UNDEFINED  no DAP file, no overlapping neighbour (or all edges rejected -> WARNING)
Gum test: old 442 M / 62 W / 41 G -> local offsets: 5 tiles > 0.10 dex, 58 > 0.04 dex.

v1.15 (Milky-Way run, DRP 1.2.1, 5700 exposures: 17% W/M, dark tiles without edges):
  * faint overlaps: if < min_match fiber pairs have S/N > 5 in BOTH exposures (faint fields, or
    an exposure that is much too faint and fails the per-fiber cut), the edge is measured from
    the ratio of the summed fluxes of pairs selected on their summed S/N (> 3; clipped, bootstrap
    error) -> dark / failed tiles get edges and are flagged instead of UNDEFINED.
  * local reference = exposures of OTHER nights within local_radius (x1.5, x2.5 until
    >= local_min): whole nights with a common calibration offset (seen in the MW run: nightly
    medians up to -0.15...-0.27 dex) are compared with independent data, not with themselves.
    night_dex = median local offset of the exposures of the night in the graph.
  * two passes: the MAJOR exposures of the first pass are not used as anchors in the second,
    which uses a larger reference radius (local_radius2 = 1.5 deg).
  * significance: the formal errors of l underestimate its real scatter (MW: NMAD(l/e_l) = 2.9);
    the errors are scaled by this run-level kappa (error_scale='auto', >= 1). No external prior
    (no Moon / sky information); an optional additive term (add_sigma, off by default) exists.
  * faint edges: the brighter summed side is the denominator, so exposures with ~zero or negative
    Halpha (over-subtracted sky) get strongly negative edges (floored at -1.5 dex) instead of none.
  * additive vs multiplicative: per exposure F = g F_ref + c is fitted to all pooled overlap
    pairs (neighbours on the common scale). In the MW run |l| grows towards faint fields and
    under a bright moon (NMAD of l 0.012 dex in bright fields, dark sky; 0.057 dex in faint fields,
    moon up with FLI > 0.8: 34% flagged, median l -0.03): an additive Halpha residual (sky
    subtraction) rather than a flux-scale error. If c explains the offset and the flux scale g is
    consistent with the local reference, the flag is capped at WARNING (cause 'additive').
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_DAP, Metric, ValueSpec, register

log = logging.getLogger("lvmqa")
LINE = "Halpha_6562.85"


def _payload_from_dap(path) -> dict:
    from astropy.table import Table, join
    pt = Table.read(path, "PT")
    r = Table.read(path, "NP_ELINES_R")[["id", f"flux_{LINE}", f"e_flux_{LINE}"]]
    t = join(pt["id", "ra", "dec"], r, keys="id")
    # join sorts by the id STRING ("e.10" < "e.100" < "e.3"): restore the PT (fiber) order
    pos = {str(k): i for i, k in enumerate(pt["id"])}
    t = t[np.argsort([pos[str(k)] for k in t["id"]], kind="stable")]
    return dict(ra=np.asarray(t["ra"], np.float32), dec=np.asarray(t["dec"], np.float32),
                flux=np.asarray(t[f"flux_{LINE}"], np.float32), eflux=np.asarray(t[f"e_flux_{LINE}"], np.float32))


# ----------------------------------------------------------------------------- Halpha map cache
# Per-exposure Halpha maps (ra, dec, flux, eflux; ~30 KB) cached in <cache_dir>/halpha_maps/,
# written whenever a DAP file is read for them (pipeline workers, context tiles, diagnostics PDF)
# and validated against the size and mtime of the DAP file. Saves re-reading the DAP files for
# the context tiles of later runs and for the fluxmap page of the diagnostics PDF.
_MAP_KEYS = ("ra", "dec", "flux", "eflux")


def has_halpha_map(cfg, tileid, mjd, expnum) -> bool:
    """DAP file or cached Halpha map available."""
    from ..paths import dap_path
    return Path(dap_path(cfg, int(tileid), int(mjd), int(expnum))).is_file() or _cache_file(cfg, expnum).is_file()


def _cache_file(cfg, expnum) -> Path:
    return Path(cfg.cache_dir) / "halpha_maps" / f"{cfg.drp_version}_{cfg.dap_label}" / f"{int(expnum):08d}.npz"


def save_halpha_map(cfg, expnum, payload, src) -> None:
    """Store a Halpha map in the cache (atomic write; failures are only logged)."""
    try:
        st = Path(src).stat()
        cf = _cache_file(cfg, expnum)
        cf.parent.mkdir(parents=True, exist_ok=True)
        tmp = cf.with_name(f".{cf.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, src_size=st.st_size, src_mtime=st.st_mtime,
                 **{k: np.asarray(payload[k], np.float32) for k in _MAP_KEYS})
        os.replace(tmp, cf)
    except Exception as exc:  # noqa: BLE001 - the cache is optional
        log.debug("qa_fluxmap: cannot cache Halpha map of %s: %s", expnum, exc)


def halpha_map(cfg, tileid, mjd, expnum, path=None):
    """Halpha map {ra, dec, flux, eflux} of an exposure: from the cache if it is up to date,
    otherwise from the DAP file (then cached). None if the DAP file does not exist."""
    from ..paths import dap_path
    path = Path(path) if path is not None else Path(dap_path(cfg, int(tileid), int(mjd), int(expnum)))
    cf = _cache_file(cfg, expnum)
    try:
        st = path.stat()
    except OSError:
        # DAP file not available (e.g. inspecting a server run on a laptop): use the cached map as is
        if cf.is_file():
            try:
                with np.load(cf) as z:
                    return {k: np.array(z[k]) for k in _MAP_KEYS}
            except Exception:  # noqa: BLE001
                return None
        return None
    if cf.is_file():
        try:
            with np.load(cf) as z:
                if int(z["src_size"]) == st.st_size and float(z["src_mtime"]) == st.st_mtime:
                    return {k: np.array(z[k]) for k in _MAP_KEYS}
        except Exception:  # noqa: BLE001 - corrupt / partial cache file: re-read
            pass
    pl = _payload_from_dap(path)
    save_halpha_map(cfg, expnum, pl, path)
    return pl


def _halpha_map_job(args):
    """Process-pool helper: (cfg, tileid, mjd, expnum) -> (expnum, map or None)."""
    import warnings
    warnings.simplefilter("ignore")
    cfg, tileid, mjd, expnum = args
    try:
        return int(expnum), halpha_map(cfg, tileid, mjd, expnum)
    except Exception as exc:  # noqa: BLE001
        log.warning("qa_fluxmap: cannot read Halpha map of %s: %s", expnum, exc)
        return int(expnum), None


def _note(loc, n_ref, radius, s, n_rej, n_edges, cause="", night=np.nan, add_frac=np.nan, mult_rel=np.nan):
    nt = (f"local offset {loc:+.3f} dex ({10 ** loc:.2f}x) vs {int(n_ref)} exposures within {radius:.2g} deg "
          f"(s = {s:+.3f}), {int(n_rej)}/{int(n_edges)} overlaps rejected")
    if np.isfinite(night):
        nt += f"; night median {night:+.3f}"
    if cause == "additive":
        nt += (f"; additive Halpha offset ({add_frac:+.0%} of the median flux), flux scale consistent "
               f"({mult_rel:+.3f} dex): at most WARNING")
    elif cause == "edges":
        nt += "; most overlaps inconsistent"
    return nt


@register
class FluxMap(Metric):
    name = "qa_fluxmap"
    slot = 19
    multi = True
    group = GROUP_DAP
    title = "Multi-tile Halpha flux consistency"
    description = ("Prototype: flux-scale offset of the exposure relative to overlapping neighbouring "
                   "tiles from the DAP Halpha maps (robust overlap-graph solution)")
    requires = ("dap", "drpall")
    defaults = dict(rings=2, spacing_deg=0.42, match_radius=20.0, min_snr=5.0, min_match=10,
                    warn_dex=0.04, major_dex=0.10, nsig=3.0, loss="tukey", max_context=300,
                    pattern="auto", pattern_min_nodes=30, pattern_min_edges=50,
                    local_radius=0.97, local_min=4,
                    faint_edges=True,       # v1.15: ratio-of-sums edges for faint overlaps / dark tiles
                    other_nights=True,      # v1.15: local reference from other nights only (radius grows)
                    soften_additive=True,   # v1.15: offset explained by an additive term -> at most WARNING
                    soften_max_dex=0.2,     # ... only for |l| <= soften_max_dex
                    dense_max=1500,         # v1.15: larger graphs -> sparse CG solver
                    add_sigma=0.0,          # v1.15: additive Halpha uncertainty (flux/fiber) in the significance ('auto'); off
                    error_scale="auto",     # v1.15: kappa = NMAD(l / e_l) of the run (>= 1) scales the errors
                    iterate=True,           # v1.15: second pass without the first-pass MAJOR anchors ...
                    local_radius2=1.5)      # ... and with this reference radius [deg]
    values = (
        ValueSpec("s_dex", "f8", "", "Flux-scale offset log10(F / F_neighbours) [dex]"),
        ValueSpec("e_s_dex", "f8", "", "Uncertainty of s_dex [dex]"),
        ValueSpec("n_edges", "i8", "", "Overlapping neighbours (edges)"),
        ValueSpec("n_rejected", "i8", "", "Edges rejected by the robust solution"),
        ValueSpec("n_nodes", "i8", "", "Exposures in the overlap graph (incl. context exposures)"),
        ValueSpec("sigma_sys", "f8", "", "Edge scatter of the solution (NMAD of residuals) [dex]"),
        ValueSpec("local_dex", "f8", "", "Local offset: s - median(s of the exposures within local_radius) [dex]"),
        ValueSpec("e_local_dex", "f8", "", "Uncertainty of local_dex [dex]"),
        ValueSpec("n_local", "i8", "", "Exposures in the local reference"),
        ValueSpec("local_radius_used", "f8", "deg", "Radius of the local reference (grows until local_min exposures)"),
        ValueSpec("night_dex", "f8", "", "Median local offset of the exposures of the same night in the graph [dex]"),
        ValueSpec("n_night", "i8", "", "Exposures of the same night in the graph"),
        ValueSpec("n_faint_edges", "i8", "", "Edges measured with the faint (ratio-of-sums) estimator"),
        ValueSpec("mult_dex", "f8", "", "Flux scale log10 g from F = g F_ref + c on the pooled overlaps [dex]"),
        ValueSpec("e_mult_dex", "f8", "", "Uncertainty of mult_dex"),
        ValueSpec("add_flux", "f8", "", "Additive term c from F = g F_ref + c (Halpha per fiber, 1e-16 erg/s/cm2)"),
        ValueSpec("e_add_flux", "f8", "", "Uncertainty of add_flux (1e-16 erg/s/cm2)"),
        ValueSpec("add_frac", "f8", "", "add_flux / median Halpha flux of the overlap fibers"),
        ValueSpec("ha_med", "f8", "", "Median Halpha flux of the exposure's fibers (1e-16 erg/s/cm2)"),
        ValueSpec("add_sigma", "f8", "", "Run-level additive Halpha uncertainty sigma_a per fiber (1e-16 erg/s/cm2)"),
        ValueSpec("e_local_tot_dex", "f8", "", "Error of local_dex used for the significance: error_scale x hypot(e_local_dex, sigma_a term)"),
        ValueSpec("error_scale", "f8", "", "Run-level error scale kappa = NMAD(local_dex / e_local_dex) (>= 1)"),
        ValueSpec("cause", "U8", "", "Cause of a flag: scale (flux scale), additive (offset c), edges (rejected overlaps)"),
        ValueSpec("pattern", "U8", "", "IFU-internal pattern fitted (dipole / none)"),
        ValueSpec("dipole_dex", "f8", "", "Common IFU dipole amplitude (edge offset towards dipole_pa) [dex] (run value)"),
        ValueSpec("dipole_pa", "f8", "", "Direction of the brighter IFU side, IFU frame (deg E of N + sci_pa) (run value)"),
    )
    main_value = "local_dex"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"|l| <= {p['warn_dex']:g} dex or < {p['nsig']:g} sigma (l = log flux-scale offset of the "
                     f"exposure relative to the exposures of other nights within {p['local_radius']:g} deg (grown "
                     f"if needed), from the overlap-graph "
                     f"solution; large mosaics: common IFU dipole fitted, no linear trend across the mosaic)"),
            "WARNING": f"|l| > {p['warn_dex']:g} dex, or > half of the overlaps rejected as inconsistent",
            "MAJOR ISSUE": (f"|l| > {p['major_dex']:g} dex (capped at WARNING if an additive Halpha term explains the "
                            f"offset and the flux scale is consistent)"),
            "UNDEFINED": "No DAP file or no overlapping neighbour tile",
        }

    def collect(self, ctx):
        if not ctx.has_dap:
            return None
        pt = ctx.dap_table("PT")
        npc = ctx.dap_np_columns([f"flux_{LINE}", f"e_flux_{LINE}"])
        pl = dict(ra=np.asarray(pt["ra"], np.float32), dec=np.asarray(pt["dec"], np.float32),
                  flux=np.asarray(npc[f"flux_{LINE}"], np.float32),
                  eflux=np.asarray(npc[f"e_flux_{LINE}"], np.float32))
        save_halpha_map(ctx.cfg, ctx.expnum, pl, ctx.dap_file)
        return pl

    def compute_multi(self, payloads, cfg, drpall, shared=None):
        from ..multitile.fluxmap import (Node, additive_sigma, build_edges, classify_node, decompose,
                                         local_offsets, night_offsets, solve)
        from ..paths import dap_path
        p = self.params
        pa_of, mjd_of = {}, {}
        if drpall is not None:
            ex = np.asarray(drpall["expnum"]).astype(int)
            if "sci_pa" in drpall.colnames:
                pa_of = dict(zip(ex, np.asarray(drpall["sci_pa"], float)))
            if "mjd" in drpall.colnames:
                mjd_of = dict(zip(ex, np.asarray(drpall["mjd"]).astype(int)))
        nodes, own = [], {}
        for e, pl in payloads.items():
            if pl is None:
                continue
            own[e] = len(nodes)
            nodes.append(Node(str(e), *(np.asarray(pl[k], float) for k in ("ra", "dec", "flux", "eflux")),
                              pa=float(np.nan_to_num(pa_of.get(int(e), 0.0))), mjd=int(mjd_of.get(int(e), -1))))
        out = {e: self.result(QAFlag.UNDEFINED, note="no DAP file") for e in payloads}
        if not nodes:
            return out
        # context exposures from drpall around the processed tiles
        ctxn = 0
        if drpall is not None and "sci_ra" in drpall.colnames:
            cen = np.array([n.center for n in nodes])
            ra, de = np.asarray(drpall["sci_ra"], float), np.asarray(drpall["sci_dec"], float)
            rmax = (p["rings"] + 0.3) * p["spacing_deg"]
            near = np.zeros(len(drpall), bool)
            for r0, d0 in cen:
                near |= np.hypot(((ra - r0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(d0)), de - d0) <= rmax
            cand = [i for i in np.where(near)[0] if int(drpall["expnum"][i]) not in payloads]
            for i in cand[:p["max_context"]]:
                row = drpall[i]
                f = dap_path(cfg, int(row["tileid"]), int(row["mjd"]), int(row["expnum"]))
                if not Path(f).is_file():
                    continue
                try:
                    pl = halpha_map(cfg, int(row["tileid"]), int(row["mjd"]), int(row["expnum"]), f)
                except Exception as exc:  # noqa: BLE001
                    log.warning("qa_fluxmap: cannot read context DAP %s: %s", f, exc)
                    continue
                nodes.append(Node(f"ctx{int(row['expnum'])}", *(np.asarray(pl[k], float)
                                                                  for k in ("ra", "dec", "flux", "eflux")),
                                  pa=float(np.nan_to_num(pa_of.get(int(row["expnum"]), 0.0))),
                                  mjd=int(row["mjd"])))
                ctxn += 1
        edges = build_edges(nodes, match_radius=p["match_radius"], min_snr=p["min_snr"],
                            max_sep_deg=1.5 * p["spacing_deg"], min_match=p["min_match"], faint=p["faint_edges"])
        if not edges:
            for e in own:
                out[e] = self.result(QAFlag.UNDEFINED, note="no overlapping neighbour", n_nodes=len(nodes), n_edges=0)
            return out
        centers = np.array([n.center for n in nodes])
        sol = solve(len(nodes), edges, loss=p["loss"], centers=centers, pa=np.array([n.pa for n in nodes]),
                    pattern=p["pattern"], pattern_min_nodes=p["pattern_min_nodes"],
                    pattern_min_edges=p["pattern_min_edges"], dense_max=p["dense_max"])
        w, I, J = sol["w"], sol["I"], sol["J"]
        nn_ = len(nodes)
        n_edges = (np.bincount(I, minlength=nn_) + np.bincount(J, minlength=nn_)).astype(int)
        rj = (w < 0.5).astype(float)
        n_rej = (np.bincount(I, rj, nn_) + np.bincount(J, rj, nn_)).astype(int)
        fe = np.array([e.faint for e in edges], float)
        n_faint = (np.bincount(I, fe, nn_) + np.bincount(J, fe, nn_)).astype(int)
        grp = np.array([n.mjd for n in nodes]) if p["other_nights"] else None
        conn = n_edges > n_rej
        dec = decompose(nodes, edges, sol["s"], w, sol.get("pattern_term"))
        with np.errstate(all="ignore"):
            fmed = np.array([np.nanmedian(n.flux) if np.isfinite(n.flux).any() else np.nan for n in nodes])
        own_i = np.array(sorted(own.values()), int)
        own_m = np.zeros(len(nodes), bool)
        own_m[own_i] = True

        def nmad(x):
            x = x[np.isfinite(x)]
            return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if x.size else np.nan

        def significance(loc, e_loc, sel):
            """error used for the flag: kappa x hypot(e_l, additive term)."""
            sa = p["add_sigma"]
            sa = additive_sigma(loc, fmed, sel) if sa == "auto" else float(sa)
            with np.errstate(all="ignore"):
                # reference brightness = what the neighbours see (F 10^-l): an additive offset c gives
                # l = log10(1 + c / F_ref); a too-faint (failed) exposure is not excused by its own low flux
                fref = fmed * 10 ** (-np.clip(loc, -3, 3))
                e = np.hypot(e_loc, np.where(fref > 0, sa / (np.log(10) * fref), np.inf) if sa > 0 else 0.0)
            kap = p["error_scale"]
            if kap == "auto":    # the formal errors underestimate the real scatter of l (MW: 2.9x)
                with np.errstate(all="ignore"):
                    kap = max(1.0, nmad((loc / e)[sel])) if np.isfinite(loc[sel] / e[sel]).sum() >= 50 else 1.0
            return float(kap) * e, float(kap), sa

        def major(loc, e_tot, i):
            mr = dec["mult_dex"][i] - (sol["s"][i] - loc[i])
            return classify_node(loc[i], e_tot[i], n_edges[i], n_rej[i], p["warn_dex"], p["major_dex"], p["nsig"],
                                 mr, dec["e_mult_dex"][i], dec["add"][i], dec["e_add"][i], p["soften_additive"],
                                 soften_max_dex=p["soften_max_dex"]) == "MAJOR ISSUE"

        loc, e_loc, n_ref, rad = local_offsets(centers, sol["s"], sol["e_s"], conn, p["local_radius"], p["local_min"],
                                               group=grp)
        e_tot, kap, sa = significance(loc, e_loc, conn & own_m)
        n_excl = 0
        if p["iterate"]:
            # second pass: the MAJOR exposures of the first pass are not used as anchors, and the
            # reference radius is larger (more independent references)
            bad = conn & np.array([major(loc, e_tot, i) for i in range(len(nodes))])
            n_excl = int(bad.sum())
            loc, e_loc, n_ref, rad = local_offsets(centers, sol["s"], sol["e_s"], conn, p["local_radius2"],
                                                   p["local_min"], group=grp, ref_ok=conn & ~bad)
            e_tot, kap, sa = significance(loc, e_loc, conn & own_m & ~bad)
        night, n_night = night_offsets(np.array([n.mjd for n in nodes]), loc, conn)
        log.info("qa_fluxmap: error scale kappa = %.2f, additive sigma_a = %.1f, %d first-pass MAJOR anchors excluded",
                 kap, sa, n_excl)
        log.info("qa_fluxmap: %d nodes (%d context), %d edges (%d faint), sigma_sys %.3f dex", len(nodes), ctxn,
                 len(edges), int(fe.sum()), sol["sigma_sys"])
        if sol["pattern"] == "dipole":
            lvl = log.warning if sol["dipole"] > p["warn_dex"] else log.info
            lvl("qa_fluxmap: common IFU flux pattern (dipole) %.3f dex towards IFU PA %.0f deg (%d nodes, %d edges)",
                sol["dipole"], sol["dipole_pa"], len(nodes), len(edges))
        if shared is not None:   # for qa_gradient: common IFU dipole of the mosaic each exposure belongs to
            shared[self.name] = {e: dict(member=bool(sol["pattern_member"][i]), dipole=float(sol["dipole"]),
                                         e_dipole=float(sol["e_dipole"]), dipole_pa=float(sol["dipole_pa"]),
                                         n_nodes=len(nodes), n_edges=len(edges))
                                 for e, i in own.items()}
        for e, i in own.items():
            s, es = float(sol["s"][i]), float(sol["e_s"][i])
            vals = dict(s_dex=s, e_s_dex=es, n_edges=int(n_edges[i]), n_rejected=int(n_rej[i]),
                        n_nodes=len(nodes), sigma_sys=float(sol["sigma_sys"]), local_dex=float(loc[i]),
                        e_local_dex=float(e_loc[i]), n_local=int(n_ref[i]), local_radius_used=float(rad[i]),
                        night_dex=float(night[i]), n_night=int(n_night[i]), n_faint_edges=int(n_faint[i]),
                        mult_dex=float(dec["mult_dex"][i]), e_mult_dex=float(dec["e_mult_dex"][i]),
                        add_flux=float(dec["add"][i]), e_add_flux=float(dec["e_add"][i]),
                        add_frac=float(dec["add"][i] / fmed[i]) if fmed[i] > 0 else np.nan, ha_med=float(fmed[i]),
                        add_sigma=float(sa), e_local_tot_dex=float(e_tot[i]), error_scale=kap,
                        pattern=sol["pattern"], dipole_dex=float(sol["dipole"]), dipole_pa=float(sol["dipole_pa"]))
            if n_edges[i] == 0:
                out[e] = self.result(QAFlag.UNDEFINED, note="no overlapping neighbour", cause="", **vals)
                continue
            mult_rel = vals["mult_dex"] - (s - vals["local_dex"])
            lab, cause = classify_node(loc[i], e_tot[i], n_edges[i], n_rej[i], p["warn_dex"], p["major_dex"], p["nsig"],
                                       mult_rel, vals["e_mult_dex"], vals["add_flux"], vals["e_add_flux"],
                                       p["soften_additive"], with_cause=True, soften_max_dex=p["soften_max_dex"])
            flag = {"GOOD": QAFlag.GOOD, "WARNING": QAFlag.WARNING, "MAJOR ISSUE": QAFlag.MAJOR}[lab]
            note = "" if flag == QAFlag.GOOD else _note(loc[i], n_ref[i], rad[i], s, n_rej[i], n_edges[i], cause,
                                                        night[i] if n_night[i] >= 3 else np.nan, vals["add_frac"],
                                                        mult_rel)
            out[e] = self.result(flag, note=note, cause=cause, **vals)
        return out
