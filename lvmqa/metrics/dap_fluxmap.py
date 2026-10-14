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


def _note(loc, n_ref, radius, s, n_rej, n_edges):
    return (f"local offset {loc:+.3f} dex ({10 ** loc:.2f}x) vs {int(n_ref)} exposures within {radius:g} deg "
            f"(s = {s:+.3f}), {int(n_rej)}/{int(n_edges)} overlaps rejected")


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
                    local_radius=0.97, local_min=3)
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
                     f"exposure relative to the exposures within {p['local_radius']:g} deg, from the overlap-graph "
                     f"solution; large mosaics: common IFU dipole fitted, no linear trend across the mosaic)"),
            "WARNING": f"|l| > {p['warn_dex']:g} dex, or > half of the overlaps rejected as inconsistent",
            "MAJOR ISSUE": f"|l| > {p['major_dex']:g} dex",
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
        from ..multitile.fluxmap import Node, build_edges, classify_node, local_offsets, solve
        from ..paths import dap_path
        p = self.params
        pa_of = {}
        if drpall is not None and "sci_pa" in drpall.colnames:
            pa_of = dict(zip(np.asarray(drpall["expnum"]).astype(int), np.asarray(drpall["sci_pa"], float)))
        nodes, own = [], {}
        for e, pl in payloads.items():
            if pl is None:
                continue
            own[e] = len(nodes)
            nodes.append(Node(str(e), *(np.asarray(pl[k], float) for k in ("ra", "dec", "flux", "eflux")),
                              pa=float(np.nan_to_num(pa_of.get(int(e), 0.0)))))
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
                near |= np.hypot((ra - r0) * np.cos(np.deg2rad(d0)), de - d0) <= rmax
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
                                  pa=float(np.nan_to_num(pa_of.get(int(row["expnum"]), 0.0)))))
                ctxn += 1
        edges = build_edges(nodes, match_radius=p["match_radius"], min_snr=p["min_snr"],
                            max_sep_deg=1.5 * p["spacing_deg"], min_match=p["min_match"])
        if not edges:
            for e in own:
                out[e] = self.result(QAFlag.UNDEFINED, note="no overlapping neighbour", n_nodes=len(nodes), n_edges=0)
            return out
        centers = np.array([n.center for n in nodes])
        sol = solve(len(nodes), edges, loss=p["loss"], centers=centers, pa=np.array([n.pa for n in nodes]),
                    pattern=p["pattern"], pattern_min_nodes=p["pattern_min_nodes"],
                    pattern_min_edges=p["pattern_min_edges"])
        A, w = sol["A"], sol["w"]
        n_edges = np.abs(A).sum(axis=0).astype(int)
        n_rej = (np.abs(A) * (w[:, None] < 0.5)).sum(axis=0).astype(int)
        loc, e_loc, n_ref = local_offsets(centers, sol["s"], sol["e_s"], n_edges > n_rej,
                                          p["local_radius"], p["local_min"])
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
                        e_local_dex=float(e_loc[i]), n_local=int(n_ref[i]), pattern=sol["pattern"],
                        dipole_dex=float(sol["dipole"]), dipole_pa=float(sol["dipole_pa"]))
            if n_edges[i] == 0:
                out[e] = self.result(QAFlag.UNDEFINED, note="no overlapping neighbour", **vals)
                continue
            lab = classify_node(loc[i], e_loc[i], n_edges[i], n_rej[i], p["warn_dex"], p["major_dex"], p["nsig"])
            flag = {"GOOD": QAFlag.GOOD, "WARNING": QAFlag.WARNING, "MAJOR ISSUE": QAFlag.MAJOR}[lab]
            note = "" if flag == QAFlag.GOOD else _note(loc[i], n_ref[i], p["local_radius"], s, n_rej[i], n_edges[i])
            out[e] = self.result(flag, note=note, **vals)
        return out
