"""
Pipeline driver: select exposures from drpall, run all metrics per exposure, assemble
the output table, write FITS / XLSX / HTML.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from astropy.table import Table

from . import __version__
from .bitmask import FLAG_LABELS, QAFlag, combine, pack
from .config import PipelineConfig
from .context import ExposureContext
from .registry import Metric, MetricResult, build_metrics, load_metric_module

log = logging.getLogger("lvmqa")

# drpall columns copied to the output: identification (main + supl files) and
# observational context (supl file only)
MAIN_ID_COLUMNS = ("expnum", "tileid", "mjd", "object", "group")   # group: exposure file (v1.16), default "all"
ID_COLUMNS = MAIN_ID_COLUMNS + ("exptime", "obstime", "dpos", "sci_ra", "sci_dec", "sci_pa",
                                "sci_amass", "fluxcal")
IO_KEYS = ("io_sframe", "io_dap", "io_dap_output")
# computations shared between metrics (ExposureContext.shared): key -> description
SHARED_KEYS = {"shared_skylines": "Shared sky-line Gaussian fits (qa_wavecal_*, qa_lsf)"}


@dataclass
class QAResult:
    table: Table
    metrics: List[Metric]
    config: PipelineConfig
    files: Dict[str, Path] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    nproc: int = 1

    @property
    def slots(self) -> Dict[str, int]:
        return {m.name: m.slot for m in self.metrics}


# ----------------------------------------------------------------------------- selection
def load_drpall(cfg: PipelineConfig) -> Table:
    path = Path(cfg.drpall_file)
    if not path.exists():
        raise FileNotFoundError(f"drpall file not found: {path}")
    t = Table.read(path, hdu=1)
    for c in t.colnames:  # strip padding of FITS strings
        if t[c].dtype.kind in "SU":
            t[c] = np.char.strip(np.asarray(t[c]).astype(str))
    return t


# drpall column <- SFrame primary-header keyword(s) (lvmdrp.utils.metadata._collect_header_data, plus
# the raw-frame keywords for tileid / mjd / expnum / exptime). Used without a drpall file (v1.15).
HEADER_COLUMNS = {
    "expnum": ("EXPOSURE",), "tileid": ("TILE_ID", "TILEID"), "mjd": ("MJD", "SMJD"), "exptime": ("EXPTIME",),
    "object": ("OBJECT",), "obstime": ("OBSTIME",), "dpos": ("DPOS",), "drpver": ("DRPVER",),
    "drpqual": ("DRPQUAL",), "fluxcal": ("FLUXCAL",),
    "sci_ra": ("SCIRA", "POSCIRA"), "sci_dec": ("SCIDEC", "POSCIDE"), "sci_pa": ("SCIPA", "POSCIPA"),
    "sci_amass": ("SCIAM",), "sci_alt": ("SKY SCI_ALT",), "sci_sh_hght": ("SKY SCI_SH_HGHT",),
    "sci_moon_sep": ("SKY SCI_MOON_SEP",), "moon_ra": ("SKY MOON_RA",), "moon_dec": ("SKY MOON_DEC",),
    "moon_phase": ("SKY MOON_PHASE",), "moon_fli": ("SKY MOON_FLI",), "sun_alt": ("SKY SUN_ALT",),
    "moon_alt": ("SKY MOON_ALT",),
    **{f"{k}_{s}_sen{c}": (f"{k.upper()}SEN{s[0].upper()}{c.upper()}",)
       for k in ("std", "sci", "mod") for s in ("mean", "rms") for c in "brz"},
}


MAX_HEADER_FILES = 500   # without --force, at most this many SFrame headers are read (no drpall)


class TooManyFilesError(RuntimeError):
    """Raised before reading more SFrame headers than allowed (exposure table without drpall)."""


def exposures_from_sframes(cfg: PipelineConfig, expnums: Optional[Iterable[int]] = None,
                           nthreads: int = 16, max_files: Optional[int] = None) -> Table:
    """Exposure table (drpall-like columns) from the SFrame primary headers of all files matching
    cfg.sframe_pattern (or the standard DRP tree). Only the requested expnums are read.
    max_files: refuse (RuntimeError) if more headers would be read (None / 0: no limit)."""
    from concurrent.futures import ThreadPoolExecutor
    from astropy.io import fits
    from .paths import default_sframe_pattern, sframe_index, tile_group
    pattern = cfg.sframe_pattern or default_sframe_pattern(cfg)
    if not cfg.sframe_pattern:
        log.warning("no drpall and no SFrame pattern: listing the whole standard DRP tree %s (can be slow)",
                    pattern)
    t0 = time.time()
    idx = sframe_index(pattern)
    exps = sorted(idx)
    if expnums:
        want = set(int(e) for e in expnums)
        exps = [e for e in exps if e in want]
    log.info("SFrames matching %s: %d (%d selected by expnum) in %.1f s; %d header(s) to read", pattern,
             len(idx), len(exps), time.time() - t0, len(exps))
    if max_files and len(exps) > max_files:
        raise TooManyFilesError(
            f"no drpall: {len(exps)} SFrame headers would be read from {pattern} (limit {max_files}). "
            "Check the SFrame pattern / quoting, narrow the selection (--expnums, --exposure-file), "
            "or use --force (or --max-header-files N) to proceed.")

    def row(e):
        try:
            h = fits.getheader(idx[e], 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read the header of %s: %s", idx[e], exc)
            h = {}
        r = {}
        for col, keys in HEADER_COLUMNS.items():
            v = None
            for k in keys:
                try:
                    v = h.get(k)
                except Exception:  # noqa: BLE001
                    v = None
                if v is not None:
                    break
            r[col] = v
        r["expnum"] = int(r["expnum"]) if r["expnum"] is not None else int(e)
        if int(r["expnum"]) != int(e):
            log.warning("%s: EXPOSURE %s in the header, %d in the file name (file name used)", idx[e], r["expnum"], e)
            r["expnum"] = int(e)
        return r

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(int(nthreads), 64))) as ex:
        rows = list(ex.map(row, exps))
    cols = {}
    for c in HEADER_COLUMNS:
        vals = [r[c] for r in rows]
        if c in ("expnum", "tileid", "mjd"):
            cols[c] = np.array([int(v) if v is not None else -1 for v in vals], dtype=np.int64)
        elif all(v is None or isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                 for v in vals):
            cols[c] = np.array([np.nan if v is None else float(v) for v in vals])
        else:
            cols[c] = np.array(["" if v is None else str(v).strip() for v in vals])
    t = Table(cols) if rows else Table({c: np.zeros(0, dtype=np.int64 if c in ("expnum", "tileid", "mjd") else float)
                                        for c in HEADER_COLUMNS})
    t["tilegrp"] = [tile_group(int(x)) if int(x) >= 0 else "" for x in t["tileid"]] if len(t) else np.zeros(0, "U6")
    bad = (np.asarray(t["tileid"]) < 0) | (np.asarray(t["mjd"]) < 0) if len(t) else np.zeros(0, bool)
    if bad.any():
        log.warning("%d SFrame(s) without TILE_ID / MJD in the header", int(bad.sum()))
    log.info("exposure table from %d SFrame headers in %.1f s (no drpall)", len(t), time.time() - t0)
    return t


def exposure_table(cfg: PipelineConfig, expnums: Optional[Iterable[int]] = None, nthreads: int = 16,
                   max_files: Optional[int] = None) -> Table:
    """drpall, or (cfg.no_drpall, or no drpall file but an SFrame pattern) the table from the SFrame
    headers. With an SFrame pattern and a drpall, only exposures with a matching file are kept."""
    if cfg.no_drpall or (cfg.sframe_pattern and not Path(cfg.drpall_file).exists()):
        if not cfg.no_drpall:
            log.warning("drpall %s not found: exposure table from the SFrame headers", cfg.drpall_file)
        return exposures_from_sframes(cfg, expnums, nthreads, max_files=max_files)
    t = load_drpall(cfg)
    if cfg.sframe_pattern:
        from .paths import sframe_index
        have = set(sframe_index(cfg.sframe_pattern))
        t = t[np.isin(np.asarray(t["expnum"]).astype(int), list(have))]
    return t


def select_exposures(drpall: Table, expnums: Optional[Iterable[int]] = None,
                     mjds: Optional[Iterable[int]] = None,
                     tileids: Optional[Iterable[int]] = None) -> Table:
    """Logical AND of the given filters (None = no constraint)."""
    sel = np.ones(len(drpall), dtype=bool)
    for col, vals in (("expnum", expnums), ("mjd", mjds), ("tileid", tileids)):
        if vals:
            sel &= np.isin(np.asarray(drpall[col]), np.asarray(list(vals), dtype=int))
    out = drpall[sel]
    out.sort("expnum")
    if expnums:
        missing = sorted(set(int(e) for e in expnums) - set(int(e) for e in out["expnum"]))
        if missing:
            log.warning("expnum(s) not in drpall (or excluded by other filters): %s",
                        missing[:20] if len(missing) > 20 else missing)
    return out


def assign_groups(rows: Table, spec=None) -> None:
    """Column ``group`` from an exposure file spec {expnum: (mjd, group)}; "all" otherwise."""
    from .expfile import DEFAULT_GROUP, check_mjds
    if spec:
        check_mjds(spec, rows)
    g = [spec.get(int(e), (0, DEFAULT_GROUP))[1] if spec else DEFAULT_GROUP for e in rows["expnum"]]
    rows["group"] = np.asarray(g, dtype=f"U{max([len(x) for x in g] + [3])}")
    rows["group"].description = "Exposure group (exposure file; default all)"


def group_counts_text(rows: Table) -> str:
    from collections import Counter
    if "group" not in rows.colnames:
        return ""
    return ", ".join(f"{g}: {n}" for g, n in sorted(Counter(str(x) for x in rows["group"]).items()))


def sframes_exist(cfg: PipelineConfig, rows: Table, nthreads: int = 16) -> np.ndarray:
    """Boolean mask: SFrame file on disk, one stat per exposure (no DAP lookup), run in threads:
    on network file systems the latency of each stat, not the CPU, dominates."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from .paths import sframe_path, tile_group
    tg = np.asarray(rows["tilegrp"]).astype(str) if "tilegrp" in rows.colnames else None
    paths = []
    for k, (t, m, e) in enumerate(zip(np.asarray(rows["tileid"]), np.asarray(rows["mjd"]), np.asarray(rows["expnum"]))):
        g = tg[k].strip() if tg is not None and tg[k].strip() else tile_group(int(t))
        paths.append(str(sframe_path(cfg, int(t), int(m), int(e), g)))
    if len(paths) < 50 or nthreads <= 1:
        return np.array([os.path.isfile(f) for f in paths], dtype=bool)
    with ThreadPoolExecutor(max_workers=min(int(nthreads), 64)) as ex:
        return np.fromiter(ex.map(os.path.isfile, paths, chunksize=32), dtype=bool, count=len(paths))


def _row_to_dict(row) -> Dict[str, Any]:
    d = {}
    for c in row.colnames:
        v = row[c]
        d[c] = v.item() if hasattr(v, "item") else v
    return d


# ----------------------------------------------------------------------------- per exposure
_WORKER_METRICS: Optional[List[Metric]] = None


def _init_worker(metric_names, metric_params, extra_modules) -> None:
    global _WORKER_METRICS
    for spec in extra_modules or ():
        load_metric_module(spec)
    ms = build_metrics(metric_names)
    for m in ms:  # carry over parameters (incl. those set in prepare())
        m.params = dict(metric_params[m.name])
    _WORKER_METRICS = ms


def process_exposure(cfg: PipelineConfig, row: Dict[str, Any],
                     metrics: Optional[List[Metric]] = None) -> Dict[str, Any]:
    metrics = metrics if metrics is not None else _WORKER_METRICS
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"flags": {}, "values": {}, "notes": {}, "timing": {}}
    results = {}
    with ExposureContext(cfg, row) as ctx:
        for m in metrics:
            if "dap_output" in m.requires:
                ctx.request_output(m.output_windows())
        rec["payloads"] = {}
        for m in [m for m in metrics if not m.summary]:
            tm, io0 = time.perf_counter(), ctx.io_time
            if m.multi:
                # multi-exposure metric: only collect the per-exposure payload here
                try:
                    rec["payloads"][m.name] = m.collect(ctx)
                except Exception as exc:  # noqa: BLE001
                    log.error("%s collect failed for expnum %s: %s", m.name, ctx.expnum, exc)
                    rec["payloads"][m.name] = None
                rec["timing"][m.name] = (time.perf_counter() - tm) - (ctx.io_time - io0)
                rec["flags"][m.name] = int(QAFlag.UNDEFINED)
                rec["values"][m.name] = m.empty_values()
                continue
            res = m.safe_compute(ctx)
            # compute time of the metric, excluding the (shared, lazily loaded) input I/O
            rec["timing"][m.name] = (time.perf_counter() - tm) - (ctx.io_time - io0)
            rec["flags"][m.name] = int(res.flag)
            rec["values"][m.name] = res.values
            if res.note:
                rec["notes"][m.name] = res.note
        for k in set(IO_KEYS) | set(ctx.timings):
            rec["timing"][k] = float(ctx.timings.get(k, 0.0))
    rec["expnum"] = int(row["expnum"])
    rec["runtime"] = time.perf_counter() - t0
    rec["timing"]["total"] = rec["runtime"]
    return rec


def finalize_records(records: List[Dict[str, Any]], metrics: List[Metric], cfg: PipelineConfig,
                     drpall: Table) -> None:
    """Main process: multi-exposure metrics (from the collected payloads), then summaries."""
    import inspect
    by_exp = {r["expnum"]: r for r in records}
    multi = [m for m in metrics if m.multi]
    # metrics that use results of other multi-exposure metrics (``multi_after``) run after them;
    # results are exchanged through ``shared`` (e.g. qa_fluxmap -> qa_gradient: IFU dipole)
    multi.sort(key=lambda m: len(tuple(getattr(m, "multi_after", ()))))
    shared: Dict[str, Any] = {}
    for m in multi:
        t0 = time.perf_counter()
        payloads = {e: r.get("payloads", {}).get(m.name) for e, r in by_exp.items()}
        try:
            if "shared" in inspect.signature(m.compute_multi).parameters:
                res = m.compute_multi(payloads, cfg, drpall, shared=shared)
            else:
                res = m.compute_multi(payloads, cfg, drpall)
        except Exception as exc:  # noqa: BLE001
            import traceback
            log.error("%s failed: %s\n%s", m.name, exc, traceback.format_exc())
            res = {e: m.result(QAFlag.UNDEFINED, note=f"error: {type(exc).__name__}: {exc}") for e in by_exp}
        dt = (time.perf_counter() - t0) / max(len(by_exp), 1)
        for e, r in by_exp.items():
            rr = res.get(e) or m.result(QAFlag.UNDEFINED, note="no result")
            vals = m.empty_values()
            vals.update(rr.values)
            r["flags"][m.name] = int(rr.flag)
            r["values"][m.name] = vals
            r["timing"][m.name] = r["timing"].get(m.name, 0.0) + dt
            if rr.note:
                r["notes"][m.name] = rr.note
    summarize_records(records, metrics)
    for r in records:
        r.pop("payloads", None)


def summarize_records(records: List[Dict[str, Any]], metrics: List[Metric]) -> None:
    """(Re)compute all summary metrics of every record from the component flags / values / notes."""
    summaries = [m for m in metrics if m.summary]
    for r in records:
        results = {name: MetricResult(QAFlag(f), r["values"].get(name, {}), r["notes"].get(name, ""))
                   for name, f in r["flags"].items() if name not in {m.name for m in summaries}}
        for m in summaries:
            tm = time.perf_counter()
            res = m.safe_summarize(results, metrics, r["expnum"])
            results[m.name] = res
            r["timing"][m.name] = time.perf_counter() - tm
            r["flags"][m.name] = int(res.flag)
            r["values"][m.name] = res.values
            r["notes"].pop(m.name, None)
            if res.note:
                r["notes"][m.name] = res.note


def _process_in_worker(args):
    cfg, row = args
    return process_exposure(cfg, row)


# ----------------------------------------------------------------------------- table
def build_table(rows: Table, records: List[Dict[str, Any]], metrics: List[Metric],
                cfg: PipelineConfig) -> Table:
    by_exp = {r["expnum"]: r for r in records}
    out = Table()
    for c in ID_COLUMNS:
        if c in rows.colnames:
            out[c] = rows[c]

    flags = {m.name: [] for m in metrics}
    for e in rows["expnum"]:
        rec = by_exp[int(e)]
        for m in metrics:
            flags[m.name].append(rec["flags"][m.name])

    n = len(rows)
    # registered metrics that were not run are encoded as UNDEFINED (11) in their slots
    from .registry import available_metrics
    not_run = {cls.slot: int(QAFlag.UNDEFINED) for name, cls in available_metrics().items()
               if name not in flags and cls.slot is not None}
    bm = [m for m in metrics if m.in_bitmask]
    bitmask = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bitmask[i] = pack({**not_run, **{m.slot: flags[m.name][i] for m in bm}})
    out["qa_bitmask"] = bitmask

    # overall (bitmask metrics only); the science summaries replace the former per-group flags
    out["qa_overall"] = [FLAG_LABELS[combine(flags[m.name][i] for m in bm)]
                         for i in range(n)]

    # per-metric flags (human readable, DR20 style); status metrics have no flag
    for m in metrics:
        if not m.status:
            out[m.name] = [FLAG_LABELS[QAFlag(f)] for f in flags[m.name]]

    # per-metric measured values
    for m in metrics:
        for v in m.values:
            col = [by_exp[int(e)]["values"][m.name].get(v.name, v.fill_value())
                   for e in rows["expnum"]]
            if v.dtype.startswith("U"):
                out[m.column(v.name)] = np.asarray([str(x) for x in col])
            else:
                arr = np.asarray(col)
                try:
                    arr = arr.astype(v.dtype)
                except (TypeError, ValueError):
                    arr = np.asarray([v.fill_value() if x is None else x for x in col]).astype(v.dtype)
                out[m.column(v.name)] = arr
            if v.unit:
                out[m.column(v.name)].unit = v.unit
            out[m.column(v.name)].description = v.description

    # timing [s]
    tcols = [("total", "time_total", "Wall time for the exposure (inputs + all metrics)"),
             ("io_sframe", "time_io_sframe", "Time reading SFrame header / tables / data"),
             ("io_dap", "time_io_dap", "Time reading DAP tables"),
             ("io_dap_output", "time_io_dap_output", "Time reading the DAP output (model RSS) file")]
    present = {k for e in rows["expnum"] for k in by_exp[int(e)]["timing"]}
    tcols += [(k, f"time_{k}", d) for k, d in SHARED_KEYS.items() if k in present]
    tcols += [(m.name, f"time_{m.short}", f"Compute time of {m.name} (excluding input I/O)")
              for m in metrics]
    for key, col, desc in tcols:
        out[col] = np.asarray([by_exp[int(e)]["timing"].get(key, np.nan) for e in rows["expnum"]],
                              dtype=float)
        out[col].unit = "s"
        out[col].description = desc

    out["qa_notes"] = ["; ".join(f"{k}: {t}" for k, t in by_exp[int(e)]["notes"].items())
                       for e in rows["expnum"]]
    out["qa_notes"].description = "Short notes from metrics (reasons for flags / UNDEFINED)"
    return out


# ----------------------------------------------------------------------------- driver
def run_qa(drp_version: str, dap_version: Optional[str] = None,
           expnums: Optional[Sequence[int]] = None,
           mjds: Optional[Sequence[int]] = None,
           tileids: Optional[Sequence[int]] = None,
           *,
           metrics: Optional[Sequence[str]] = None,
           metric_options: Optional[Dict[str, Dict[str, Any]]] = None,
           metric_modules: Optional[Sequence[str]] = None,
           redux_dir=None, analysis_dir=None, output_dir=None, drpall_file=None,
           cache_dir=None,
           dap_n_models: int = 108, dap_target_snr: int = 20,
           existing_only: bool = False,
           nproc: int = 1,
           write: bool = True, html: bool = True, tag: Optional[str] = None,
           pdf: bool = True, pdf_n: int = 5, pdf_seed: int = 1, xlsx: bool = False,
           html_detail: Optional[bool] = None,
           sframe_pattern: Optional[str] = None, no_dap: bool = False, no_drpall: bool = False,
           exposure_file=None, max_header_files: Optional[int] = MAX_HEADER_FILES,
           ) -> QAResult:
    """
    Run the QA pipeline.

    Parameters
    ----------
    drp_version, dap_version : versions of the DRP and DAP products to analyse. dap_version
                               None / '' / 'none': legacy layout analysis/{drpver}/{tilegrp}/...
    expnums, mjds, tileids   : optional subsets (combined with logical AND).
    metrics                  : names of metrics to run (default: all registered).
    metric_options           : per-metric parameter overrides.
    metric_modules           : extra modules (dotted name or .py path) defining metrics.
    existing_only            : only keep exposures whose SFrame exists on disk.
    sframe_pattern           : SFrame files of a non-standard layout, pattern with an {expnum} placeholder
                               (glob wildcards allowed; zero-padding irrelevant), e.g.
                               "/data/test/SFrame-*-{expnum}.fits". With a drpall: only matching exposures.
    no_dap                   : no DAP products: the DAP-based metrics are UNDEFINED.
    no_drpall                : exposure table from the SFrame primary headers (also automatic when an
                               sframe_pattern is given and the drpall file does not exist).
    exposure_file            : TOML file MJD -> group (optional) -> expnums (see lvmqa.expfile); selects
                               these exposures (AND with the other filters) and fills the column
                               ``group`` (default "all").
    max_header_files         : without a drpall, refuse to read more SFrame headers than this (safety
                               against listing a whole production tree by mistake; None / 0 = no limit).
    nproc                    : number of worker processes.
    write, html              : write FITS+XLSX (and HTML report) to ``output_dir``.
    tag                      : optional suffix for output file names.
    xlsx                     : also write the XLSX copies of the FITS tables (slow for large runs)
    pdf, pdf_n, pdf_seed     : write the diagnostics PDF (<base>_diagnostics.pdf; up to pdf_n random
                               exposures per status and metric) after the tables (needs write).
    """
    t_start = time.time()
    for spec in metric_modules or ():
        load_metric_module(spec)
    cfg = PipelineConfig(drp_version=drp_version, dap_version=dap_version,
                         redux_dir=redux_dir, analysis_dir=analysis_dir,
                         output_dir=output_dir, drpall_file=drpall_file,
                         cache_dir=cache_dir, dap_n_models=dap_n_models,
                         dap_target_snr=dap_target_snr,
                         metric_options=dict(metric_options or {}),
                         sframe_pattern=sframe_pattern, no_dap=bool(no_dap), no_drpall=bool(no_drpall))
    mlist = build_metrics(metrics, cfg.metric_options)
    log.info("lvmqa %s | DRP %s | DAP %s | metrics: %s", __version__, drp_version,
             cfg.dap_label, ", ".join(m.name for m in mlist))

    groups_spec = None
    if exposure_file:
        from .expfile import read_exposure_file
        groups_spec = read_exposure_file(exposure_file)
        fexp = sorted(groups_spec)
        expnums = [e for e in fexp if e in set(int(x) for x in expnums)] if expnums else fexp
        log.info("exposure file %s: %d exposure(s) in %d group(s)", exposure_file, len(fexp),
                 len({g for _, g in groups_spec.values()}))
    drpall = exposure_table(cfg, expnums, nthreads=max(16, 2 * nproc), max_files=max_header_files)
    rows = select_exposures(drpall, expnums, mjds, tileids)
    assign_groups(rows, groups_spec)
    if existing_only and len(rows):
        t_ex = time.time()
        n0 = len(rows)
        rows = rows[sframes_exist(cfg, rows, nthreads=max(16, 2 * nproc))]
        log.info("--existing-only: %d of %d SFrames on disk (checked in %.1f s)", len(rows), n0,
                 time.time() - t_ex)
    from_headers = bool(cfg.no_drpall or (cfg.sframe_pattern and not Path(cfg.drpall_file).exists()))
    log.info("%d exposure(s) selected (%s: %d rows)", len(rows), "SFrame headers" if from_headers else "drpall",
             len(drpall))

    for m in mlist:
        m.prepare(cfg)

    row_dicts = [_row_to_dict(r) for r in rows]
    records: List[Dict[str, Any]] = []
    if nproc > 1 and len(row_dicts) > 1:
        params = {m.name: m.params for m in mlist}
        with ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker,
                                 initargs=([m.name for m in mlist], params,
                                           list(metric_modules or ()))) as ex:
            for i, rec in enumerate(ex.map(_process_in_worker,
                                           [(cfg, r) for r in row_dicts], chunksize=1)):
                records.append(rec)
                _log_progress(i, len(row_dicts), rec)
    else:
        for i, r in enumerate(row_dicts):
            rec = process_exposure(cfg, r, mlist)
            records.append(rec)
            _log_progress(i, len(row_dicts), rec)

    finalize_records(records, mlist, cfg, drpall)
    table = build_table(rows, records, mlist, cfg)
    selection = {**({"exposure_file": str(exposure_file), "groups": group_counts_text(rows)}
                    if exposure_file else {}),
                 "expnums": list(expnums or []), "mjds": list(mjds or []),
                 "tileids": list(tileids or []), "existing_only": existing_only,   # report: shown as ranges
                 "sframe_pattern": cfg.sframe_pattern or "", "no_dap": cfg.no_dap,
                 "no_drpall": from_headers}
    result = QAResult(table=table, metrics=mlist, config=cfg, selection=selection,
                      nproc=nproc)
    table.meta["RUNTIME"] = round(time.time() - t_start, 1)

    if write:
        from .output import write_outputs
        t_w = time.time()
        result.files.update(write_outputs(result, tag, xlsx=xlsx))
        log.info("tables written in %.1f s%s", time.time() - t_w, "" if xlsx else " (no XLSX; --xlsx to add)")
        if html:
            from .output import output_basename
            base = Path(cfg.output_dir) / output_basename(cfg, tag)
            from .report import write_html
            t_w = time.time()
            result.files["html"] = write_html(result, Path(f"{base}.html"), detail=html_detail)
            log.info("HTML report written in %.1f s", time.time() - t_w)
        if pdf and len(table):
            from .diagnostics import make_pdf
            t_pdf = time.time()
            try:
                result.files["pdf"] = make_pdf(
                    drp_version, dap_version, n=pdf_n, seed=pdf_seed, nproc=max(1, nproc), tag=tag,
                    metric_options=cfg.metric_options, redux_dir=cfg.redux_dir,
                    analysis_dir=cfg.analysis_dir, output_dir=cfg.output_dir,
                    drpall_file=cfg.drpall_file, cache_dir=cfg.cache_dir,
                    dap_n_models=cfg.dap_n_models, dap_target_snr=cfg.dap_target_snr,
                    sframe_pattern=cfg.sframe_pattern, no_dap=cfg.no_dap, no_drpall=selection["no_drpall"])
                log.info("diagnostics PDF in %.1f s", time.time() - t_pdf)
            except Exception as exc:  # noqa: BLE001 - the tables are already written
                log.error("diagnostics PDF failed: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
        for k, p in result.files.items():
            log.info("wrote %s: %s", k, p)
    log.info("done in %.1f s", time.time() - t_start)
    return result


def _log_progress(i: int, n: int, rec: Dict[str, Any]) -> None:
    flags = " ".join(f"{k[3:]}={QAFlag(v).name[0]}" for k, v in rec["flags"].items())
    log.info("[%d/%d] expnum %08d  %s  (%.1f s)", i + 1, n, rec["expnum"], flags, rec["runtime"])
