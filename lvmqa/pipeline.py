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
from .registry import GROUP_KEYS, GROUPS, Metric, build_metrics, load_metric_module

log = logging.getLogger("lvmqa")

# drpall columns copied to the output: identification (main + supl files) and
# observational context (supl file only)
MAIN_ID_COLUMNS = ("expnum", "tileid", "mjd", "object")
ID_COLUMNS = MAIN_ID_COLUMNS + ("exptime", "obstime", "dpos", "sci_ra", "sci_dec", "sci_pa",
                                "sci_amass", "fluxcal")
IO_KEYS = ("io_sframe", "io_dap")


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
        for m in [m for m in metrics if not m.summary] + [m for m in metrics if m.summary]:
            if m.summary:
                tm = time.perf_counter()
                res = m.safe_summarize(results, metrics, ctx.expnum)
                results[m.name] = res
                rec["timing"][m.name] = time.perf_counter() - tm
                rec["flags"][m.name] = int(res.flag)
                rec["values"][m.name] = res.values
                if res.note:
                    rec["notes"][m.name] = res.note
                continue
            tm, io0 = time.perf_counter(), ctx.io_time
            res = m.safe_compute(ctx)
            results[m.name] = res
            # compute time of the metric, excluding the (shared, lazily loaded) input I/O
            rec["timing"][m.name] = (time.perf_counter() - tm) - (ctx.io_time - io0)
            rec["flags"][m.name] = int(res.flag)
            rec["values"][m.name] = res.values
            if res.note:
                rec["notes"][m.name] = res.note
        for k in IO_KEYS:
            rec["timing"][k] = float(ctx.timings.get(k, 0.0))
    rec["expnum"] = int(row["expnum"])
    rec["runtime"] = time.perf_counter() - t0
    rec["timing"]["total"] = rec["runtime"]
    return rec


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

    # overall (bitmask metrics only) + per-group summary flags (HTML only)
    out["qa_overall"] = [FLAG_LABELS[combine(flags[m.name][i] for m in bm)]
                         for i in range(n)]
    for g in GROUPS:
        gm = [m for m in bm if m.group == g]
        if not gm:
            continue
        out[f"qa_grp_{GROUP_KEYS[g]}"] = [
            FLAG_LABELS[combine(flags[m.name][i] for m in gm)] for i in range(n)]

    # per-metric flags (human readable, DR20 style)
    for m in metrics:
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
             ("io_dap", "time_io_dap", "Time reading DAP tables")]
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
    nproc                    : number of worker processes.
    write, html              : write FITS+XLSX (and HTML report) to ``output_dir``.
    tag                      : optional suffix for output file names.
    """
    t_start = time.time()
    for spec in metric_modules or ():
        load_metric_module(spec)
    cfg = PipelineConfig(drp_version=drp_version, dap_version=dap_version,
                         redux_dir=redux_dir, analysis_dir=analysis_dir,
                         output_dir=output_dir, drpall_file=drpall_file,
                         cache_dir=cache_dir, dap_n_models=dap_n_models,
                         dap_target_snr=dap_target_snr,
                         metric_options=dict(metric_options or {}))
    mlist = build_metrics(metrics, cfg.metric_options)
    log.info("lvmqa %s | DRP %s | DAP %s | metrics: %s", __version__, drp_version,
             cfg.dap_label, ", ".join(m.name for m in mlist))

    drpall = load_drpall(cfg)
    rows = select_exposures(drpall, expnums, mjds, tileids)
    if existing_only and len(rows):
        keep = [ExposureContext(cfg, _row_to_dict(r)).has_sframe for r in rows]
        rows = rows[np.asarray(keep, dtype=bool)]
    log.info("%d exposure(s) selected (drpall: %d rows)", len(rows), len(drpall))

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

    table = build_table(rows, records, mlist, cfg)
    selection = {"expnums": list(expnums or []), "mjds": list(mjds or []),
                 "tileids": list(tileids or []), "existing_only": existing_only}
    result = QAResult(table=table, metrics=mlist, config=cfg, selection=selection,
                      nproc=nproc)
    table.meta["RUNTIME"] = round(time.time() - t_start, 1)

    if write:
        from .output import write_outputs
        result.files.update(write_outputs(result, tag))
        if html:
            from .output import output_basename
            base = Path(cfg.output_dir) / output_basename(cfg, tag)
            from .report import write_html
            result.files["html"] = write_html(result, Path(f"{base}.html"))
        for k, p in result.files.items():
            log.info("wrote %s: %s", k, p)
    log.info("done in %.1f s", time.time() - t_start)
    return result


def _log_progress(i: int, n: int, rec: Dict[str, Any]) -> None:
    flags = " ".join(f"{k[3:]}={QAFlag(v).name[0]}" for k, v in rec["flags"].items())
    log.info("[%d/%d] expnum %08d  %s  (%.1f s)", i + 1, n, rec["expnum"], flags, rec["runtime"])
