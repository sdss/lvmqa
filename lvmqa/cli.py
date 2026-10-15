"""Command-line interface:  python -m lvmqa DRP_VERSION DAP_VERSION [options]"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from typing import Dict, List, Optional, Sequence

from . import __version__


def parse_int_list(tokens: Optional[Sequence[str]]) -> Optional[List[int]]:
    """'1,2 5-7 9' -> [1, 2, 5, 6, 7, 9]"""
    if not tokens:
        return None
    out: List[int] = []
    for tok in tokens:
        for part in str(tok).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part[1:]:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                out.extend(range(min(a, b), max(a, b) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def parse_metric_options(items: Optional[Sequence[str]]) -> Dict[str, Dict]:
    """['qa_fit_consistency.min_snr=3', ...] -> {'qa_fit_consistency': {'min_snr': 3}}"""
    opts: Dict[str, Dict] = {}
    for it in items or ():
        key, _, val = it.partition("=")
        metric, _, param = key.partition(".")
        if not param or not _:
            raise SystemExit(f"bad --set value {it!r}; expected METRIC.PARAM=VALUE")
        try:
            v = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            v = val
        opts.setdefault(metric.strip(), {})[param.strip()] = v
    return opts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lvmqa",
        description="LVM QA pipeline: per-exposure QA metrics and bitmasks from SFrames, "
                    "DAP results and the drpall file.")
    ap.add_argument("drp_version", nargs="?", help="DRP version, e.g. 1.3.2")
    ap.add_argument("dap_version", nargs="?",
                    help="DAP version, e.g. 1.2.0.251218. Omit, or give '' / none, for the legacy "
                         "(DR20 and earlier) layout analysis/<drpver>/<tilegrp>/...")
    sel = ap.add_argument_group("selection (combined with AND; default: all drpall exposures)")
    sel.add_argument("--expnums", nargs="+", metavar="N", help="exposure numbers (lists/ranges: 1,2 5-9)")
    sel.add_argument("--mjds", nargs="+", metavar="N", help="MJDs (lists/ranges allowed)")
    sel.add_argument("--tileids", nargs="+", metavar="N", help="tile ids (lists/ranges allowed)")
    sel.add_argument("--existing-only", action="store_true",
                     help="skip exposures whose SFrame is not on disk")
    mg = ap.add_argument_group("metrics")
    mg.add_argument("--metrics", nargs="+", metavar="NAME", help="run only these metrics")
    mg.add_argument("--metric-module", action="append", default=[], metavar="MOD",
                    help="extra module (dotted name or .py file) defining metrics; repeatable")
    mg.add_argument("--set", action="append", default=[], metavar="METRIC.PARAM=VALUE",
                    help="override a metric parameter, e.g. qa_fit_consistency.min_snr=3; repeatable")
    mg.add_argument("--list-metrics", action="store_true", help="list available metrics and exit")
    pg = ap.add_argument_group("paths")
    pg.add_argument("--redux-dir", help="root of DRP products (default: $LVM_SPECTRO_REDUX or dev path)")
    pg.add_argument("--analysis-dir", help="root of DAP products (default: $LVM_SPECTRO_ANALYSIS or dev path)")
    pg.add_argument("--drpall", help="drpall file (default: <redux>/<drpver>/drpall-<drpver>.fits)")
    pg.add_argument("--output-dir", help="output directory (default: $LVM_QA_OUTPUT_DIR or dev path)")
    pg.add_argument("--cache-dir", help="cache directory (default: <output-dir>/cache)")
    pg.add_argument("--dap-n-models", type=int, default=108)
    pg.add_argument("--dap-target-snr", type=int, default=20)
    og = ap.add_argument_group("output / execution")
    og.add_argument("--tag", help="suffix for output file names")
    og.add_argument("--no-html", action="store_true", help="do not write the HTML report")
    og.add_argument("--html-detail", action="store_true",
                    help="detailed per-exposure HTML plots also for >= 1000 exposures (default: compact layout there)")
    og.add_argument("--xlsx", action="store_true",
                    help="also write XLSX copies of the QA and SUPL tables (slow for large runs: ~800 SUPL columns)")
    og.add_argument("--no-pdf", action="store_true", help="do not write the diagnostics PDF")
    og.add_argument("--pdf-n", type=int, default=5, metavar="N",
                    help="diagnostics PDF: random exposures per status and metric (default 5)")
    og.add_argument("--pdf-seed", type=int, default=1, help="diagnostics PDF: random seed (default 1)")
    og.add_argument("--no-write", action="store_true", help="do not write any output")
    og.add_argument("-j", "--nproc", type=int, default=1, help="worker processes (default 1)")
    og.add_argument("-v", "--verbose", action="count", default=0)
    og.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=f"lvmqa {__version__}")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_metrics and not args.drp_version:
        parser.error("drp_version is required")
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose > 1 else logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    from .registry import build_metrics, load_metric_module
    for spec in args.metric_module:
        load_metric_module(spec)
    if args.list_metrics:
        for m in build_metrics():
            slot = f"{m.slot:2d}" if m.slot is not None else " -"
            print(f"slot {slot}  {m.name:18s} [{m.group}] ({m.role})  {m.title}")
            for k, v in m.params.items():
                print(f"{'':10s}{k} = {v!r}")
        return 0

    from .pipeline import run_qa
    res = run_qa(
        args.drp_version, args.dap_version,
        expnums=parse_int_list(args.expnums), mjds=parse_int_list(args.mjds),
        tileids=parse_int_list(args.tileids),
        metrics=args.metrics, metric_options=parse_metric_options(args.set),
        metric_modules=args.metric_module,
        redux_dir=args.redux_dir, analysis_dir=args.analysis_dir, drpall_file=args.drpall,
        output_dir=args.output_dir, cache_dir=args.cache_dir,
        dap_n_models=args.dap_n_models, dap_target_snr=args.dap_target_snr,
        existing_only=args.existing_only, nproc=args.nproc,
        write=not args.no_write, html=not args.no_html, tag=args.tag,
        pdf=not args.no_pdf, pdf_n=args.pdf_n, pdf_seed=args.pdf_seed, xlsx=args.xlsx,
        html_detail=True if args.html_detail else None,
    )
    t = res.table
    if len(t):
        from collections import Counter
        print(f"\n{len(t)} exposures; overall: {dict(Counter(str(x) for x in t['qa_overall']))}")
        for m in res.metrics:
            if getattr(m, "status", False):
                for c in [m.column(v.name) for v in m.values if v.dtype in ("?", "bool")]:
                    print(f"  {c:18s} {dict(Counter(str(x) for x in t[c]))}")
            elif m.name in t.colnames:
                print(f"  {m.name:18s} {dict(Counter(str(x) for x in t[m.name]))}")
    for k, p in res.files.items():
        print(f"  {k:12s} {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
