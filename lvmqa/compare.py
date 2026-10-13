"""
Compare two lvmqa runs (e.g. DRP 1.3.2 vs 1.2.0, or before / after changing thresholds) on the
exposures they have in common, and write an HTML report.

    python -m lvmqa.compare RUN_A.fits RUN_B.fits [--labels A B] [--out compare.html]
    from lvmqa.compare import compare_runs

RUN_*.fits are the main QA files (lvmqa-*.fits); the measured values are read from the
matching *_supl.fits. "B" is the new run: "better" means B has a less severe flag than A.

Per flag (overall = worst bitmask flag, science summaries, qa_obs, every metric):
  * status counts in A and B, 4x4 transition matrix A -> B
  * n_better / n_worse / n_same among exposures defined (not UNDEFINED) in both runs,
    newly defined / newly undefined exposures
  * mean severity (GOOD 0, WARNING 1, MAJOR 2) in A and B, its change, and the two-sided sign
    test p-value of better vs worse (binomial, H0: equal probability)
  * verdict: improved / worse / unchanged (p < 0.05 and |n_better - n_worse| >= 2, else
    'mixed' if both occur, 'unchanged' if no changes)
Per main measured value of each metric: paired median of A and B, median of B - A, the median
of |value| in A and B (for deviation-type values smaller |value| is better), the fraction of
exposures where |value| decreased, and overlaid histograms.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.table import Table

from .report import COLORS, CSS, GRID, ICON, INK2, STATUS, SURFACE, _layout, _table_html

SEV = {"GOOD": 0, "WARNING": 1, "MAJOR ISSUE": 2}
VERDICT_COL = {"improved": "#0ca30c", "worse": "#d03b3b", "mixed": "#fab219", "unchanged": "#898781",
               "n/a": "#c3c2b7"}


def _supl_path(p: Path) -> Path:
    s = str(p)
    return Path(s[:-5] + "_supl.fits") if s.endswith(".fits") else Path(s + "_supl.fits")


def _sign_test(nb: int, nw: int) -> float:
    """Two-sided exact binomial p-value for nb successes out of nb + nw (p = 0.5)."""
    n = nb + nw
    if n == 0:
        return 1.0
    k = min(nb, nw)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _bitmask_flags(desc: Table) -> List[str]:
    out = []
    for r in desc:
        bits = str(r["Bits"]) if "Bits" in desc.colnames else ""
        if bits and not bits.startswith("-"):
            out.append(str(r["Metric"]))
    return out


def _overall(t: Table, cols: List[str]) -> np.ndarray:
    order = {"GOOD": 0, "UNDEFINED": 1, "WARNING": 2, "MAJOR ISSUE": 3}
    inv = {v: k for k, v in order.items()}
    cols = [c for c in cols if c in t.colnames]
    if not cols:
        return np.array(["UNDEFINED"] * len(t))
    a = np.array([[order.get(str(x), 1) for x in t[c]] for c in cols])
    return np.array([inv[v] for v in a.max(axis=0)])


def compare_flags(fa: np.ndarray, fb: np.ndarray) -> Dict:
    fa, fb = np.asarray(fa).astype(str), np.asarray(fb).astype(str)
    trans = np.array([[int(np.sum((fa == s) & (fb == t))) for t in STATUS] for s in STATUS])
    da = np.array([SEV.get(x, -1) for x in fa])
    db = np.array([SEV.get(x, -1) for x in fb])
    both = (da >= 0) & (db >= 0)
    nb, nw = int(np.sum(both & (db < da))), int(np.sum(both & (db > da)))
    res = dict(trans=trans, n_both=int(both.sum()), n_better=nb, n_worse=nw,
               n_same=int(np.sum(both & (db == da))),
               newly_defined=int(np.sum((da < 0) & (db >= 0))), newly_undefined=int(np.sum((da >= 0) & (db < 0))),
               counts_a={s: int(np.sum(fa == s)) for s in STATUS}, counts_b={s: int(np.sum(fb == s)) for s in STATUS},
               sev_a=float(da[da >= 0].mean()) if (da >= 0).any() else np.nan,
               sev_b=float(db[db >= 0].mean()) if (db >= 0).any() else np.nan,
               sev_a_both=float(da[both].mean()) if both.any() else np.nan,
               sev_b_both=float(db[both].mean()) if both.any() else np.nan)
    res["p_sign"] = _sign_test(nb, nw)
    if res["n_both"] == 0:
        res["verdict"] = "n/a"
    elif nb == nw == 0:
        res["verdict"] = "unchanged"
    elif res["p_sign"] < 0.05 and abs(nb - nw) >= 2:
        res["verdict"] = "improved" if nb > nw else "worse"
    elif nb and nw:
        res["verdict"] = "mixed"
    else:
        res["verdict"] = "improved" if nb > nw else "worse"
    return res


def compare_runs(file_a, file_b, labels=("A", "B"), out=None) -> Tuple[Path, Dict]:
    file_a, file_b = Path(file_a), Path(file_b)
    qa, qb = Table.read(file_a, hdu="QA"), Table.read(file_b, hdu="QA")
    da_, db_ = Table.read(file_a, hdu="DESC"), Table.read(file_b, hdu="DESC")
    sa, sb = Table.read(_supl_path(file_a), hdu="SUPL"), Table.read(_supl_path(file_b), hdu="SUPL")
    ea, eb = set(int(x) for x in qa["expnum"]), set(int(x) for x in qb["expnum"])
    common = sorted(ea & eb)
    ia = {int(e): i for i, e in enumerate(qa["expnum"])}
    ib = {int(e): i for i, e in enumerate(qb["expnum"])}
    qa, qb = qa[[ia[e] for e in common]], qb[[ib[e] for e in common]]
    ja = {int(e): i for i, e in enumerate(sa["expnum"])}
    jb = {int(e): i for i, e in enumerate(sb["expnum"])}
    sa, sb = sa[[ja[e] for e in common]], sb[[jb[e] for e in common]]

    roles = {str(r["Metric"]): str(r["Role"]) if "Role" in da_.colnames else "" for r in da_}
    roles.update({str(r["Metric"]): str(r["Role"]) if "Role" in db_.colnames else "" for r in db_})
    flags = [c for c in qa.colnames if c.startswith("qa_") and c != "qa_bitmask" and c in qb.colnames]
    summ = [c for c in flags if "summary" in roles.get(c, "")]
    indiv = [c for c in flags if c not in summ]
    bm = sorted(set(_bitmask_flags(da_)) | set(_bitmask_flags(db_)))
    rows_cmp = {"Overall (bitmask flags)": compare_flags(_overall(qa, bm), _overall(qb, bm))}
    for c in summ + indiv:
        rows_cmp[c] = compare_flags(qa[c], qb[c])

    # ---- main values
    from .registry import available_metrics
    reg = available_metrics()
    vals = []
    for c in indiv + summ:
        cls = reg.get(c)
        mv = getattr(cls, "main_value", None) if cls else None
        if not mv:
            continue
        col = f"{c[3:]}_{mv}"
        if col not in sa.colnames or col not in sb.colnames:
            continue
        try:
            xa = np.asarray(sa[col], dtype=float)
            xb = np.asarray(sb[col], dtype=float)
        except (TypeError, ValueError):
            continue
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 2:
            continue
        d = xb[ok] - xa[ok]
        vals.append(dict(metric=c, column=col, n=int(ok.sum()), med_a=float(np.median(xa[ok])),
                         med_b=float(np.median(xb[ok])), med_diff=float(np.median(d)),
                         medabs_a=float(np.median(np.abs(xa[ok]))), medabs_b=float(np.median(np.abs(xb[ok]))),
                         frac_abs_decreased=float(np.mean(np.abs(xb[ok]) < np.abs(xa[ok]))),
                         xa=xa[ok], xb=xb[ok]))

    # ---- figures
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    la, lb = labels
    figs = []
    keys = list(rows_cmp)
    # (1) severity shift per flag
    fig = go.Figure()
    for lab, k, col in ((la, "sev_a_both", "#898781"), (lb, "sev_b_both", "#2a78d6")):
        fig.add_bar(y=keys[::-1], x=[rows_cmp[x][k] for x in keys[::-1]], name=lab, orientation="h",
                    marker=dict(color=col))
    fig.update_layout(barmode="group")
    figs.append(("Mean severity per flag (GOOD 0, WARNING 1, MAJOR 2; exposures defined in both runs)",
                 _layout(fig, "", height=140 + 30 * len(keys), xaxis_title="mean severity (lower is better)",
                         margin=dict(l=60, r=20, t=30, b=50))))
    # (2) better / worse counts
    fig = go.Figure()
    fig.add_bar(y=keys[::-1], x=[rows_cmp[x]["n_better"] for x in keys[::-1]], name=f"better in {lb}",
                orientation="h", marker=dict(color=COLORS["GOOD"]))
    fig.add_bar(y=keys[::-1], x=[-rows_cmp[x]["n_worse"] for x in keys[::-1]], name=f"worse in {lb}",
                orientation="h", marker=dict(color=COLORS["MAJOR ISSUE"]))
    fig.update_layout(barmode="relative")
    figs.append((f"Exposures with a better / worse flag in {lb} than in {la}",
                 _layout(fig, "", height=140 + 30 * len(keys), xaxis_title="number of exposures (worse < 0 < better)",
                         margin=dict(l=60, r=20, t=30, b=50))))
    # (3) status counts A vs B for overall + summaries
    sk = ["Overall (bitmask flags)"] + summ
    fig = make_subplots(rows=1, cols=len(sk), subplot_titles=[s.replace(" (bitmask flags)", "") for s in sk],
                        horizontal_spacing=0.03)
    for i, k in enumerate(sk, start=1):
        for s in STATUS:
            fig.add_bar(x=[la, lb], y=[rows_cmp[k]["counts_a"][s], rows_cmp[k]["counts_b"][s]], name=f"{ICON[s]} {s}",
                        marker=dict(color=COLORS[s], line=dict(color=SURFACE, width=1)), legendgroup=s,
                        showlegend=(i == 1), row=1, col=i)
    fig.update_layout(barmode="stack")
    fig.update_annotations(font=dict(size=11, color=INK2))
    _layout(fig, "", height=410, margin=dict(l=40, r=20, t=90, b=40))
    fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1.18, yanchor="bottom"))
    figs.append(("Status counts: overall and summaries", fig))
    # (4) transition matrices for overall + summaries
    fig = make_subplots(rows=1, cols=len(sk), subplot_titles=[s.replace(" (bitmask flags)", "") for s in sk],
                        horizontal_spacing=0.04)
    short = ["G", "W", "M", "U"]
    for i, k in enumerate(sk, start=1):
        T = rows_cmp[k]["trans"]
        fig.add_trace(go.Heatmap(z=T, x=short, y=short, text=T, texttemplate="%{text}", colorscale="Blues",
                                 showscale=False, hovertemplate=f"{la} %{{y}} -> {lb} %{{x}}: %{{z}}<extra></extra>"),
                      row=1, col=i)
        fig.update_yaxes(autorange="reversed", title=la if i == 1 else None, row=1, col=i)
        fig.update_xaxes(title=lb, row=1, col=i)
    fig.update_annotations(font=dict(size=11, color=INK2))
    figs.append((f"Transition matrices {la} (rows) -> {lb} (columns)", _layout(fig, "", height=330,
                                                                               margin=dict(l=40, r=20, t=60, b=50))))
    # (5) value histograms
    if vals:
        ncol = 3
        nrow = int(np.ceil(len(vals) / ncol))
        from .report import _grid
        H, vs = _grid(nrow, panel_px=180, gap_px=80)
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[v["column"] for v in vals],
                            horizontal_spacing=0.07, vertical_spacing=vs)
        for i, v in enumerate(vals):
            r, c = i // ncol + 1, i % ncol + 1
            lo, hi = np.nanpercentile(np.r_[v["xa"], v["xb"]], [1, 99])
            if not hi > lo:
                lo, hi = lo - 0.5, hi + 0.5
            bins = dict(start=lo, end=hi, size=(hi - lo) / 30)
            for lab, x, col in ((la, v["xa"], "#898781"), (lb, v["xb"], "#2a78d6")):
                fig.add_trace(go.Histogram(x=x, xbins=bins, name=lab, legendgroup=lab, showlegend=(i == 0),
                                           marker=dict(color=col), opacity=0.6), row=r, col=c)
        fig.update_layout(barmode="overlay")
        fig.update_annotations(font=dict(size=10, color=INK2))
        _layout(fig, "", height=H, margin=dict(l=50, r=20, t=90, b=50))
        fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=1.0 + 50 / H, yanchor="bottom"))
        figs.append(("Main measured values (paired exposures)", fig))

    # ---- tables
    trows = []
    for k, r in rows_cmp.items():
        trows.append({"flag": k, "verdict": r["verdict"], f"better in {lb}": r["n_better"],
                      f"worse in {lb}": r["n_worse"], "same": r["n_same"],
                      "newly defined": r["newly_defined"], "newly undefined": r["newly_undefined"],
                      f"severity {la}": f"{r['sev_a_both']:.2f}", f"severity {lb}": f"{r['sev_b_both']:.2f}",
                      "sign-test p": f"{r['p_sign']:.3f}",
                      f"{la} G/W/M/U": "/".join(str(r["counts_a"][s]) for s in STATUS),
                      f"{lb} G/W/M/U": "/".join(str(r["counts_b"][s]) for s in STATUS)})
    tcols = list(trows[0].keys())
    body = []
    for r in trows:
        tds = []
        for c in tcols:
            v = str(r[c])
            if c == "verdict":
                tds.append(f'<td><span class="pill" style="--c:{VERDICT_COL.get(v, "#898781")}">{v}</span></td>')
            else:
                tds.append(f"<td>{_html.escape(v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    flag_table = (f'<div class="tbl"><table><thead><tr>{"".join(f"<th>{_html.escape(c)}</th>" for c in tcols)}'
                  f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    vrows = [{"metric": v["metric"], "value": v["column"], "n": v["n"], f"median {la}": f"{v['med_a']:.4g}",
              f"median {lb}": f"{v['med_b']:.4g}", f"median ({lb}-{la})": f"{v['med_diff']:+.4g}",
              f"median |x| {la}": f"{v['medabs_a']:.4g}", f"median |x| {lb}": f"{v['medabs_b']:.4g}",
              f"frac |x| smaller in {lb}": f"{v['frac_abs_decreased']:.2f}"} for v in vals]
    val_table = _table_html(vrows, list(vrows[0].keys())) if vrows else "<p class='meta'>no common values</p>"
    ov_a, ov_b = _overall(qa, bm), _overall(qb, bm)
    ch = [{"expnum": str(e), "tileid": str(qa["tileid"][i]), f"overall {la}": ov_a[i], f"overall {lb}": ov_b[i],
           "changed flags": ", ".join(f"{c}: {qa[c][i]}->{qb[c][i]}" for c in flags if str(qa[c][i]) != str(qb[c][i]))}
          for i, e in enumerate(common) if any(str(qa[c][i]) != str(qb[c][i]) for c in flags)]
    ch_table = _table_html(ch, list(ch[0].keys()), status_cols={f"overall {la}", f"overall {lb}"}) if ch else \
        "<p class='meta'>no flag differences</p>"

    n_imp = sum(r["verdict"] == "improved" for r in rows_cmp.values())
    n_wor = sum(r["verdict"] == "worse" for r in rows_cmp.values())
    tiles = [("common exposures", str(len(common))), (f"only in {la}", str(len(ea - eb))),
             (f"only in {lb}", str(len(eb - ea))), ("flags improved", str(n_imp)), ("flags worse", str(n_wor)),
             ("overall verdict", rows_cmp["Overall (bitmask flags)"]["verdict"])]
    warn_html = ("" if common else "<p class='meta' style='color:#d03b3b'><b>No exposures in common</b>: the runs "
                 "cover different expnums, so no paired comparison is possible.</p>")
    parts, first = [], True
    for title, f in figs:
        div = f.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                        config={"displaylogo": False, "responsive": True})
        first = False
        parts.append(f'<h2>{_html.escape(title)}</h2><div class="card">{div}</div>')
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LVM QA comparison</title><style>{CSS}</style></head>
<body><main>
<h1>LVM QA comparison &mdash; {_html.escape(la)} vs {_html.escape(lb)}</h1>
<div class="meta">generated {_dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC<br>
{_html.escape(la)}: <code>{_html.escape(file_a.name)}</code><br>{_html.escape(lb)}: <code>{_html.escape(file_b.name)}</code><br>
"better" = less severe flag in {_html.escape(lb)} (GOOD &lt; WARNING &lt; MAJOR; UNDEFINED counted separately).
Verdict: improved / worse if the sign test of better vs worse gives p &lt; 0.05 and the counts differ by &ge; 2,
otherwise mixed (both directions) or the direction of the few changes.</div>
{warn_html}<div class="tiles">{''.join(f'<div class="tile"><div class="v">{v}</div><div class="l">{k}</div></div>' for k, v in tiles)}</div>
<h2>Flags: overall, summaries and metrics</h2>{flag_table}
{''.join(parts)}
<h2>Main measured values</h2>{val_table}
<h2>Exposures with different flags ({len(ch)})</h2>{ch_table}
</main></body></html>"""
    out = Path(out) if out else file_b.parent / f"compare_{file_a.stem}_vs_{file_b.stem}.html"
    out.write_text(doc, encoding="utf-8")
    return out, rows_cmp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_a", help="main QA FITS of the reference run (lvmqa-*.fits)")
    ap.add_argument("run_b", help="main QA FITS of the new run")
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    labels = a.labels or (Path(a.run_a).stem, Path(a.run_b).stem)
    out, res = compare_runs(a.run_a, a.run_b, labels, a.out)
    if all(r["n_both"] == 0 and sum(r["counts_a"].values()) == 0 for r in res.values()):
        print("WARNING: no exposures in common")
    w = max(len(k) for k in res)
    for k, r in res.items():
        print(f"{k:{w}s} {r['verdict']:9s} better {r['n_better']:3d} worse {r['n_worse']:3d} "
              f"severity {r['sev_a_both']:.2f} -> {r['sev_b_both']:.2f}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
