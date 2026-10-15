"""
HTML report (plotly). v1.15: detailed per-exposure layout (below) for runs with < 1000 exposures
(or --html-detail); larger runs get the compact layout (see write_html / "compact report"):
per-night status maps, one sky marker per tile, nightly medians, precomputed histograms, an exposure
lookup and the worst exposures per metric. Timing: summary only. Detailed layout:

1. run summary + status tiles
2. status counts per metric (stacked bars)
3. exposure x metric status map (heatmap, hover shows values / notes)
4. sky map coloured by status (dropdown: overall / group / metric)
5. main value of every metric vs MJD, coloured by status
6. distributions of the main values
7. processing time per exposure and per metric
8. metric definitions (DESC) and list of flagged exposures
"""

from __future__ import annotations

import datetime as _dt
import html as _html
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import __version__
from .registry import GROUP_KEYS, GROUP_OBS, GROUPS

STATUS = ["GOOD", "WARNING", "MAJOR ISSUE", "UNDEFINED"]
COLORS = {"GOOD": "#0ca30c", "WARNING": "#fab219", "MAJOR ISSUE": "#d03b3b",
          "UNDEFINED": "#c3c2b7"}
ICON = {"GOOD": "✔", "WARNING": "⚠", "MAJOR ISSUE": "✖", "UNDEFINED": "?"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig, title: str, height: int = 420, **kw):
    kw.setdefault("margin", dict(l=60, r=20, t=50, b=50))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK)),
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    return fig


def _grid(nrow: int, panel_px: int = 170, gap_px: int = 70, top: int = 90, bottom: int = 60):
    """Figure height and plotly vertical_spacing for nrow panels of panel_px with gap_px gaps."""
    plot_h = nrow * panel_px + max(nrow - 1, 0) * gap_px
    vs = gap_px / plot_h if nrow > 1 else 0.0
    return plot_h + top + bottom, vs


def _flag_columns(result) -> List[tuple]:
    """[(column, label), ...] for overall and metrics (summaries first, see registry.sort_key)."""
    cols = [("qa_overall", "Overall (bitmask flags)")]
    for m in result.metrics:
        tag = "" if m.in_bitmask else " [not in bitmask]"
        cols.append((m.name, f"{m.name} ({m.title}){tag}"))
    return cols


def _notes_by_metric(result) -> List[Dict[str, str]]:
    out = []
    names = {m.name for m in result.metrics}
    for r in result.table:
        d: Dict[str, str] = {}
        note = str(r["qa_notes"]) if "qa_notes" in result.table.colnames else ""
        for part in note.split("; "):
            k, _, v = part.partition(": ")
            if k in names:
                d[k] = v
        out.append(d)
    return out


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "nan" if not np.isfinite(v) else f"{v:.4g}"
    return str(v)


def _hover_base(result) -> np.ndarray:
    t = result.table
    obj = t["object"] if "object" in t.colnames else [""] * len(t)
    return np.asarray([f"<b>{int(e)}</b> tile {int(ti)} MJD {int(mj)}<br>{_html.escape(str(o))}"
                       for e, ti, mj, o in zip(t["expnum"], t["tileid"], t["mjd"], obj)], dtype=object)


def _hover_metric(result, m, notes) -> np.ndarray:
    """Short per-exposure description of one metric: flag, main value, note."""
    t = result.table
    out = []
    for i, r in enumerate(t):
        s = f"{m.name}: <b>{r[m.name]}</b>"
        if m.main_value:
            s += f"<br>{m.main_value} = {_fmt(r[m.column(m.main_value)])}"
        n = notes[i].get(m.name)
        if n:
            s += f"<br><i>{_html.escape(n)}</i>"
        out.append(s)
    return np.asarray(out, dtype=object)


def _hover_for_column(result, col, base, per_metric) -> np.ndarray:
    t = result.table
    if col in per_metric:
        return base + "<br>" + per_metric[col]
    return base + np.asarray([f"<br>{col}: <b>{v}</b>" for v in t[col]], dtype=object)


def fig_counts(result):
    import plotly.graph_objects as go
    t = result.table
    cols = _flag_columns(result)[::-1]
    fig = go.Figure()
    for s in STATUS:
        counts = [int(np.sum(np.asarray(t[c]) == s)) for c, _ in cols]
        fig.add_bar(y=[lab for _, lab in cols], x=counts, name=f"{ICON[s]} {s}", orientation="h",
                    marker=dict(color=COLORS[s], line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>" + s + ": %{x}<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    return _layout(fig, "Status counts per metric", height=120 + 34 * len(cols),
                   xaxis_title="number of exposures")


def fig_heatmap(result, base, per_metric, which="all"):
    """Status map; which = 'summary' (overall + summary metrics), 'individual' or 'all'."""
    import plotly.graph_objects as go
    t = result.table
    cols = _flag_columns(result)
    summ = {m.name for m in result.metrics if m.summary}
    if which == "summary":
        cols = [c for c in cols if c[0] == "qa_overall" or c[0] in summ]
    elif which == "individual":
        cols = [c for c in cols if c[0] != "qa_overall" and c[0] not in summ]
    if not cols:
        return None
    code = {s: i for i, s in enumerate(STATUS)}
    z = np.array([[code.get(str(v), 3) for v in t[c]] for c, _ in cols], dtype=float)
    x = [str(int(e)) for e in t["expnum"]]
    text = [list(_hover_for_column(result, c, base, per_metric)) for c, _ in cols]
    cs = []
    for i, s in enumerate(STATUS):
        cs += [[i / 4, COLORS[s]], [(i + 1) / 4, COLORS[s]]]
    gap = 1 if len(t) < 400 else 0
    fig = go.Figure(go.Heatmap(z=z, x=x, y=[lab for _, lab in cols], text=text,
                               hovertemplate="%{text}<extra></extra>", colorscale=cs,
                               zmin=-0.5, zmax=3.5, xgap=gap, ygap=2, showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="category", showticklabels=len(t) <= 80, title="expnum")
    ttl = {"summary": "Overall and summary flags per exposure (hover for components)",
           "individual": "Individual QA flags per exposure (hover for values)"}.get(which, "Status per exposure")
    return _layout(fig, ttl, height=140 + 30 * len(cols), margin=dict(l=60, r=20, t=50, b=50))


def fig_skymap(result, base, per_metric):
    import plotly.graph_objects as go
    t = result.table
    if "sci_ra" not in t.colnames:
        return None
    ra = np.asarray(t["sci_ra"], dtype=float)
    dec = np.asarray(t["sci_dec"], dtype=float)
    cols = _flag_columns(result)
    fig = go.Figure()
    buttons = []
    ntr = len(cols) * len(STATUS)
    for k, (c, lab) in enumerate(cols):
        hv = _hover_for_column(result, c, base, per_metric)
        vals = np.asarray(t[c]).astype(str)
        for s in STATUS:
            sel = vals == s
            fig.add_trace(go.Scattergl(
                x=ra[sel], y=dec[sel], mode="markers", name=f"{ICON[s]} {s}",
                marker=dict(color=COLORS[s], size=9, line=dict(color=SURFACE, width=1)),
                text=hv[sel], hovertemplate="%{text}<extra></extra>", visible=(k == 0),
                legendgroup=s, showlegend=True))
        vis = [False] * ntr
        for i in range(len(STATUS)):
            vis[k * len(STATUS) + i] = True
        buttons.append(dict(label=lab, method="update", args=[{"visible": vis}]))
    fig.update_xaxes(autorange="reversed", title="RA [deg]")
    fig.update_yaxes(title="Dec [deg]")
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.12,
                                        yanchor="bottom", bgcolor="white", bordercolor=GRID)])
    return _layout(fig, "", height=520, margin=dict(l=60, r=20, t=80, b=50))


def fig_values_vs_mjd(result, base, per_metric):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = result.table
    ms = [m for m in result.metrics if m.main_value and not getattr(m, "status", False)
          and m.column(m.main_value) in t.colnames]
    if not ms:
        return None
    titles = [f"{m.name}: {m.column(m.main_value)}" for m in ms]
    H, vs = _grid(len(ms), panel_px=150, gap_px=55)
    fig = make_subplots(rows=len(ms), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=vs)
    mjd = np.asarray(t["mjd"], dtype=float)
    for i, m in enumerate(ms, start=1):
        hv = _hover_for_column(result, m.name, base, per_metric)
        y = np.asarray(t[m.column(m.main_value)], dtype=float)
        flags = np.asarray(t[m.name]).astype(str)
        for s in STATUS:
            sel = (flags == s) & np.isfinite(y)
            if not sel.any():
                continue
            fig.add_trace(go.Scatter(
                x=mjd[sel], y=y[sel], mode="markers", name=f"{ICON[s]} {s}", legendgroup=s,
                showlegend=(i == 1), text=hv[sel], hovertemplate="%{text}<extra></extra>",
                marker=dict(color=COLORS[s], size=8, line=dict(color=SURFACE, width=1))),
                row=i, col=1)
    fig.update_xaxes(title="MJD", row=len(ms), col=1)
    fig.update_xaxes(showticklabels=True, tickformat="d")
    fig.update_annotations(font=dict(size=12, color=INK2), x=0, xanchor="left")
    return _layout(fig, "Main metric values vs MJD", height=H, margin=dict(l=70, r=20, t=90, b=60))


def fig_flux_pm_np(result, base):
    """log10 median parametric vs non-parametric flux per bright line (one point per exposure)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    m = next((m for m in result.metrics if m.name == "qa_fit_consistency"), None)
    if m is None:
        return None
    from .metrics.dap_fit_consistency import line_key
    t = result.table
    lines = [l for l in m.params["lines"] if m.column(f"fnp_{line_key(l)}") in t.colnames]
    if not lines:
        return None
    ncol = 4
    nrow = int(np.ceil(len(lines) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[l.replace("_", " ") for l in lines],
                        horizontal_spacing=0.06, vertical_spacing=0.16)
    flags = np.asarray(t[m.name]).astype(str)
    shown = set()
    for i, l in enumerate(lines):
        r, c = i // ncol + 1, i % ncol + 1
        k = line_key(l)
        fnp = np.asarray(t[m.column(f"fnp_{k}")], dtype=float)
        fpm = np.asarray(t[m.column(f"fpm_{k}")], dtype=float)
        n = np.asarray(t[m.column(f"n_{k}")])
        ok = np.isfinite(fnp) & np.isfinite(fpm) & (fnp > 0) & (fpm > 0)
        if not ok.any():
            continue
        x, y = np.log10(fnp), np.log10(fpm)
        lo = float(np.nanmin(np.r_[x[ok], y[ok]])) - 0.1
        hi = float(np.nanmax(np.r_[x[ok], y[ok]])) + 0.1
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"),
                      row=r, col=c)
        for st in STATUS:
            sel = ok & (flags == st)
            if not sel.any():
                continue
            txt = [f"{b}<br>{l}: N fibers = {int(nn)}<br>log F_np = {xx:.2f}, log F_pm = {yy:.2f}"
                   for b, nn, xx, yy in zip(base[sel], n[sel], x[sel], y[sel])]
            fig.add_trace(go.Scatter(x=x[sel], y=y[sel], mode="markers", name=f"{ICON[st]} {st}",
                                     legendgroup=st, showlegend=st not in shown, text=txt,
                                     hovertemplate="%{text}<extra></extra>",
                                     marker=dict(color=COLORS[st], size=9, line=dict(color=SURFACE, width=1))),
                          row=r, col=c)
            shown.add(st)
    fig.update_xaxes(title_text="log F_np", title_font=dict(size=11))
    fig.update_yaxes(title_text="log F_pm", title_font=dict(size=11))
    fig.update_annotations(font=dict(size=12, color=INK2))
    return _layout(fig, "Median flux of selected fibers per exposure [1e-16 erg/s/cm2], log10",
                   height=140 + 300 * nrow, margin=dict(l=60, r=20, t=90, b=50))


def fig_metrics_vs_obs(result):
    """Status distribution of every non-observational metric, split by an observational flag
    (dropdown). Bars show fractions within each category of the observational flag."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = result.table
    obs = [m for m in result.metrics if m.group == GROUP_OBS]
    other = [m for m in result.metrics if m.group != GROUP_OBS and not getattr(m, "status", False)
             and m.name in t.colnames]
    if not obs or not other or not len(t):
        return None
    ncol = 3
    nrow = int(np.ceil(len(other) / ncol))
    Hg, vsg = _grid(nrow, panel_px=200, gap_px=70, top=110, bottom=50)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[m.name for m in other],
                        horizontal_spacing=0.06, vertical_spacing=vsg)
    buttons = []
    ntr = 0
    for k, om in enumerate(obs):
        ov = np.asarray(t[om.name]).astype(str)
        cats = [s_ for s_ in STATUS if np.any(ov == s_)]
        first = len(fig.data)
        for i, m in enumerate(other):
            r, c = i // ncol + 1, i % ncol + 1
            mv = np.asarray(t[m.name]).astype(str)
            n_cat = np.array([np.sum(ov == cat) for cat in cats], dtype=float)
            for st in STATUS:
                cnt = np.array([np.sum((ov == cat) & (mv == st)) for cat in cats], dtype=float)
                frac = np.divide(cnt, n_cat, out=np.zeros_like(cnt), where=n_cat > 0)
                fig.add_trace(go.Bar(
                    x=[f"{ICON[cat]} {cat}" for cat in cats], y=frac, name=f"{ICON[st]} {st}",
                    legendgroup=st, showlegend=(i == 0 and k == 0), visible=(k == 0),
                    marker=dict(color=COLORS[st], line=dict(color=SURFACE, width=1)),
                    customdata=np.stack([cnt, n_cat], axis=1),
                    hovertemplate=(f"{om.name} = %{{x}}<br>{m.name}: {st}<br>"
                                   "%{customdata[0]:.0f} of %{customdata[1]:.0f} exposures (%{y:.0%})<extra></extra>")),
                    row=r, col=c)
        ntr_k = len(fig.data) - first
        buttons.append((om.name, first, ntr_k))
        ntr = len(fig.data)
    menu = []
    for name, first, n_k in buttons:
        vis = [False] * ntr
        for j in range(first, first + n_k):
            vis[j] = True
        menu.append(dict(label=f"split by {name}", method="update", args=[{"visible": vis}]))
    fig.update_layout(barmode="stack",
                      updatemenus=[dict(buttons=menu, x=0, xanchor="left", y=1.0 + 60 / (Hg - 160),
                                        yanchor="bottom", bgcolor="white", bordercolor=GRID)])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_annotations(font=dict(size=12, color=INK2))
    H = Hg
    _layout(fig, "", height=H, margin=dict(l=60, r=20, t=110, b=50))
    fig.update_layout(legend=dict(orientation="h", y=1.0 + 60 / H, yanchor="bottom", x=1, xanchor="right"))
    return fig


def fig_distributions(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = result.table
    ms = [m for m in result.metrics if m.main_value and not getattr(m, "status", False)
          and m.name in t.colnames and m.column(m.main_value) in t.colnames]
    if not ms:
        return None
    ncol = 3
    nrow = int(np.ceil(len(ms) / ncol))
    H, vs = _grid(nrow, panel_px=190, gap_px=80)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[f"{m.name}: {m.column(m.main_value)}" for m in ms],
                        horizontal_spacing=0.07, vertical_spacing=vs)
    for i, m in enumerate(ms):
        r, c = i // ncol + 1, i % ncol + 1
        y = np.asarray(t[m.column(m.main_value)], dtype=float)
        flags = np.asarray(t[m.name]).astype(str)
        for s in STATUS:
            sel = (flags == s) & np.isfinite(y)
            if not sel.any():
                continue
            fig.add_trace(go.Histogram(x=y[sel], name=f"{ICON[s]} {s}", legendgroup=s,
                                       showlegend=(i == 0), marker=dict(color=COLORS[s],
                                       line=dict(color=SURFACE, width=1)), nbinsx=40),
                          row=r, col=c)
    fig.update_layout(barmode="stack")
    fig.update_annotations(font=dict(size=11, color=INK2))
    return _layout(fig, "Distributions of main metric values", height=H, margin=dict(l=60, r=20, t=90, b=60))


CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _timing_components(result):
    """[(column, label), ...]: I/O first, then metric compute times."""
    t = result.table
    comps = [("time_io_sframe", "I/O: SFrame"), ("time_io_dap", "I/O: DAP"),
             ("time_io_dap_output", "I/O: DAP output"),
             ("time_shared_skylines", "Shared: sky-line fits")]
    comps += [(f"time_{m.short}", m.name) for m in result.metrics]
    return [(c, l) for c, l in comps if c in t.colnames]


def _with_inputs(result) -> np.ndarray:
    """Exposures for which input files were actually read."""
    t = result.table
    io = np.zeros(len(t))
    for c in ("time_io_sframe", "time_io_dap", "time_io_dap_output"):
        if c in t.colnames:
            io += np.nan_to_num(np.asarray(t[c], dtype=float))
    return io > 0


def timing_stats(result) -> List[Dict[str, str]]:
    t = result.table[_with_inputs(result)]
    rows = []
    for c, lab in [("time_total", "Total per exposure")] + _timing_components(result):
        if c not in t.colnames:
            continue
        v = np.asarray(t[c], dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        tot = float(np.nansum(np.asarray(t["time_total"], dtype=float)))
        rows.append({"Component": lab, "median [s]": f"{np.median(v):.3f}",
                     "mean [s]": f"{np.mean(v):.3f}", "p90 [s]": f"{np.percentile(v, 90):.3f}",
                     "max [s]": f"{np.max(v):.3f}", "sum [s]": f"{np.sum(v):.1f}",
                     "share of total": f"{np.sum(v) / tot:.1%}" if tot > 0 else ""})
    return rows


def fig_timing_box(result):
    import plotly.graph_objects as go
    t = result.table[_with_inputs(result)]
    comps = _timing_components(result)
    if not comps:
        return None
    fig = go.Figure()
    expn = np.asarray([str(int(e)) for e in t["expnum"]])
    for i, (c, lab) in enumerate([("time_total", "Total per exposure")] + comps):
        v = np.asarray(t[c], dtype=float)
        ok = np.isfinite(v) & (v > 0)
        color = INK2 if i == 0 else CATEGORICAL[(i - 1) % len(CATEGORICAL)]
        fig.add_trace(go.Box(x=np.maximum(v[ok], 1e-5), y=[lab] * int(ok.sum()), name=lab,
                             orientation="h", marker=dict(color=color, size=5),
                             line=dict(color=color, width=1.5), boxpoints="all", jitter=0.4,
                             pointpos=0, text=expn[ok],
                             hovertemplate="expnum %{text}<br>%{x:.3f} s<extra>" + lab + "</extra>",
                             showlegend=False))
    fig.update_xaxes(type="log", title="time [s] (log)")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Time per exposure: total, input I/O and compute time of each metric",
                   height=140 + 46 * (len(comps) + 1))


def fig_timing_exposures(result):
    import plotly.graph_objects as go
    t = result.table
    comps = _timing_components(result)
    if not comps:
        return None
    x = [str(int(e)) for e in t["expnum"]]
    fig = go.Figure()
    if len(t) <= 600:
        for i, (c, lab) in enumerate(comps[:8]):
            fig.add_bar(x=x, y=np.asarray(t[c], dtype=float), name=lab,
                        marker=dict(color=CATEGORICAL[i], line=dict(color=SURFACE, width=1)),
                        hovertemplate="expnum %{x}<br>" + lab + ": %{y:.3f} s<extra></extra>")
        fig.update_layout(barmode="stack", legend_traceorder="normal")
        fig.update_xaxes(type="category", showticklabels=len(t) <= 80)
    else:
        fig.add_trace(go.Scattergl(x=x, y=np.asarray(t["time_total"], dtype=float), mode="markers",
                                   marker=dict(color=CATEGORICAL[0], size=4), name="total"))
    fig.update_xaxes(title="expnum")
    fig.update_yaxes(title="time [s]")
    return _layout(fig, "Time budget per exposure", height=420)


def _table_html(rows: List[Dict[str, str]], cols: List[str], status_cols=()) -> str:
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in cols)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            v = str(r.get(c, ""))
            if c in status_cols and v in COLORS:
                tds.append(f'<td><span class="pill" style="--c:{COLORS[v]}">{ICON[v]} {v}</span></td>')
            else:
                tds.append(f"<td>{_html.escape(v)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; }
body { margin:0; background:var(--page); color:var(--ink); font-family:%s; }
main { max-width:1280px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px; }
.meta { color:var(--ink2); font-size:13px; line-height:1.6; }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
.tile { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:12px 16px; min-width:140px; }
.tile .v { font-size:28px; font-weight:600; } .tile .l { font-size:13px; color:var(--ink2); }
.tile .dot { display:inline-block; width:10px; height:10px; border-radius:50%%; margin-right:6px; }
.card { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:8px; margin:12px 0; }
.tbl { overflow-x:auto; max-height:520px; overflow-y:auto; background:var(--surface); border:1px solid var(--grid); border-radius:8px; }
table { border-collapse:collapse; font-size:12px; width:100%%; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { position:sticky; top:0; background:#f3f2ee; color:var(--ink2); }
.pill { white-space:nowrap; } .pill::before { content:""; display:inline-block; width:8px; height:8px; border-radius:50%%; background:var(--c); margin-right:4px; }
code { font-size:12px; }
""" % FONT


def fig_fluxmap(result, base):
    """Multi-tile flux consistency: tile centres coloured by the fitted flux-scale offset."""
    import plotly.graph_objects as go
    t = result.table
    m = next((m for m in result.metrics if m.name == "qa_fluxmap"), None)
    if m is None or "sci_ra" not in t.colnames:
        return None
    s = np.asarray(t[m.column("s_dex")], dtype=float)
    ok = np.isfinite(s)
    if not ok.any():
        return None
    flags = np.asarray(t[m.name]).astype(str)
    ra, dec = np.asarray(t["sci_ra"], float), np.asarray(t["sci_dec"], float)
    # show the region of the solved tiles only (+-1 deg); far-away unsolved exposures are omitted
    r0, r1 = np.nanmin(ra[ok]) - 1.0, np.nanmax(ra[ok]) + 1.0
    d0, d1 = np.nanmin(dec[ok]) - 1.0, np.nanmax(dec[ok]) + 1.0
    inside = (ra >= r0) & (ra <= r1) & (dec >= d0) & (dec <= d1)
    fig = go.Figure()
    sym = {"GOOD": "circle", "WARNING": "diamond", "MAJOR ISSUE": "x", "UNDEFINED": "circle-open"}
    for st in STATUS:
        sel = (flags == st) & inside
        if not sel.any():
            continue
        txt = [f"{b}<br>{m.name}: {st}<br>s = {v:+.3f} dex ({10 ** v if np.isfinite(v) else np.nan:.2f}x)"
               f"<br>edges {int(ne)} (rejected {int(nr)})"
               for b, v, ne, nr in zip(base[sel], s[sel], np.asarray(t[m.column("n_edges")])[sel],
                                       np.asarray(t[m.column("n_rejected")])[sel])]
        fig.add_trace(go.Scatter(
            x=np.asarray(t["sci_ra"], float)[sel], y=np.asarray(t["sci_dec"], float)[sel],
            mode="markers", name=f"{ICON[st]} {st}", text=txt, hovertemplate="%{text}<extra></extra>",
            marker=dict(size=26, symbol=sym[st], color=np.nan_to_num(s[sel]), cmin=-0.1, cmax=0.1,
                        colorscale="RdBu_r", line=dict(color=COLORS[st], width=3),
                        colorbar=dict(title="s [dex]") if st == "GOOD" else None)))
    fig.update_xaxes(title="RA [deg]", autorange="reversed")
    fig.update_yaxes(title="Dec [deg]", scaleanchor="x",
                     scaleratio=1.0 / max(np.cos(np.deg2rad(np.nanmedian(dec[ok]))), 0.05))
    return _layout(fig, "Flux-scale offsets from tile overlaps (qa_fluxmap; colour = s, outline = status)",
                   height=560)


def fig_tile_status(result):
    """Tile status (information) as a map: rows = status_* columns, columns = exposures."""
    import plotly.graph_objects as go
    sm = [m for m in result.metrics if m.status]
    t = result.table
    if not sm or not len(t):
        return None
    m = sm[0]
    cols = [m.column(v.name) for v in m.values if v.dtype == "bool"]
    extra = {"status_auroral": ("frac_auroral", "best_auroral"), "status_bright": ("frac_bright", None),
             "status_faint": ("frac_faint", "best_faint"), "status_sp": ("n_sp_regions", None)}
    x = [str(int(e)) for e in t["expnum"]]
    z, text = [], []
    for c in cols:
        v = np.asarray(t[c]).astype(bool)
        z.append(v.astype(float))
        fx, bx = extra.get(c, (None, None))
        row = []
        for i, r in enumerate(t):
            h = f"expnum {r['expnum']} tile {r['tileid']} mjd {r['mjd']}<br>{r['object']}<br><b>{c} = {bool(v[i])}</b>"
            if fx and m.column(fx) in t.colnames:
                h += f"<br>{fx}: {_fmt(r[m.column(fx)])}"
            if bx and m.column(bx) in t.colnames:
                h += f"<br>{bx}: {_fmt(r[m.column(bx)])}"
            row.append(h)
        text.append(row)
    cs = [[0, "#e7e6df"], [0.5, "#e7e6df"], [0.5, "#2a78d6"], [1, "#2a78d6"]]
    gap = 1 if len(t) < 400 else 0
    fig = go.Figure(go.Heatmap(z=np.array(z), x=x, y=[c.replace("status_", "") for c in cols], text=text,
                               hovertemplate="%{text}<extra></extra>", colorscale=cs, zmin=0, zmax=1,
                               xgap=gap, ygap=2, showscale=False))
    for lab, col in (("True", "#2a78d6"), ("False", "#e7e6df")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab,
                                 marker=dict(color=col, size=12, symbol="square")))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="category", showticklabels=len(t) <= 80, title="expnum")
    ntrue = ", ".join(f"{c.replace('status_', '')} {int(np.sum(np.asarray(t[c]).astype(bool)))}" for c in cols)
    return _layout(fig, f"Tile status (information, not QA flags); exposures with True: {ntrue}",
                   height=140 + 40 * len(cols))


def status_table_html(result) -> str:
    sm = [m for m in result.metrics if m.status]
    t = result.table
    if not sm or not len(t):
        return ""
    cols = [m.column(v.name) for m in sm for v in m.values if v.dtype == "bool"]
    rows = [{"expnum": str(r["expnum"]), "tileid": str(r["tileid"]), "object": str(r["object"]),
             **{c: ("True" if bool(r[c]) else "False") for c in cols}} for r in t]
    tot = {c: int(np.sum(np.asarray(t[c], bool))) for c in cols}
    head = "<p class='meta'>Information per tile (not QA flags). Number of exposures with True: " + \
        ", ".join(f"<code>{c}</code> {tot[c]}" for c in cols) + "</p>"
    return head + _table_html(rows, ["expnum", "tileid", "object"] + cols)


# ============================================================================ compact report (v1.15)
# For large runs (>= DETAIL_LIMIT exposures; up to ~40000) the per-exposure plots are replaced by
# aggregated views: per-night status maps, one sky marker per tile, nightly medians with bands,
# precomputed histograms / 2D densities. Numeric arrays are passed as numpy arrays (plotly binary
# encoding) and hover texts are built with hovertemplate + numeric customdata. An exposure lookup
# (bitmask decoded in the browser) and tables of the worst exposures per metric replace the table
# of all flagged exposures.
DETAIL_LIMIT = 500
RANK = {"GOOD": 0, "UNDEFINED": 1, "WARNING": 2, "MAJOR ISSUE": 3}   # severity (worst-of)
RANK_COLORS = [[0.0, COLORS["GOOD"]], [1 / 6, COLORS["GOOD"]], [1 / 6, COLORS["UNDEFINED"]], [0.5, COLORS["UNDEFINED"]],
               [0.5, COLORS["WARNING"]], [5 / 6, COLORS["WARNING"]], [5 / 6, COLORS["MAJOR ISSUE"]],
               [1.0, COLORS["MAJOR ISSUE"]]]
SCORE_COLORS = [[0.0, COLORS["GOOD"]], [0.25, "#9ccf3a"], [0.5, COLORS["WARNING"]], [1.0, COLORS["MAJOR ISSUE"]]]


def _ranges(vals) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10' (exact values of the request, compactly)."""
    v = sorted({int(x) for x in vals})
    if not v:
        return ""
    out, a, b = [], v[0], v[0]
    for x in v[1:] + [None]:
        if x is not None and x == b + 1:
            b = x
            continue
        out.append(str(a) if a == b else (f"{a}, {b}" if b == a + 1 else f"{a}-{b}"))
        if x is not None:
            a = b = x
    return ", ".join(out)


def selection_text(sel) -> str:
    parts = []
    for k, v in (sel or {}).items():
        if v is None or v is False or (hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 0):
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) and all(isinstance(x, (int, np.integer)) for x in v):
            parts.append(f"{k}: {_ranges(v)}")
        elif v is True:
            parts.append(k)
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts) or "all drpall exposures"


def _codes(col) -> np.ndarray:
    """Status labels -> codes 0..3 (STATUS order: GOOD, WARNING, MAJOR ISSUE, UNDEFINED)."""
    lab = np.asarray(col).astype(str)
    out = np.full(lab.shape, 3, np.int8)
    for i, s in enumerate(STATUS):
        out[lab == s] = i
    return out


def _nights(t):
    mjd = np.asarray(t["mjd"]).astype(int)
    u, idx = np.unique(mjd, return_inverse=True)
    return u, idx


def fig_night_status(result, which="summary"):
    """Rows = metrics, columns = nights; colour = (n_WARNING + 2 n_MAJOR) / (2 n_evaluated)."""
    import plotly.graph_objects as go
    t = result.table
    cols = _flag_columns(result)
    summ = {m.name for m in result.metrics if m.summary}
    cols = ([c for c in cols if c[0] == "qa_overall" or c[0] in summ] if which == "summary"
            else [c for c in cols if c[0] != "qa_overall" and c[0] not in summ])
    if not cols:
        return None
    u, idx = _nights(t)
    nn = len(u)
    z = np.full((len(cols), nn), np.nan, np.float32)
    cd = np.zeros((len(cols), nn, 5), np.int32)
    for r, (c, _) in enumerate(cols):
        cnt = np.bincount(idx * 4 + _codes(t[c]), minlength=nn * 4).reshape(nn, 4)
        n = cnt.sum(1)
        ev = n - cnt[:, 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            z[r] = np.where(ev > 0, (cnt[:, 1] + 2 * cnt[:, 2]) / (2.0 * ev), np.nan)
        cd[r] = np.c_[n, cnt]
    fig = go.Figure(go.Heatmap(
        z=z, x=u.astype(str), y=[lab for _, lab in cols], customdata=cd, colorscale=SCORE_COLORS, zmin=0, zmax=1,
        xgap=1 if nn < 300 else 0, ygap=2,
        colorbar=dict(title="(W+2M)/2N", thickness=10, len=0.8),
        hovertemplate=("MJD %{x}<br>%{y}<br>%{customdata[0]} exposures: %{customdata[1]} G, %{customdata[2]} W, "
                       "%{customdata[3]} M, %{customdata[4]} U<extra></extra>")))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="category", title="night (MJD)", showticklabels=nn <= 60)
    ttl = {"summary": "Overall and summary flags per night", "individual": "Individual QA flags per night"}[which]
    return _layout(fig, ttl + " (colour: 0 = all GOOD, 0.5 = all WARNING, 1 = all MAJOR; hover for counts)",
                   height=150 + 30 * len(cols))


def fig_tiles_sky(result):
    """One marker per tile pointing (median position of its exposures), worst status of the selected
    flag column; the dropdown swaps only the colour array."""
    import plotly.graph_objects as go
    t = result.table
    if "sci_ra" not in t.colnames or not len(t):
        return None
    tid = np.asarray(t["tileid"]).astype(np.int64)
    ra, dec = np.asarray(t["sci_ra"], float), np.asarray(t["sci_dec"], float)
    ut, ti = np.unique(tid, return_inverse=True)
    nt = len(ut)
    n = np.bincount(ti, minlength=nt)
    # median position per tile (RA wrap around the tile's first exposure)
    order = np.argsort(ti, kind="stable")
    starts = np.r_[0, np.cumsum(n)[:-1]]
    tra, tde = np.empty(nt), np.empty(nt)
    for k in range(nt):
        sl = order[starts[k]:starts[k] + n[k]]
        r0 = ra[sl[0]]
        tra[k] = (np.nanmedian((ra[sl] - r0 + 180.0) % 360.0 - 180.0) + r0) % 360.0
        tde[k] = np.nanmedian(dec[sl])
    cols = _flag_columns(result)
    ranks = []
    for c, _ in cols:
        lab = np.asarray(t[c]).astype(str)
        rk = np.zeros(len(t), np.int8)
        for s_, v in RANK.items():
            rk[lab == s_] = v
        w = np.zeros(nt, np.int8)
        np.maximum.at(w, ti, rk)
        ranks.append(w)
    ok = np.isfinite(tra) & np.isfinite(tde)
    fig = go.Figure(go.Scattergl(
        x=tra[ok].astype(np.float32), y=tde[ok].astype(np.float32), mode="markers",
        marker=dict(color=ranks[0][ok], colorscale=RANK_COLORS, cmin=0, cmax=3, size=5, showscale=False),
        customdata=np.c_[ut[ok], n[ok]].astype(np.int64),
        hovertemplate="tile %{customdata[0]} (%{customdata[1]} exp.)<br>RA %{x:.2f}, Dec %{y:.2f}<extra></extra>",
        showlegend=False))
    for s_ in ("GOOD", "UNDEFINED", "WARNING", "MAJOR ISSUE"):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{ICON[s_]} {s_}",
                                 marker=dict(color=COLORS[s_], size=10)))
    buttons = [dict(label=lab, method="restyle", args=[{"marker.color": [w[ok].tolist()]}, [0]])
               for (c, lab), w in zip(cols, ranks)]
    fig.update_xaxes(autorange="reversed", title="RA [deg]")
    fig.update_yaxes(title="Dec [deg]")
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.12, yanchor="bottom",
                                        bgcolor="white", bordercolor=GRID)])
    return _layout(fig, "", height=560, margin=dict(l=60, r=20, t=80, b=50))


def fig_values_nightly(result, max_points=300):
    """Main value of every metric: nightly median with the 16-84% band; MAJOR exposures as points
    (at most max_points per metric, the largest deviations from their night median)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = result.table
    ms = [m for m in result.metrics if m.main_value and not getattr(m, "status", False)
          and m.column(m.main_value) in t.colnames]
    if not ms:
        return None
    u, idx = _nights(t)
    H, vs = _grid(len(ms), panel_px=150, gap_px=55)
    fig = make_subplots(rows=len(ms), cols=1, shared_xaxes=True, vertical_spacing=vs,
                        subplot_titles=[f"{m.name}: {m.column(m.main_value)}" for m in ms])
    expn = np.asarray(t["expnum"]).astype(np.int64)
    mjd = np.asarray(t["mjd"], float)
    for i, m in enumerate(ms, start=1):
        try:
            y = np.asarray(t[m.column(m.main_value)], dtype=float)
        except (TypeError, ValueError):
            continue
        fin = np.isfinite(y)
        if not fin.any():
            continue
        q = np.full((len(u), 4), np.nan)
        order = np.lexsort((y, idx))
        for k in np.unique(idx[fin]):
            v = y[fin & (idx == k)]
            q[k] = (np.percentile(v, 16), np.median(v), np.percentile(v, 84), v.size)
        okn = np.isfinite(q[:, 1])
        x = u[okn].astype(np.float32)
        fig.add_trace(go.Scatter(x=x, y=q[okn, 2].astype(np.float32), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=q[okn, 0].astype(np.float32), mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(42,120,214,0.20)", hoverinfo="skip",
                                 name="16-84% per night", legendgroup="band", showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=q[okn, 1].astype(np.float32), mode="lines+markers",
                                 line=dict(color="#2a78d6", width=1), marker=dict(size=3),
                                 customdata=q[okn][:, [0, 2, 3]].astype(np.float32),
                                 hovertemplate=("MJD %{x:d}: median %{y:.4g} (16-84%: %{customdata[0]:.4g} .. "
                                                "%{customdata[1]:.4g}, n=%{customdata[2]:.0f})<extra></extra>"),
                                 name="nightly median", legendgroup="med", showlegend=(i == 1)), row=i, col=1)
        maj = fin & (np.asarray(t[m.name]).astype(str) == "MAJOR ISSUE")
        if maj.any():
            sel = np.where(maj)[0]
            if sel.size > max_points:
                dev = np.abs(y[sel] - q[idx[sel], 1])
                sel = sel[np.argsort(-dev)[:max_points]]
            fig.add_trace(go.Scattergl(x=mjd[sel].astype(np.float32), y=y[sel].astype(np.float32), mode="markers",
                                       marker=dict(color=COLORS["MAJOR ISSUE"], size=5),
                                       customdata=expn[sel], name=f"{ICON['MAJOR ISSUE']} MAJOR (<= {max_points})",
                                       legendgroup="maj", showlegend=(i == 1),
                                       hovertemplate="expnum %{customdata}<br>%{y:.4g}<extra></extra>"),
                          row=i, col=1)
    fig.update_xaxes(title="MJD", row=len(ms), col=1)
    fig.update_xaxes(showticklabels=True, tickformat="d")
    fig.update_annotations(font=dict(size=12, color=INK2), x=0, xanchor="left")
    return _layout(fig, "Main metric values per night (median, 16-84% band) and MAJOR exposures", height=H,
                   margin=dict(l=70, r=20, t=90, b=60))


def fig_flux_pm_np_density(result, nb=60):
    """log10 parametric vs non-parametric median flux per bright line: 2D density of exposures."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    m = next((m for m in result.metrics if m.name == "qa_fit_consistency"), None)
    if m is None:
        return None
    from .metrics.dap_fit_consistency import line_key
    t = result.table
    lines = [l for l in m.params["lines"] if m.column(f"fnp_{line_key(l)}") in t.colnames]
    if not lines:
        return None
    ncol = 4
    nrow = int(np.ceil(len(lines) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[l.replace("_", " ") for l in lines],
                        horizontal_spacing=0.06, vertical_spacing=0.16)
    for i, l in enumerate(lines):
        r, c = i // ncol + 1, i % ncol + 1
        k = line_key(l)
        fnp = np.asarray(t[m.column(f"fnp_{k}")], dtype=float)
        fpm = np.asarray(t[m.column(f"fpm_{k}")], dtype=float)
        ok = np.isfinite(fnp) & np.isfinite(fpm) & (fnp > 0) & (fpm > 0)
        if ok.sum() < 2:
            continue
        x, y = np.log10(fnp[ok]), np.log10(fpm[ok])
        xy = np.r_[x, y]
        lo, hi = (float(v) for v in np.percentile(xy, [0.5, 99.5]))   # outliers in the edge bins
        lo, hi = lo - 0.05, hi + 0.05
        x, y = np.clip(x, lo, hi - 1e-9), np.clip(y, lo, hi - 1e-9)
        e = np.linspace(lo, hi, nb + 1)
        h, _, _ = np.histogram2d(x, y, bins=[e, e])
        cen = (0.5 * (e[1:] + e[:-1])).astype(np.float32)
        with np.errstate(divide="ignore"):
            lz = np.where(h.T > 0, np.log10(np.maximum(h.T, 1)), np.nan).astype(np.float32)
        fig.add_trace(go.Heatmap(x=cen, y=cen, z=lz,
                                 colorscale="Viridis", showscale=(i == 0),
                                 colorbar=dict(title="log N", thickness=10) if i == 0 else None,
                                 customdata=h.T.astype(np.int32),
                                 hovertemplate="log F_np %{x:.2f}, log F_pm %{y:.2f}: %{customdata} exp.<extra></extra>"),
                      row=r, col=c)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"), row=r, col=c)
    fig.update_xaxes(title_text="log F_np", title_font=dict(size=11))
    fig.update_yaxes(title_text="log F_pm", title_font=dict(size=11))
    fig.update_annotations(font=dict(size=12, color=INK2))
    return _layout(fig, "Median flux of selected fibers per exposure [1e-16 erg/s/cm2], log10 (density of exposures)",
                   height=140 + 300 * nrow, margin=dict(l=60, r=20, t=90, b=50))


def fig_distributions_compact(result, nbins=40):
    """Histograms of the main values (1-99% range, outliers in the end bins), counts precomputed."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    t = result.table
    ms = [m for m in result.metrics if m.main_value and not getattr(m, "status", False)
          and m.name in t.colnames and m.column(m.main_value) in t.colnames]
    if not ms:
        return None
    ncol = 3
    nrow = int(np.ceil(len(ms) / ncol))
    H, vs = _grid(nrow, panel_px=190, gap_px=80)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[f"{m.name}: {m.column(m.main_value)}" for m in ms],
                        horizontal_spacing=0.07, vertical_spacing=vs)
    for i, m in enumerate(ms):
        r, c = i // ncol + 1, i % ncol + 1
        try:
            y = np.asarray(t[m.column(m.main_value)], dtype=float)
        except (TypeError, ValueError):
            continue
        flags = np.asarray(t[m.name]).astype(str)
        fin = np.isfinite(y)
        if not fin.any():
            continue
        lo, hi = np.percentile(y[fin], [1, 99])
        if not hi > lo:
            lo, hi = lo - 0.5, hi + 0.5
        e = np.linspace(lo, hi, nbins + 1)
        cen = (0.5 * (e[1:] + e[:-1])).astype(np.float32)
        for s in STATUS:
            sel = (flags == s) & fin
            if not sel.any():
                continue
            h, _ = np.histogram(np.clip(y[sel], lo, hi), bins=e)
            fig.add_trace(go.Bar(x=cen, y=h.astype(np.int32), name=f"{ICON[s]} {s}", legendgroup=s,
                                 showlegend=(i == 0), marker=dict(color=COLORS[s], line=dict(width=0)),
                                 hovertemplate=f"{s}: %{{y}} exp. near %{{x:.4g}}<extra></extra>"),
                          row=r, col=c)
    fig.update_layout(barmode="stack", bargap=0)
    fig.update_annotations(font=dict(size=11, color=INK2))
    return _layout(fig, "Distributions of main metric values (1-99% range; outliers in the end bins)", height=H,
                   margin=dict(l=60, r=20, t=90, b=60))


def fig_fluxmap_compact(result):
    """Exposure positions coloured by the qa_fluxmap local offset; WARNING / MAJOR outlined."""
    import plotly.graph_objects as go
    t = result.table
    m = next((m for m in result.metrics if m.name == "qa_fluxmap"), None)
    if m is None or "sci_ra" not in t.colnames:
        return None
    col = m.column("local_dex") if m.column("local_dex") in t.colnames else m.column("s_dex")
    if col not in t.colnames:
        return None
    lv = np.asarray(t[col], dtype=float)
    ok = np.isfinite(lv)
    if not ok.any():
        return None
    ra, dec = np.asarray(t["sci_ra"], float), np.asarray(t["sci_dec"], float)
    expn = np.asarray(t["expnum"]).astype(np.int64)
    flags = np.asarray(t[m.name]).astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ra[ok].astype(np.float32), y=dec[ok].astype(np.float32), mode="markers",
                               marker=dict(color=np.clip(lv[ok], -0.15, 0.15).astype(np.float32), cmin=-0.15,
                                           cmax=0.15, colorscale="RdBu_r", size=5, colorbar=dict(title="l [dex]")),
                               customdata=np.c_[expn[ok], lv[ok]], name="all",
                               hovertemplate="expnum %{customdata[0]:.0f}: l = %{customdata[1]:+.3f} dex<extra></extra>"))
    for st in ("WARNING", "MAJOR ISSUE"):
        sel = ok & (flags == st)
        if sel.any():
            fig.add_trace(go.Scattergl(x=ra[sel].astype(np.float32), y=dec[sel].astype(np.float32), mode="markers",
                                       marker=dict(color="rgba(0,0,0,0)", size=9, line=dict(color=COLORS[st], width=1.5)),
                                       customdata=np.c_[expn[sel], lv[sel]], name=f"{ICON[st]} {st} ({sel.sum()})",
                                       hovertemplate="expnum %{customdata[0]:.0f}: l = %{customdata[1]:+.3f} dex<extra></extra>"))
    fig.update_xaxes(title="RA [deg]", autorange="reversed")
    fig.update_yaxes(title="Dec [deg]")
    return _layout(fig, "qa_fluxmap: local Halpha flux-scale offset l of every exposure (outline = status)", height=560)


def fig_tile_status_nightly(full):
    """Tile status (information): fraction of exposures with True per night."""
    import plotly.graph_objects as go
    sm = [m for m in full.metrics if m.status]
    t = full.table
    if not sm or not len(t):
        return None
    m = sm[0]
    cols = [m.column(v.name) for v in m.values if v.dtype == "bool" and m.column(v.name) in t.colnames]
    if not cols:
        return None
    u, idx = _nights(t)
    n = np.bincount(idx, minlength=len(u))
    z = np.zeros((len(cols), len(u)), np.float32)
    cd = np.zeros((len(cols), len(u), 2), np.int32)
    for r, c in enumerate(cols):
        k = np.bincount(idx, weights=np.asarray(t[c]).astype(bool).astype(float), minlength=len(u))
        z[r] = k / np.maximum(n, 1)
        cd[r] = np.c_[k, n]
    fig = go.Figure(go.Heatmap(z=z, x=u.astype(str), y=[c.replace("status_", "") for c in cols], customdata=cd,
                               colorscale=[[0, "#e7e6df"], [1, "#2a78d6"]], zmin=0, zmax=1, xgap=0, ygap=2,
                               colorbar=dict(title="fraction True", thickness=10),
                               hovertemplate="MJD %{x}<br>%{y}: %{customdata[0]} of %{customdata[1]} exposures<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="category", title="night (MJD)", showticklabels=len(u) <= 60)
    ntrue = ", ".join(f"{c.replace('status_', '')} {int(np.sum(np.asarray(t[c]).astype(bool)))}" for c in cols)
    return _layout(fig, f"Tile status per night (information, not QA flags); exposures with True: {ntrue}",
                   height=150 + 40 * len(cols))


def lookup_html(result) -> str:
    """Exposure lookup: the qa_bitmask of every exposure is decoded in the browser (BigInt)."""
    import json
    t = result.table
    if "qa_bitmask" not in t.colnames or not len(t):
        return ""
    slots = [(m.name, int(m.slot)) for m in result.metrics if getattr(m, "slot", None) is not None
             and getattr(m, "in_bitmask", True)]
    data = dict(e=[int(x) for x in t["expnum"]], t=[int(x) for x in t["tileid"]], m=[int(x) for x in t["mjd"]],
                b=[format(int(x) & ((1 << 64) - 1), "x") for x in t["qa_bitmask"]],
                o="".join(str(int(c)) for c in _codes(t["qa_overall"])) if "qa_overall" in t.colnames else "",
                s=slots, L=STATUS, C=[COLORS[s] for s in STATUS])
    js = """
<div class="card" style="padding:12px">
<label>expnum: <input id="lk-in" type="number" style="width:140px" placeholder="e.g. 12345"></label>
<span id="lk-msg" class="meta"></span><div id="lk-out"></div></div>
<script>
(function(){const D=%s;const ix=new Map();D.e.forEach((e,i)=>ix.set(e,i));
const pill=(c)=>'<span class="pill" style="--c:'+D.C[c]+'">'+D.L[c]+'</span>';
function show(){const v=parseInt(document.getElementById('lk-in').value);const i=ix.get(v);
const out=document.getElementById('lk-out'),msg=document.getElementById('lk-msg');
if(i===undefined){out.innerHTML='';msg.textContent=isNaN(v)?'':' not in this run';return;}
const bm=BigInt('0x'+D.b[i]);msg.textContent=' tile '+D.t[i]+', MJD '+D.m[i]+', qa_bitmask '+bm.toString();
let h='<table><thead><tr><th>metric</th><th>status</th></tr></thead><tbody>';
if(D.o)h+='<tr><td><b>qa_overall</b></td><td>'+pill(+D.o[i])+'</td></tr>';
for(const [n,s] of D.s){const c=Number((bm>>BigInt(2*s))&3n);h+='<tr><td>'+n+'</td><td>'+pill(c)+'</td></tr>';}
out.innerHTML=h+'</tbody></table>';}
document.getElementById('lk-in').addEventListener('input',show);})();
</script>""" % json.dumps(data, separators=(",", ":"))
    return js


def worst_tables_html(result, notes, n=20) -> str:
    """Per metric: the n worst exposures (MAJOR before WARNING, then largest |main value|) with notes."""
    t = result.table
    out = []
    expn = np.asarray(t["expnum"]).astype(int)
    for m in result.metrics:
        if m.name not in t.colnames:
            continue
        lab = np.asarray(t[m.name]).astype(str)
        nm, nw = int(np.sum(lab == "MAJOR ISSUE")), int(np.sum(lab == "WARNING"))
        if not nm + nw:
            continue
        mv = np.full(len(t), np.nan)
        if m.main_value and m.column(m.main_value) in t.colnames:
            try:
                mv = np.abs(np.asarray(t[m.column(m.main_value)], dtype=float))
            except (TypeError, ValueError):
                pass
        sev = np.where(lab == "MAJOR ISSUE", 2, np.where(lab == "WARNING", 1, 0))
        cand = np.where(sev > 0)[0]
        cand = cand[np.lexsort((-np.nan_to_num(mv[cand], nan=-np.inf), -sev[cand]))][:n]
        rows = [{"expnum": str(expn[i]), "tileid": str(t["tileid"][i]), "mjd": str(t["mjd"][i]),
                 "object": str(t["object"][i]) if "object" in t.colnames else "", "status": lab[i],
                 (m.main_value or "value"): _fmt(t[m.column(m.main_value)][i]) if m.main_value and
                 m.column(m.main_value) in t.colnames else "", "note": notes[i].get(m.name, "")} for i in cand]
        cols = ["expnum", "tileid", "mjd", "object", "status", m.main_value or "value", "note"]
        out.append(f"<details><summary><code>{m.name}</code> &mdash; {nm} MAJOR, {nw} WARNING</summary>"
                   + _table_html(rows, cols, status_cols={"status"}) + "</details>")
    return "".join(out)


def write_html(result, path, detail=None) -> Path:
    """HTML report. detail=None: detailed per-exposure plots for runs with < DETAIL_LIMIT exposures,
    the compact (aggregated) layout otherwise; True / False force one of them."""
    import copy
    full = result
    result = copy.copy(full)
    result.metrics = [m for m in full.metrics if not m.status]   # flag metrics only
    t = result.table
    cfg = result.config
    detail = (len(t) < DETAIL_LIMIT) if detail is None else bool(detail)
    figs = []
    notes = _notes_by_metric(result) if len(t) else []
    if len(t) and detail:
        base = _hover_base(result)
        per_metric = {m.name: _hover_metric(result, m, notes) for m in result.metrics}
        figs = [("Status counts", fig_counts(result)),
                ("Status map: overall and summaries", fig_heatmap(result, base, per_metric, "summary")),
                ("Status map: individual QA flags", fig_heatmap(result, base, per_metric, "individual")),
                ("Sky map", fig_skymap(result, base, per_metric)),
                ("Values vs MJD", fig_values_vs_mjd(result, base, per_metric)),
                ("DAP: parametric vs non-parametric fluxes (bright lines, fibers with NP S/N > min_snr; colour = qa_fit_consistency)",
                 fig_flux_pm_np(result, base)),
                ("DAP: multi-tile flux consistency (qa_fluxmap)", fig_fluxmap(result, base)),
                ("Distributions", fig_distributions(result)),
                ("QA metrics vs observational conditions (fraction of exposures per status of the "
                 "selected observational flag)", fig_metrics_vs_obs(result))]
    elif len(t):
        figs = [("Status counts", fig_counts(result)),
                ("Status per night: overall and summaries", fig_night_status(result, "summary")),
                ("Status per night: individual QA flags", fig_night_status(result, "individual")),
                ("Sky map (one marker per tile: worst status of its exposures)", fig_tiles_sky(result)),
                ("Values per night", fig_values_nightly(result)),
                ("DAP: parametric vs non-parametric fluxes (bright lines, fibers with NP S/N > min_snr)",
                 fig_flux_pm_np_density(result)),
                ("DAP: multi-tile flux consistency (qa_fluxmap)", fig_fluxmap_compact(result)),
                ("Distributions", fig_distributions_compact(result)),
                ("QA metrics vs observational conditions (fraction of exposures per status of the "
                 "selected observational flag)", fig_metrics_vs_obs(result))]
    parts = []
    first = True

    def _div(f):
        nonlocal first
        d = f.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                      config={"displaylogo": False, "responsive": True})
        first = False
        return d

    for title, f in figs:
        if f is None:
            continue
        parts.append(f'<h2>{_html.escape(title)}</h2><div class="card">{_div(f)}</div>')

    # ---- timing: summary only (no plots)
    tparts = []
    if len(t) and "time_total" in t.colnames:
        has_in = _with_inputs(result)
        tt = np.asarray(t["time_total"], dtype=float)[has_in]
        rt = t.meta.get("RUNTIME", np.nan)
        tiles_t = [("pipeline run time", f"{rt:.1f} s" if np.isfinite(rt) else "n/a"),
                   ("worker processes", str(result.nproc)),
                   ("exposures with input files", f"{int(has_in.sum())} / {len(t)}"),
                   ("median per exposure", f"{np.nanmedian(tt):.2f} s" if tt.size else "n/a"),
                   ("mean per exposure", f"{np.nanmean(tt):.2f} s" if tt.size else "n/a"),
                   ("throughput", f"{len(t) / rt * 3600:.0f} exp/h" if rt and np.isfinite(rt) and rt > 0 else "n/a")]
        tparts.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="v">{v}</div><div class="l">{k}</div></div>'
            for k, v in tiles_t) + "</div>")
        ts = timing_stats(result)
        if ts:
            tparts.append(_table_html(ts, list(ts[0].keys())))
        tparts.append("<p class='meta'>Statistics over exposures with input files only. "
                      "Metric times exclude input I/O, which is loaded lazily "
                      "once per exposure and shared by all metrics (reported separately). "
                      "With several worker processes, times include contention for disk/CPU.</p>")

    overall = np.asarray(t["qa_overall"]).astype(str) if len(t) else np.array([])
    tiles = "".join(
        f'<div class="tile"><div class="v">{int(np.sum(overall == s))}</div>'
        f'<div class="l"><span class="dot" style="background:{COLORS[s]}"></span>{ICON[s]} {s}</div></div>'
        for s in STATUS)

    from .output import build_desc_table
    desc = build_desc_table(full)
    ft = fig_tile_status(full) if detail else fig_tile_status_nightly(full)
    status_html = f'<div class="card">{_div(ft)}</div>' if ft is not None else ""
    desc_rows = [{c: str(r[c]) for c in desc.colnames} for r in desc]
    desc_html = _table_html(desc_rows, ["Metric", "Role", "Bits", "Description", "WARNING",
                                        "MAJOR ISSUE", "Parameters"])
    flagged_html = ""
    if detail:
        flagged = [r for r in t if str(r["qa_overall"]) != "GOOD"]
        fcols = ["expnum", "tileid", "mjd", "object", "qa_overall"] + [m.name for m in result.metrics] + ["qa_notes"]
        frows = [{c: (str(r[c]) if str(r[c]) != "--" else "") for c in fcols if c in t.colnames}
                 for r in flagged[:1000]]
        flagged_html = (f"<h2>Exposures not GOOD ({len(flagged)})</h2>"
                        + _table_html(frows, [c for c in fcols if c in t.colnames],
                                      status_cols=set(["qa_overall"] + [m.name for m in result.metrics]))
                        + (f"<p class='meta'>showing first 1000 of {len(flagged)}</p>" if len(flagged) > 1000 else ""))
    worst = worst_tables_html(result, notes) if len(t) else ""
    look = lookup_html(result)

    sel_txt = selection_text(result.selection)
    files = " | ".join(f"<code>{_html.escape(Path(p).name)}</code>" for k, p in result.files.items())
    mode = ("detailed (per-exposure plots)" if detail else
            f"compact (aggregated per night / tile; >= {DETAIL_LIMIT} exposures; --html-detail for per-exposure plots)")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LVM QA report</title><style>{CSS}
details {{ margin:6px 0; }} summary {{ cursor:pointer; font-size:13px; padding:4px 0; }}</style></head>
<body><main>
<h1>LVM QA report &mdash; DRP {cfg.drp_version} / DAP {_html.escape(cfg.dap_label)}</h1>
<div class="meta">lvmqa {__version__} &middot; generated {_dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC &middot;
{len(t)} exposures &middot; selection: {_html.escape(sel_txt)} &middot; layout: {mode}<br>
files: {files}<br>
metrics: {", ".join(f"<code>{m.name}</code>" + (" (summary)" if m.summary else "") for m in result.metrics)}</div>
<h2>Overall status</h2><div class="tiles">{tiles}</div>
<h2>Exposure lookup</h2>{look}
{''.join(parts)}
<h2>Tile status (information)</h2>{status_html}
<h2>Worst exposures per metric (20 each: MAJOR first, then the largest |main value|)</h2>{worst}
{flagged_html}
<h2>Processing time</h2>{''.join(tparts)}
<h2>Metric definitions</h2>{desc_html}
</main></body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
