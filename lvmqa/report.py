"""
HTML report (plotly). Temporary v1 layout:

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


def write_html(result, path) -> Path:
    import copy
    full = result
    result = copy.copy(full)
    result.metrics = [m for m in full.metrics if not m.status]   # flag metrics only
    t = result.table
    cfg = result.config
    figs = []
    if len(t):
        notes = _notes_by_metric(result)
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
        timing_figs = [("Processing time per component", fig_timing_box(result)),
                       ("Processing time per exposure", fig_timing_exposures(result))]
    else:
        timing_figs = []
    parts = []
    first = True
    for title, f in figs:
        if f is None:
            continue
        div = f.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                        config={"displaylogo": False, "responsive": True})
        first = False
        parts.append(f'<h2>{_html.escape(title)}</h2><div class="card">{div}</div>')

    # ---- timing section
    tparts = []
    if len(t) and "time_total" in t.colnames:
        has_in = _with_inputs(result)
        tt = np.asarray(t["time_total"], dtype=float)[has_in]
        rt = t.meta.get("RUNTIME", np.nan)
        tiles_t = [("pipeline run time", f"{rt:.1f} s"), ("worker processes", str(result.nproc)),
                   ("exposures with input files", f"{int(has_in.sum())} / {len(t)}"),
                   ("median per exposure", f"{np.nanmedian(tt):.2f} s" if tt.size else "n/a"),
                   ("mean per exposure", f"{np.nanmean(tt):.2f} s" if tt.size else "n/a"),
                   ("throughput", f"{len(t) / rt * 3600:.0f} exp/h" if rt and rt > 0 else "n/a")]
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
        for title, f in timing_figs:
            if f is None:
                continue
            div = f.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                            config={"displaylogo": False, "responsive": True})
            first = False
            tparts.append(f'<div class="card">{div}</div>')

    overall = np.asarray(t["qa_overall"]).astype(str) if len(t) else np.array([])
    tiles = "".join(
        f'<div class="tile"><div class="v">{int(np.sum(overall == s))}</div>'
        f'<div class="l"><span class="dot" style="background:{COLORS[s]}"></span>{ICON[s]} {s}</div></div>'
        for s in STATUS)

    from .output import build_desc_table
    desc = build_desc_table(full)
    ft = fig_tile_status(full)
    status_html = ""
    if ft is not None:
        status_html = '<div class="card">' + ft.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                                                         config={"displaylogo": False, "responsive": True}) + "</div>"
        first = False
    desc_rows = [{c: str(r[c]) for c in desc.colnames} for r in desc]
    desc_html = _table_html(desc_rows, ["Metric", "Role", "Bits", "Description", "WARNING",
                                        "MAJOR ISSUE", "Parameters"])

    flagged = [r for r in t if str(r["qa_overall"]) != "GOOD"]
    fcols = ["expnum", "tileid", "mjd", "object", "qa_overall"] + [m.name for m in result.metrics] + ["qa_notes"]
    frows = [{c: (str(r[c]) if str(r[c]) != "--" else "") for c in fcols if c in t.colnames}
             for r in flagged[:1000]]
    flagged_html = _table_html(frows, [c for c in fcols if c in t.colnames],
                               status_cols=set(["qa_overall"] + [m.name for m in result.metrics]))
    more = f"<p class='meta'>showing first 1000 of {len(flagged)}</p>" if len(flagged) > 1000 else ""

    sel = result.selection
    sel_txt = ", ".join(f"{k}={v}" for k, v in sel.items() if v) or "all drpall exposures"
    files = " | ".join(f"<code>{_html.escape(Path(p).name)}</code>" for k, p in result.files.items())
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LVM QA report</title><style>{CSS}</style></head>
<body><main>
<h1>LVM QA report &mdash; DRP {cfg.drp_version} / DAP {_html.escape(cfg.dap_label)}</h1>
<div class="meta">lvmqa {__version__} &middot; generated {_dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC &middot;
{len(t)} exposures &middot; selection: {_html.escape(sel_txt)}<br>
files: {files}<br>
metrics: {", ".join(f"<code>{m.name}</code>" + (" (summary)" if m.summary else "") for m in result.metrics)}</div>
<h2>Overall status</h2><div class="tiles">{tiles}</div>
{''.join(parts)}
<h2>Tile status (information)</h2>{status_html}
<h2>Processing time</h2>{''.join(tparts)}
<h2>Metric definitions</h2>{desc_html}
<h2>Exposures not GOOD ({len(flagged)})</h2>{flagged_html}{more}
</main></body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
