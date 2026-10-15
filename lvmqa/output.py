"""
FITS / XLSX writers.

Two groups of files are written, ``<base> = lvmqa-<drpver>-<dapver>[-tag]``:

<base>.fits / <base>.xlsx  -- final QA information (DR20-like)
    QA      : expnum, tileid, mjd, object, qa_bitmask, qa_<metric> (GOOD / WARNING /
              MAJOR ISSUE / UNDEFINED) for every metric
    DESC    : one row per metric: role, slot, bits, description, criteria, parameters
    (xlsx: sheets "QA" with DR20 cell colours, and "DESC")

<base>_supl.fits / <base>_supl.xlsx  -- supplementary measurements
    SUPL    : expnum, tileid, mjd, object, observing info (exptime, obstime, dpos,
              coordinates, airmass, fluxcal), all values measured by the metrics,
              timing, notes
    COLUMNS : description, unit and parent metric of every SUPL column
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from astropy.io import fits
from astropy.table import MaskedColumn, Table

from . import __version__
from .config import PipelineConfig
from .pipeline import ID_COLUMNS, MAIN_ID_COLUMNS

ENCODING = "2 bits per metric at bits 2*slot,2*slot+1: GOOD=00 WARNING=01 MAJOR_ISSUE=10 UNDEFINED=11"
# NB: observational-condition components (qa_illumination, qa_brightstars, qa_guider, ...) are
# not in the bitmask; they enter it only through the summary flag qa_obs (Slot = -1 in DESC).

# DR20 xlsx colours (ARGB)
XLSX_FILLS = {"GOOD": "FF92D050", "WARNING": "FFFFC000", "MAJOR ISSUE": "FFFF0000",
              "UNDEFINED": "FFBFBFBF"}


def output_basename(cfg: PipelineConfig, tag: Optional[str] = None, suffix: str = "") -> str:
    """lvmqa-<drp>[-<dap>][-<tag>][<suffix>], e.g. suffix '_recalculated'."""
    base = f"lvmqa-{cfg.drp_version}" + (f"-{cfg.dap_version}" if cfg.dap_version else "")
    return (f"{base}-{tag}" if tag else base) + (suffix or "")


def to_ascii(value) -> str:
    """FITS string columns must be ASCII."""
    if value is None:
        return ""
    s = (str(value).replace("–", "-").replace("—", "-").replace("−", "-")
         .replace(" ", " ").replace("σ", "sigma").replace("≤", "<=")
         .replace("≥", ">=").replace("Å", "A"))
    return s.encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------------------- tables
def main_table(result) -> Table:
    t = result.table
    cols = [c for c in MAIN_ID_COLUMNS if c in t.colnames] + ["qa_bitmask"]
    cols += [m.name for m in result.metrics if not m.status]
    # tile status (information): boolean columns, e.g. status_auroral
    for m in result.metrics:
        if m.status:
            cols += [m.column(v.name) for v in m.values if v.dtype == "bool"]
    return t[cols]


def supl_columns(result) -> List[str]:
    t = result.table
    cols = [c for c in ID_COLUMNS if c in t.colnames]
    for m in result.metrics:
        cols += [m.column(v.name) for v in m.values]
    cols += [c for c in t.colnames if c.startswith("time_")]
    cols.append("qa_notes")
    return cols


def supl_table(result) -> Table:
    return result.table[supl_columns(result)]


def build_desc_table(result) -> Table:
    rows = []
    for m in result.metrics:
        rows.append(dict(
            Metric=m.name, Role=m.role,
            Slot=m.slot if m.slot is not None else -1,
            Bits=f"{2 * m.slot}-{2 * m.slot + 1}" if m.slot is not None else "- (not in bitmask)",
            Title=m.title, Description=m.description,
            GOOD=m.criteria.get("GOOD", "None of the conditions below"),
            WARNING=m.criteria.get("WARNING", ""),
            **{"MAJOR ISSUE": m.criteria.get("MAJOR ISSUE", "")},
            UNDEFINED=m.criteria.get("UNDEFINED", "Required input missing or metric could not be evaluated"),
            Inputs=",".join(m.requires), Parameters=m.params_string(),
        ))
    t = Table(rows=rows, names=list(rows[0].keys())) if rows else Table()
    for c in t.colnames:
        if t[c].dtype.kind in "SUO":
            t[c] = [to_ascii(v) for v in t[c]]
    return t


_ID_DESC = {
    "expnum": "Exposure number", "tileid": "Tile id", "mjd": "MJD of the observation",
    "object": "Target name", "exptime": "Exposure time", "obstime": "Start of the observation (UTC)",
    "dpos": "Dither position", "sci_ra": "Science IFU center RA", "sci_dec": "Science IFU center Dec",
    "sci_pa": "Science IFU position angle", "sci_amass": "Science airmass",
    "fluxcal": "Flux calibration method",
}


def build_columns_table(result, columns: List[str]) -> Table:
    tab = result.table
    owner: Dict[str, str] = {}
    for m in result.metrics:
        for v in m.values:
            owner[m.column(v.name)] = m.name
        owner[f"time_{m.short}"] = m.name
    rows = []
    for c in columns:
        d = tab[c].description or _ID_DESC.get(c, "")
        if c in _ID_DESC:
            d = f"{d} (drpall)"
        rows.append((c, owner.get(c, ""), str(tab[c].unit or ""), d))
    t = Table(rows=rows, names=["Column", "Metric", "Unit", "Description"])
    for c in t.colnames:
        t[c] = [to_ascii(v) for v in t[c]]
    return t


# --------------------------------------------------------------------------- FITS
def _fits_safe(table: Table) -> Table:
    t = table.copy(copy_data=True)
    t.meta.clear()
    for c in t.colnames:
        if t[c].dtype.kind in "UO":
            t[c] = [to_ascii(v) for v in t[c]]
            if t[c].dtype.itemsize == 0:
                t[c] = t[c].astype("U1")
    return t


def _primary(result, kind: str) -> fits.PrimaryHDU:
    cfg = result.config
    prim = fits.PrimaryHDU()
    h = prim.header
    h["FILETYPE"] = (kind, "QA = final flags, SUPL = supplementary values")
    h["DRPVER"] = (cfg.drp_version, "DRP version analysed")
    h["DAPVER"] = (cfg.dap_version or "none", "DAP version analysed (none = legacy layout)")
    h["QAVER"] = (__version__, "lvmqa version")
    h["DATE"] = (_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), "file creation (UTC)")
    h["NEXP"] = (len(result.table), "number of exposures")
    h["NMETRIC"] = (len(result.metrics), "number of QA metrics")
    h["QA_ENC"] = ENCODING[:68]
    for m in result.metrics:
        if m.slot is not None:
            h[f"QASLOT{m.slot:02d}"] = (m.name, f"bits {2 * m.slot}-{2 * m.slot + 1}")
    sel = result.selection
    for key, name in (("expnums", "SELEXP"), ("mjds", "SELMJD"), ("tileids", "SELTILE")):
        h[name] = (to_ascii(",".join(map(str, sel.get(key, []))) or "all")[:68], f"{key} selection")
    h["RUNTIME"] = (result.table.meta.get("RUNTIME", -1), "[s] pipeline run time")
    return prim


def _bintable(table: Table, name: str) -> fits.BinTableHDU:
    hdu = fits.BinTableHDU(_fits_safe(table), name=name)
    for i, c in enumerate(table.colnames, start=1):
        d = table[c].description
        if d:
            hdu.header[f"TCOMM{i}"] = to_ascii(d)[:68]
    return hdu


def _writeto(hdul: fits.HDUList, path: Path) -> None:
    """Write (overwrite) a FITS file by truncating it in place: works also in folders where
    files may be created/modified but not deleted (astropy's overwrite=True unlinks first)."""
    with open(path, "wb") as fh:
        hdul.writeto(fh)


def write_fits_main(result, path: Path) -> Path:
    hdul = fits.HDUList([_primary(result, "QA"), _bintable(main_table(result), "QA"),
                         _bintable(build_desc_table(result), "DESC")])
    _writeto(hdul, path)
    return Path(path)


def write_fits_supl(result, path: Path) -> Path:
    cols = supl_columns(result)
    hdul = fits.HDUList([_primary(result, "SUPL"), _bintable(supl_table(result), "SUPL"),
                         _bintable(build_columns_table(result, cols), "COLUMNS")])
    _writeto(hdul, path)
    return Path(path)


# --------------------------------------------------------------------------- XLSX
def _cell_value(v):
    if isinstance(v, np.generic):
        v = v.item()
    if v is np.ma.masked:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, str) and v == "--":
        return None
    return v


def _column_values(col) -> list:
    """Python values of a column (masked -> None), converted once per column."""
    if isinstance(col, MaskedColumn) or np.ma.isMaskedArray(col):
        data = np.ma.asarray(col)
        vals = data.data.tolist()
        mask = np.ma.getmaskarray(data)
        if mask.any():
            for k in np.where(mask)[0]:
                vals[k] = None
        return vals
    return np.asarray(col).tolist()


def _write_sheet(ws, table: Table, status_cols=(), units: bool = False,
                 max_width: int = 48, number_format: Optional[str] = "0.0000") -> None:
    """Write a table to a WRITE-ONLY worksheet (rows streamed, no cell objects kept in memory).
    Styled cells (header, units, status colours, float format) are created only where needed;
    number_format=None writes plain numbers (large sheets)."""
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fills = {k: PatternFill(fill_type="solid", start_color=v, end_color=v)
             for k, v in XLSX_FILLS.items()}
    bold = Font(bold=True)
    white_bold = Font(color="FFFFFFFF", bold=True)
    italic = Font(italic=True, color="FF52514E")
    wrap = Alignment(vertical="top", wrap_text=True)
    status_cols = set(status_cols)
    names = list(table.colnames)
    # column widths and frozen panes must be set before the first row of a write-only sheet
    for j, n in enumerate(names, start=1):
        if table[n].dtype.kind in "UO" and len(table):
            w = max(len(n), max(len(str(x)) for x in table[n]))
        else:
            w = len(n) + 2
        ws.column_dimensions[get_column_letter(j)].width = min(max_width, max(10, w + 2))
    r0 = 3 if units else 2
    ws.freeze_panes = f"B{r0}"

    def styled(v, font=None, fill=None, align=None, fmt=None):
        c = WriteOnlyCell(ws, value=v)
        if font is not None:
            c.font = font
        if fill is not None:
            c.fill = fill
        if align is not None:
            c.alignment = align
        if fmt is not None:
            c.number_format = fmt
        return c

    ws.append([styled(n, font=bold, align=wrap) for n in names])
    if units:
        ws.append([styled(str(table[n].unit or ""), font=italic) for n in names])
    cols = [[_cell_value(v) for v in _column_values(table[n])] for n in names]
    kinds = [("status" if n in status_cols else ("float" if table[n].dtype.kind == "f" and number_format else ""))
             for n in names]
    special = [j for j, k in enumerate(kinds) if k]
    for i in range(len(table)):
        row = [c[i] for c in cols]
        for j in special:
            v = row[j]
            if v is None:
                continue
            if kinds[j] == "float":
                row[j] = styled(v, fmt=number_format)
            elif isinstance(v, str) and v in fills:
                row[j] = styled(v, fill=fills[v], font=white_bold if v == "MAJOR ISSUE" else None)
        ws.append(row)


def _write_xlsx(path: Path, sheets) -> Path:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("openpyxl is required for the XLSX output (pip install openpyxl)") from exc
    wb = Workbook(write_only=True)
    for title, table, kw in sheets:
        ws = wb.create_sheet(title=title)
        _write_sheet(ws, table, **kw)
    wb.save(path)
    return Path(path)


def write_xlsx_main(result, path: Path) -> Path:
    desc = build_desc_table(result)
    return _write_xlsx(path, [
        ("QA", main_table(result), dict(status_cols=[m.name for m in result.metrics if not m.status])),
        ("DESC", desc, dict(max_width=60)),
    ])


def write_xlsx_supl(result, path: Path) -> Path:
    cols = supl_columns(result)
    return _write_xlsx(path, [
        ("SUPL", supl_table(result), dict(units=True, number_format=None)),
        ("COLUMNS", build_columns_table(result, cols), dict(max_width=80)),
    ])


def write_outputs(result, tag: Optional[str] = None, suffix: str = "", xlsx: bool = False) -> Dict[str, Path]:
    """FITS always; the XLSX copies (same content, for visual inspection) only if xlsx=True:
    the supplementary sheet has ~800 columns, i.e. millions of cells for large runs."""
    # NB: no Path.with_suffix() - the DAP version contains dots
    base = Path(result.config.output_dir) / output_basename(result.config, tag, suffix)
    base.parent.mkdir(parents=True, exist_ok=True)
    out = {"fits": write_fits_main(result, Path(f"{base}.fits")),
           "fits_supl": write_fits_supl(result, Path(f"{base}_supl.fits"))}
    if xlsx:
        out["xlsx"] = write_xlsx_main(result, Path(f"{base}.xlsx"))
        out["xlsx_supl"] = write_xlsx_supl(result, Path(f"{base}_supl.xlsx"))
    return out


def read_qa_fits(path) -> Dict[str, Table]:
    """Convenience reader: {'QA': Table, 'DESC': Table} or {'SUPL': ..., 'COLUMNS': ...}."""
    with fits.open(path) as hdul:
        names = [h.name for h in hdul[1:]]
    return {n: Table.read(path, hdu=n) for n in names}
