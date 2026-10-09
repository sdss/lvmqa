#!/usr/bin/env python3
"""
Create DR20 QA bitmasks for LVM IFU exposures.

The output is a FITS binary table (and optionally the same data in XLSX) with one row
per exposure from `dr20_pointings_final_v1.2.0.csv` and columns:
  - expnum, mjd, tileid
  - qa_bitmask (string), 2 bits per metric (LSB-first within the used width; trimmed `0x...` hex)
  - human-readable metric columns (GOOD / WARNING / MAJOR ISSUE / UNDEFINED)
  - ancillary numeric columns used to derive metrics (when available)

Metric encoding (2 bits):
  GOOD=00, WARNING=01, MAJOR ISSUE=10, UNDEFINED=11

XLSX export uses openpyxl; status cells are filled green / gold / red / gray.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np

try:
    from astropy.table import Table
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: astropy. Install it (e.g. `pip install astropy`)."
    ) from exc

import pickle


METRIC_DIR_DEFAULT = Path("/Users/mors/Science/LVM/QA/DR20/Metrics_tables/")
FOV_OUTER_DIAMETER_ARCSEC = 1813.0

# Sky emission line subtraction criteria (metric)
SKYLINE_BAD_ABS_ALL_LINES = 0.1
SKYLINE_BAD_ABS_EXCEPT_5577 = 0.30
SKYLINE_MINOR_ABS_EXCEPT_5577 = 0.1

# Knox wavecal / flatfield: fiber-to-spectrograph map and LSF from reference SFrame
SLITMAP_REFERENCE_FITS = Path(
    "/Users/mors/Science/LVM/QA/DR20/test_dap/lvmSFrame-00030930.fits"
)

SKY_LINE_REST_WAVELENGTH_A: Dict[str, float] = {
    "sky5577": 5577.34,
    # "sky6300": 6300.30,
    # "sky6363": 6363.78,
    "sky6533": 6533.01,
    # "sky6553": 6553.56,
    # "sky6577": 6577.31,
    # "sky6912": 6912.59,
    "sky6923": 6923.15,
    "sky6939": 6939.55,
    # "sky7358": 7358.64,
    "sky7392": 7392.18,
    "sky7914": 7913.66,
    "sky8344": 8344.57,
    "sky8399": 8399.23,
    "sky8827": 8827.07,
    "sky8988": 8988.42,
    "sky9552": 9552.52,
    "sky9719": 9719.78,
}

# Mean LSF (Angstrom) at each sky line from SLITMAP_REFERENCE_FITS LSF extension
SKY_LINE_LSF_A: Dict[str, float] = {
    "sky5577": 1.5551,
    # "sky6300": 1.3512,
    # "sky6363": 1.3585,
    "sky6533": 1.3730,
    # "sky6553": 1.3741,
    # "sky6577": 1.3752,
    # "sky6912": 1.3622,
    "sky6923": 1.3610,
    "sky6939": 1.3591,
    # "sky7358": 1.3133,
    "sky7392": 1.3138,
    "sky7914": 1.7637,
    "sky8344": 1.7357,
    "sky8399": 1.7319,
    "sky8827": 1.6968,
    "sky8988": 1.6840,
    "sky9552": 1.7054,
    "sky9719": 1.7598,
}

WAVECAL_CHANNEL_LINES: Dict[str, Tuple[str, ...]] = {
    "b": ("sky5577",),
    "r": (
        # "sky6300",
        # "sky6363",
        "sky6533",
        # "sky6553",
        # "sky6577",
        # "sky6912",
        "sky6923",
        "sky6939",
        # "sky7358",
        "sky7392",
    ),
    "z": (
        "sky7914",
        "sky8344",
        "sky8399",
        "sky8827",
        "sky8988",
        "sky9552",
        "sky9719",
    ),
}

FLATFIELD_SKY_LINES: Tuple[str, ...] = tuple(SKY_LINE_REST_WAVELENGTH_A.keys())
FLATFIELD_ISSUE_FRAC_THRESHOLD = 0.30

# DAP QA bright emission lines (column prefixes in DAP_flux_error_comparison.fits)
DAP_BRIGHT_LINE_PREFIXES: Tuple[str, ...] = (
    "[OII]_3726.03",
    "[OII]_3728.82",
    "[OIII]_5006.84",
    "Hbeta_4861.36",
    "Halpha_6562.85",
    "[NII]_6583.45",
    "[SII]_6716.44",
    "[SII]_6730.82",
)
DAP_SIII_LINE_PREFIXES: Tuple[str, ...] = (
    "[SIII]_9069.0",
    "[SIII]_9531.1",
)

# qa_dap_npfit from DR20_pe_np_comparison_stats.fits (median_perc_diff, percent)
DAP_NPFIT_WARN_ABS_PERC_DIFF = 5.0
DAP_NPFIT_BAD_ABS_PERC_DIFF = 15.0

_FIBER_TO_SPECTROGRAPH: Optional[Dict[int, int]] = None

# Bitmask metric order (first -> last == LSB -> MSB by bit index), must match packing order.
QA_METRIC_COLUMNS_ORDERED: Tuple[str, ...] = (
    "qa_brightstars",
    "qa_sky_brt",
    "qa_absflux",
    "qa_relflux_br",
    "qa_relflux_rz",
    "qa_wavecal_b",
    "qa_wavecal_r",
    "qa_wavecal_z",
    "qa_flatfield",
    "qa_skycont_sub",
    "qa_skyline_sub",
    "qa_dap_npfit",
    "qa_dap_pmfit",
    "qa_visual_drp",
    "qa_visual_dap",
)

class QA2Bit:
    GOOD = 0
    WARNING = 1
    BAD = 2
    UNDEFINED = 3


QA2BIT_TO_STR = {
    QA2Bit.GOOD: "GOOD",
    QA2Bit.WARNING: "WARNING",
    QA2Bit.BAD: "MAJOR ISSUE",
    QA2Bit.UNDEFINED: "UNDEFINED",
}

QA2BIT_FROM_CODE = {
    "00": QA2Bit.GOOD,
    "01": QA2Bit.WARNING,
    "10": QA2Bit.BAD,
    "11": QA2Bit.UNDEFINED,
}


def parse_qa2bit_flag(value: Any) -> int:
    """Parse a 2-bit QA code string (00/01/10/11) to QA2Bit."""
    if value is None:
        return QA2Bit.UNDEFINED
    s = str(value).strip()
    if not s:
        return QA2Bit.UNDEFINED
    return QA2BIT_FROM_CODE.get(s, QA2Bit.UNDEFINED)


def _to_fits_ascii(value: str) -> str:
    """
    FITS string columns are ASCII by default. Normalize common Unicode punctuation
    to ASCII to avoid UnicodeEncodeError on write.
    """
    if value is None:
        return ""
    s = str(value)
    s = (
        s.replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .replace("\u2212", "-")  # minus sign
        .replace("\u00a0", " ")  # non-breaking space
        .replace("\u03c3", "sigma")  # σ
    )
    # Final guard: replace anything still non-ascii
    return s.encode("ascii", "replace").decode("ascii")


def build_desc_table() -> Table:
    """
    Build the DESC extension table describing each metric and its criteria.

    Rows follow `QA_METRIC_COLUMNS_ORDERED`; `Bits` uses integer indices with bit 0 = LSB,
    matching `pack_2bit_metrics()` (LSB-first within the used width).
    """
    metrics = []
    for i, name in enumerate(QA_METRIC_COLUMNS_ORDERED):
        bit_lo = 2 * i
        bit_hi = 2 * i + 1
        bits = f"{bit_lo}-{bit_hi}"

        if name == "qa_brightstars":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Obs. conditions: Bright stars within IFU field",
                    # GOOD="No Hipparcos stars with Vmag < 6 within FoV.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: "Star with 4 <= Vmag < 6 within FoV"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "Stars with Vmag < 4 within FoV"},
                )
            )
        elif name == "qa_sky_brt":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Obs. conditions: Proximity to bright moon or twilight conditions",
                    # GOOD="sun_alt < -18 and (moon below horizon OR sci_moon_sep > 90 OR moon_fli < 0.5).",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: "Moon illumination 0.5 < FLI < 0.8, and separation from moon < 90 degrees (if moon is above horizon)"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "Sun altitude > -18, or moon FLI > 0.8 and separation from moon < 90 degrees (if moon is above horizon)"},
                )
            )
        elif name == "qa_absflux":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Absolute flux calibration: MOD vs SCI agreement across all channels",
                    # GOOD="All 3 channels agree within 1sigma.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: "MOD and SCI sensitivities differ by >1sigma in any channel"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "MOD and SCI sensitivities differ by >1sigma in >1 channel, "
                                                  "OR MOD is absent"},
                )
            )
        elif name == "qa_relflux_br":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Relative flux calibration: continuity at the b–r arm boundary",
                    # GOOD="No jumps exceeding 1sigma in 95% of fibers (2sigma fraction <5% and 1sigma fraction = 0, when available).",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:"Sky+stellar continuum differences between channels >1sigma in >5% "
                                                     "of fibers, but <2sigma in 95% of fibers"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "Sky+stellar continuum differences between channels >2sigma "
                                                  "in >5% of fibers"},
                )
            )
        elif name == "qa_relflux_rz":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Relative flux calibration: continuity at the b–r arm boundary",
                    # GOOD="No jumps exceeding 1sigma in 95% of fibers (2sigma fraction ≤5% and 1sigma fraction = 0, when available).",
                    **{QA2BIT_TO_STR[
                           QA2Bit.WARNING]: "Sky+stellar continuum differences between channels >1sigma in >5% "
                                            "of fibers, but <2sigma in 95% of fibers"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "Sky+stellar continuum differences between channels >2sigma "
                                                  "in >5% of fibers"},
                )
            )
        elif name in {"qa_wavecal_b", "qa_wavecal_r", "qa_wavecal_z"}:
            ch = name.split("_")[-1]
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description=(
                        f"Wavelength calibration in {ch} channel from bright sky-line centroids "
                    ),
                    # GOOD="<=5% of fibers exceed 1/5 LSF offset on any channel sky line.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:
                        ">5% of fibers exceed 1/5 LSF, but <=5% exceed 1 LSF"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: ">5% of fibers exceed 1 LSF on any channel sky line"},
                )
            )
        elif name == "qa_flatfield":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description=(
                        "Flat-field quality: bright sky-line fluxes "
                        "consistency between spectrographs"
                    ),
                    # GOOD="WARNING/BAD line fractions both below 30%.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:
                        f"Median for one spectrograph deviates by >1 sigma for >={int(FLATFIELD_ISSUE_FRAC_THRESHOLD * 100)}% "
                        f"of bright sky lines, "
                        f"but all spectrographs are within 2 sigma"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]:
                        f"Median for one spectrograph deviates by >2 sigma, or median for two spectrographs deviate by >1 sigma for "
                        f">={int(FLATFIELD_ISSUE_FRAC_THRESHOLD * 100)}% of bright sky lines"},
                )
            )
        elif name == "qa_skycont_sub":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Sky continuum subtraction quality",
                    # GOOD="All three arm medians in [-1, 1].",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: "Median abs(continuum) exceed zero level by >1 sigma in any channel, "
                                                      "but within 3sigma in all channels"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "Median abs(continuum) exceed zero level by >3 sigma in any channel"},
                )
            )
        elif name == "qa_skyline_sub":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Sky emission line subtraction quality",
                    # GOOD="No trigger for WARNING/MAJOR ISSUE.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:
                        f"abs(residuals)/(sky line flux) > {SKYLINE_MINOR_ABS_EXCEPT_5577} in any of "
                        f"OH6865, Na5891, O2 lines/bands"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]:
                        (
                            f"abs(residuals)/(sky line flux) > {SKYLINE_BAD_ABS_ALL_LINES} "
                            f"in all of OI5577, OH6865, Na5891, O2 lines/bands, "
                            f"OR > {SKYLINE_BAD_ABS_EXCEPT_5577} in any of "
                        f"OH6865, Na5891, O2 lines/bands"
                        )
                    },
                )
            )
        elif name == "qa_dap_npfit":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description=(
                        "DAP fit quality: parametric vs non-parametric bright-line consistency "
                    ),
                    # GOOD=f"|median_perc_diff| <= {DAP_NPFIT_WARN_ABS_PERC_DIFF}%.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:
                        f"Median difference for bright emission lines > {DAP_NPFIT_WARN_ABS_PERC_DIFF}%"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]:
                        f"Median difference for bright emission lines > {DAP_NPFIT_BAD_ABS_PERC_DIFF}%"},
                )
            )
        elif name == "qa_dap_pmfit":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description=(
                        "DAP fit quality: fit residuals vs "
                        "DAP parametric flux error for bright emission lines"
                    ),
                    # GOOD=">95% of bright lines with mean resid_rss <= 1 sigma.",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]:
                        ">5% of per-fiber measurements exceed 3sigma, but <=5% exceed 5sigma"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: ">5% of per-fiber measurements exceed 5sigma"},
                )
            )
        elif name == "qa_visual_drp":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Visual inspection of SFrames",
                    # GOOD='Visual QC: Spectra == "Good".',
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: 'Subjective assessment by the QA team'},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: 'Subjective assessment by the QA team'},
                )
            )
        elif name == "qa_visual_dap":
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="Visual inspection of DAP fit results",
                    # GOOD='Visual QC: DAP == "Good".',
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: 'Subjective assessment by the QA team'},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: 'Subjective assessment by the QA team'},
                )
            )
        else:
            metrics.append(
                dict(
                    Metric=name,
                    Bits=bits,
                    Description="(no description)",
                    # GOOD="TBD",
                    **{QA2BIT_TO_STR[QA2Bit.WARNING]: "TBD"},
                    **{QA2BIT_TO_STR[QA2Bit.BAD]: "TBD"},
                )
            )

    desc = Table(rows=metrics, names=["Metric", "Bits", "Description",
                                      QA2BIT_TO_STR[QA2Bit.WARNING], QA2BIT_TO_STR[QA2Bit.BAD]])
    # sanitize to ASCII for FITS writing
    for col in desc.colnames:
        desc[col] = [_to_fits_ascii(v) for v in desc[col]]
    return desc


def normalize_expnum(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(value: Any) -> Optional[int]:
    v = safe_float(value)
    return int(v) if v is not None else None

def load_stars_cache(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load Hipparcos bright-star cache created by `scripts/qa_find_bright_stars.py`.
    Returns the `by_expnum` mapping (expnum -> record).
    """
    with path.open("rb") as f:
        payload = pickle.load(f)
    by_expnum = payload.get("by_expnum")
    if not isinstance(by_expnum, dict):
        raise SystemExit(f"Unexpected stars_cache format in {path}")
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in by_expnum.items():
        expnum = normalize_expnum(k)
        if expnum is None or not isinstance(v, dict):
            continue
        out[expnum] = v
    return out

def pack_2bit_metrics(codes: Iterable[int]) -> np.uint64:
    """
    Pack 2-bit metric codes LSB-first within the used width.

    The first code in `codes` occupies bits 0-1, the next bits 2-3, and so on.
    """
    bitmask = np.uint64(0)
    for idx, code in enumerate(codes):
        bitmask |= np.uint64(int(code) & 0b11) << np.uint64(2 * idx)
    return bitmask


def bitmask_to_hex(value: np.uint64) -> str:
    """Format bitmask as hex string with trimmed leading zeros (Python `hex()`)."""
    return hex(int(value))


def parse_pipe_dat_table(path: Path) -> Dict[int, Dict[str, str]]:
    """
    Parses files like `drpqa_metrics_dr20_sky.dat` that look like a markdown table:
      | col1 | col2 |
      |  1.0 |  ... |
    Returns {expnum: {col: raw_str, ...}}
    """
    rows_by_expnum: Dict[int, Dict[str, str]] = {}
    lines = path.read_text().splitlines()
    if not lines:
        return rows_by_expnum

    # Find header
    header = None
    for line in lines:
        if "|" in line and "expnum" in line:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            break
    if not header:
        return rows_by_expnum

    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        if parts[0] == "expnum":
            continue
        expnum = normalize_expnum(parts[0])
        if expnum is None:
            continue
        rows_by_expnum[expnum] = {header[i]: parts[i] for i in range(len(header))}
    return rows_by_expnum


def read_pointings_master(path: Path) -> Dict[int, Dict[str, Any]]:
    master: Dict[int, Dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expnum = normalize_expnum(row.get("expnum"))
            if expnum is None:
                continue
            master[expnum] = {
                "expnum": expnum,
                "mjd": safe_int(row.get("mjd")),
                "tileid": safe_int(row.get("tileid")),
                "sun_alt": safe_float(row.get("sun_alt")),
                "moon_alt": safe_float(row.get("moon_alt")),
                "moon_fli": safe_float(row.get("moon_fli")),
                "sci_moon_sep": safe_float(row.get("sci_moon_sep")),
                "sci_ra": safe_float(row.get("sci_ra")),
                "sci_dec": safe_float(row.get("sci_dec")),
                "sci_pa": safe_float(row.get("sci_pa")),
            }
    return master


def compute_metric_brightstars_from_cache(
    *,
    expnum: int,
    stars_cache: Dict[int, Dict[str, Any]],
    mag_limits: Tuple[float, float] = (6.0, 10.0),
) -> Tuple[int, Dict[str, Any]]:
    """
    Metric (a): bright stars within the LVM IFU hexagonal field of view.

    Uses a precomputed Hipparcos cache (no network access here).

      - MAJOR ISSUE: any star brighter than 6 mag in V within the hex.
      - MINOR ISSUE: any star brighter than 10 mag in V within the hex.
    """
    rec = stars_cache.get(expnum)
    if not rec:
        return QA2Bit.UNDEFINED, {"hip_cache_present": False}

    vmag_list = rec.get("vmag_list") or []
    try:
        mags = np.array([float(m) for m in vmag_list], dtype=float)
    except Exception:
        return QA2Bit.UNDEFINED, {"hip_cache_present": True}

    n_lt_major = int(np.sum(mags < mag_limits[0]))
    n_lt_minor = int(np.sum(mags < mag_limits[1]))
    brightest = float(np.min(mags)) if mags.size else float("nan")

    anc = {
        "hip_cache_present": True,
        "hip_band": rec.get("band", "Vmag"),
        "hip_n_lt6": n_lt_major,
        "hip_n_lt10": n_lt_minor,
        "hip_brightest_mag": brightest,
        "hip_max_mag_in_cache": rec.get("max_mag_in_cache", np.nan),
        "hip_fov_outer_diameter_arcsec": rec.get(
            "fov_outer_diameter_arcsec", FOV_OUTER_DIAMETER_ARCSEC
        ),
    }

    if n_lt_major > 0:
        return QA2Bit.BAD, anc
    if n_lt_minor > 0:
        return QA2Bit.WARNING, anc
    return QA2Bit.GOOD, anc


def _get_sky_resid(sky_row: Optional[Dict[str, str]], key: str) -> Optional[float]:
    if not sky_row:
        return None
    # Headers in the .dat include brackets, e.g. "[OI]5577_resid_sky"
    return safe_float(sky_row.get(key))


def compute_metric_sky_emission_line_subtraction(
    sky_row: Optional[Dict[str, str]],
) -> int:
    """
    Sky emission line subtraction quality.

    Use `_resid_sky` for 4 lines:
      [OI]5577, [OH]6865, [Na]5891, O2

    BAD:
      - abs(resid) > 0.15 in *all* lines, OR
      - abs(resid) > 0.30 in any line except 5577
    MINOR ISSUE:
      - abs(resid) > 0.15 in any line except 5577
    UNDEFINED:
      - any required residual missing
    """
    keys = {
        "oi5577": "[OI]5577_resid_sky",
        "oh6865": "[OH]6865_resid_sky",
        "na5891": "[Na]5891_resid_sky",
        "o2": "O2_resid_sky",
    }
    vals = {name: _get_sky_resid(sky_row, k) for name, k in keys.items()}

    if any(v is None for v in vals.values()):
        return QA2Bit.UNDEFINED

    absvals = {k: abs(v) for k, v in vals.items() if v is not None}

    bad_all = all(v > SKYLINE_BAD_ABS_ALL_LINES for v in absvals.values())
    bad_any_except_5577 = any(
        absvals[k] > SKYLINE_BAD_ABS_EXCEPT_5577 for k in ("oh6865", "na5891", "o2")
    )
    minor_any_except_5577 = any(
        absvals[k] > SKYLINE_MINOR_ABS_EXCEPT_5577 for k in ("oh6865", "na5891", "o2")
    )

    if bad_all or bad_any_except_5577:
        return QA2Bit.BAD
    if minor_any_except_5577:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def get_fiber_to_spectrograph() -> Dict[int, int]:
    """fiberid -> spectrographid (1, 2, 3) from reference SLITMAP."""
    global _FIBER_TO_SPECTROGRAPH
    if _FIBER_TO_SPECTROGRAPH is not None:
        return _FIBER_TO_SPECTROGRAPH
    if not SLITMAP_REFERENCE_FITS.exists():
        _FIBER_TO_SPECTROGRAPH = {}
        return _FIBER_TO_SPECTROGRAPH
    from astropy.io import fits

    with fits.open(SLITMAP_REFERENCE_FITS, memmap=True) as hdul:
        slitmap = hdul["SLITMAP"].data
        col = "spectrographid" if "spectrographid" in slitmap.dtype.names else "spectrographID"
        _FIBER_TO_SPECTROGRAPH = {
            int(row["fiberid"]): int(row[col]) for row in slitmap
        }
    return _FIBER_TO_SPECTROGRAPH


def knox_cframe_path(knox_dir: Path, expnum: int) -> Path:
    return knox_dir / f"lvmCFrame-{expnum:08d}.txt"


def read_knox_cframe_table(path: Path) -> Optional[Table]:
    """Load per-fiber Knox Gaussian-fit table for one exposure."""
    if not path.exists():
        return None
    from astropy.io import ascii

    return ascii.read(path, format="fixed_width_two_line")


def compute_metric_wavelength_calibration_accuracy(
    knox_table: Optional[Table], channel: str
) -> int:
    """
    Wavelength calibration QA for one spectroscopic channel (b/r/z).

    Per fiber, flag if any channel sky line has |wave - rest| > LSF/5 or > LSF.
    GOOD: <=5% fibers exceed LSF/5; WARNING: >5% exceed LSF/5 but <=5% exceed LSF;
    BAD: >5% exceed LSF.
    """
    if knox_table is None:
        return QA2Bit.UNDEFINED
    lines = WAVECAL_CHANNEL_LINES.get(channel)
    if not lines:
        return QA2Bit.UNDEFINED

    n_fibers = len(knox_table)
    if n_fibers == 0:
        return QA2Bit.UNDEFINED

    exceeds_lsf5 = np.zeros(n_fibers, dtype=bool)
    exceeds_lsf1 = np.zeros(n_fibers, dtype=bool)
    has_valid = np.zeros(n_fibers, dtype=bool)

    for line in lines:
        wave_col = f"wave_{line}"
        if wave_col not in knox_table.colnames:
            return QA2Bit.UNDEFINED
        wave = np.asarray(knox_table[wave_col], dtype=float)
        rest = SKY_LINE_REST_WAVELENGTH_A[line]
        lsf = SKY_LINE_LSF_A[line]
        finite = np.isfinite(wave)
        has_valid |= finite
        delta = np.abs(wave - rest)
        exceeds_lsf5 |= finite & (delta > lsf / 5.0)
        exceeds_lsf1 |= finite & (delta > lsf)

    n_valid = int(np.sum(has_valid))
    if n_valid == 0:
        return QA2Bit.UNDEFINED

    frac_lsf5 = float(np.sum(exceeds_lsf5 & has_valid)) / n_valid
    frac_lsf1 = float(np.sum(exceeds_lsf1 & has_valid)) / n_valid

    if frac_lsf1 > 0.05:
        return QA2Bit.BAD
    if frac_lsf5 > 0.05:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def classify_flatfield_line(
    med_flux: Dict[int, float], std_flux: Dict[int, float]
) -> Optional[int]:
    """
    Classify one sky line from spectrograph median flux and per-spec std.

    sigma = median(std per spectrograph); offsets = |med_s - median(meds)| / sigma.
    """
    if not all(sid in med_flux and sid in std_flux for sid in (1, 2, 3)):
        return None
    sigma = float(np.median([std_flux[1], std_flux[2], std_flux[3]]))
    if sigma <= 0 or not np.isfinite(sigma):
        return None
    meds = [med_flux[1], med_flux[2], med_flux[3]]
    ref = float(np.median(meds))
    offsets = [abs(m - ref) / sigma for m in meds]
    n_over_1 = sum(o > 1.0 for o in offsets)
    n_over_2 = sum(o > 2.0 for o in offsets)
    if n_over_2 >= 1 or n_over_1 >= 2:
        return QA2Bit.BAD
    if n_over_1 == 1:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def flatfield_line_classes(
    knox_table: Table, spec_ids: np.ndarray
) -> list[int]:
    """Classify each sky line for flat-field consistency."""
    line_classes: list[int] = []
    for line in FLATFIELD_SKY_LINES:
        flux_col = f"flux_{line}"
        if flux_col not in knox_table.colnames:
            continue
        flux = np.asarray(knox_table[flux_col], dtype=float)
        med_flux: Dict[int, float] = {}
        std_flux: Dict[int, float] = {}
        for sid in (1, 2, 3):
            sel = spec_ids == sid
            if not np.any(sel):
                continue
            fvals = flux[sel]
            ffinite = fvals[np.isfinite(fvals)]
            if len(ffinite) < 2:
                continue
            std_val = float(np.nanstd(ffinite))
            if std_val <= 0 or not np.isfinite(std_val):
                continue
            med_flux[sid] = float(np.nanmedian(ffinite))
            std_flux[sid] = std_val
        line_class = classify_flatfield_line(med_flux, std_flux)
        if line_class is not None:
            line_classes.append(line_class)
    return line_classes


def compute_metric_flat_fielding_accuracy(knox_table: Optional[Table]) -> int:
    """
    Flat-field QA from all Knox sky-line fluxes (no per-channel split).

    Per line: same 1-sigma / 2-sigma spectrograph consistency rules.
    Exposure: return worst severity whose line fraction is >= 30%
    (BAD checked before WARNING).
    """
    if knox_table is None:
        return QA2Bit.UNDEFINED
    if "fiberid" not in knox_table.colnames:
        return QA2Bit.UNDEFINED

    fiber_map = get_fiber_to_spectrograph()
    if not fiber_map:
        return QA2Bit.UNDEFINED

    fiberids = np.asarray(knox_table["fiberid"], dtype=int)
    spec_ids = np.array([fiber_map.get(int(fid), -1) for fid in fiberids], dtype=int)

    line_classes = flatfield_line_classes(knox_table, spec_ids)
    if not line_classes:
        return QA2Bit.UNDEFINED

    n_lines = len(line_classes)
    frac_bad = sum(c == QA2Bit.BAD for c in line_classes) / n_lines
    frac_warn = sum(c == QA2Bit.WARNING for c in line_classes) / n_lines

    if frac_bad >= FLATFIELD_ISSUE_FRAC_THRESHOLD:
        return QA2Bit.BAD
    if frac_warn >= FLATFIELD_ISSUE_FRAC_THRESHOLD:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def dap_bright_line_prefixes(*, include_siii: bool = False) -> Tuple[str, ...]:
    if include_siii:
        return DAP_BRIGHT_LINE_PREFIXES + DAP_SIII_LINE_PREFIXES
    return DAP_BRIGHT_LINE_PREFIXES


def _table_row_float(row: Any, col: str) -> Optional[float]:
    if col not in row.colnames:
        return None
    return safe_float(row[col])


def classify_from_sigma_fractions(
    frac_exceed_1sigma: float, frac_exceed_3sigma: float, n_samples: int
) -> int:
    """Tiered QA from fractions exceeding 1-sigma and 3-sigma thresholds."""
    if n_samples <= 0:
        return QA2Bit.UNDEFINED
    if frac_exceed_3sigma > 0.05:
        return QA2Bit.BAD
    if frac_exceed_1sigma > 0.05:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def read_pe_np_comparison_stats(path: Path) -> Dict[int, float]:
    """
    Load median parametric vs non-parametric flux difference per exposure.

    Returns {expnum: median_perc_diff} from DR20_pe_np_comparison_stats.fits.
    """
    if not path.exists():
        return {}
    table = Table.read(path)
    if "expnum" not in table.colnames or "median_perc_diff" not in table.colnames:
        return {}
    out: Dict[int, float] = {}
    for row in table:
        expnum = normalize_expnum(row["expnum"])
        if expnum is None:
            continue
        val = safe_float(row["median_perc_diff"])
        if val is not None:
            out[expnum] = val
    return out


def compute_metric_dap_npfit_from_median_perc_diff(
    median_perc_diff: Optional[float],
) -> int:
    """
    NP vs parametric consistency from exposure-level median percent flux difference.

    WARNING: |median_perc_diff| > 5%; BAD: |median_perc_diff| > 15%.
    """
    if median_perc_diff is None:
        return QA2Bit.UNDEFINED
    abs_diff = abs(median_perc_diff)
    if abs_diff > DAP_NPFIT_BAD_ABS_PERC_DIFF:
        return QA2Bit.BAD
    if abs_diff > DAP_NPFIT_WARN_ABS_PERC_DIFF:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


def read_dap_flags_fits(path: Path) -> Dict[int, Dict[str, str]]:
    """
    Load per-exposure DAP QA flags from DAP_flux_comparison_flags.fits.

    Returns {expnum: {"PM_NP_flag": "01", "resid_flag": "10", ...}}.
    """
    table = Table.read(path)
    out: Dict[int, Dict[str, str]] = {}
    for row in table:
        expnum = normalize_expnum(row["expnum"])
        if expnum is None:
            continue
        rec: Dict[str, str] = {}
        for col in ("PM_NP_flag", "resid_flag"):
            if col in table.colnames:
                rec[col] = str(row[col]).strip()
        out[expnum] = rec
    return out


def read_dap_error_comparison_by_expnum(
    error_path: Path, flags_path: Path
) -> Dict[int, Table]:
    """
    Load DAP_flux_error_comparison.fits rows keyed by expnum.

    Row order matches DAP_flux_comparison_flags.fits (same index -> same exposure).
    """
    if not error_path.exists() or not flags_path.exists():
        return {}
    error_table = Table.read(error_path)
    flags_table = Table.read(flags_path)
    if len(error_table) != len(flags_table):
        raise SystemExit(
            f"DAP error comparison ({len(error_table)} rows) and flags "
            f"({len(flags_table)} rows) must have the same length."
        )
    out: Dict[int, Table] = {}
    for i in range(len(flags_table)):
        expnum = normalize_expnum(flags_table["expnum"][i])
        if expnum is None:
            continue
        out[expnum] = error_table[i]
    return out


def compute_metric_dap_pmfit_from_error_row(
    error_row: Optional[Table], *, include_siii: bool = False
) -> int:
    """
    Parametric-fit residual QA from exposure-level bright-line stats.

    Uses mean resid_rss and mean flux_err per line from DAP_flux_error_comparison.fits.
    resid_sigma = resid_rss / (flux_err * 1e-16), compared to 1-sigma and 3-sigma bands.
    """
    if error_row is None:
        return QA2Bit.UNDEFINED

    exceed_1 = 0
    exceed_3 = 0
    n_valid = 0
    for prefix in dap_bright_line_prefixes(include_siii=include_siii):
        flux_err = _table_row_float(error_row, f"{prefix}_flux_err")
        resid = _table_row_float(error_row, f"{prefix}_resid_rss")
        if resid is None:
            resid = _table_row_float(error_row, f"{prefix}_resid_rms")
        if flux_err is None or resid is None or flux_err <= 0 or resid < 0:
            continue
        sigma = flux_err * 1e-16
        if sigma <= 0:
            continue
        resid_sigma = resid / sigma
        n_valid += 1
        if resid_sigma > 5.0:
            exceed_3 += 1
        if resid_sigma > 3.0:
            exceed_1 += 1

    if n_valid == 0:
        return QA2Bit.UNDEFINED
    return classify_from_sigma_fractions(
        exceed_1 / n_valid, exceed_3 / n_valid, n_valid
    )


def compute_metric_a_sky_brightness(
    *, moon_fli: Optional[float], sun_alt: Optional[float], moon_alt: Optional[float], sci_moon_sep: Optional[float]
) -> int:
    if sun_alt is None:
        return QA2Bit.UNDEFINED
    twilight = sun_alt > -18.0
    if twilight:
        return QA2Bit.BAD
    if (moon_alt is None) or (moon_fli is None) or (sci_moon_sep is None):
        return QA2Bit.UNDEFINED
    if moon_alt <= 0:
        return QA2Bit.GOOD
    if sci_moon_sep >= 90:
        return QA2Bit.GOOD
    if moon_fli < 0.5:
        return QA2Bit.GOOD
    if (moon_fli < 0.8) and (sci_moon_sep < 90):
        return QA2Bit.WARNING
    return QA2Bit.BAD


def read_continuum_json_tar(path: Path) -> Dict[int, Dict[str, float]]:
    """
    Reads `dr20_json.tar.gz` and extracts per-expnum continuum metrics for BR and RZ boundaries.
    Returns mapping:
      expnum -> {
        'br_flagged_frac', 'br_max_arm_diff', 'rz_flagged_frac', 'rz_max_arm_diff'
      }
    """
    out: Dict[int, Dict[str, float]] = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.name.endswith("_qa.json"):
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = json.load(f)
            except Exception:
                continue

            expnum = normalize_expnum(data.get("expnum"))
            if expnum is None:
                # Fallback: .../lvmSFrame-00005754_qa.json
                stem = Path(m.name).stem  # lvmSFrame-00005754_qa
                expnum = normalize_expnum(stem)
            if expnum is None:
                continue

            er = data.get("extended_report", {}) or {}
            br = er.get("continuum_check_blue_red", {}) or {}
            rz = er.get("continuum_check_red_ir", {}) or {}
            rec: Dict[str, float] = {}
            for prefix, block in (("br", br), ("rz", rz)):
                flagged_frac_bad = block.get("flagged_frac_bad")
                flagged_frac_warn = block.get("flagged_frac_warn")
                if isinstance(flagged_frac_bad, (int, float)):
                    rec[f"{prefix}_flagged_frac_bad"] = float(flagged_frac_bad)
                if isinstance(flagged_frac_warn, (int, float)):
                    rec[f"{prefix}_flagged_frac_warn"] = float(flagged_frac_warn)
            if rec:
                out[expnum] = rec
    return out


def compute_metric_jump_from_flagged_frac(
    flagged_frac: Optional[float], flagged_frac_2sig: Optional[float]
) -> int:
    """
    Continuum jump metrics (b–r and r–z) from QA JSON `flagged_frac` (at 1σ and 2σ thresholds).

    Rules:
      BAD:
        - flagged_frac_2sig > 0.05  (>5% of fibers exceed 1σ)
      WARNING:
        - flagged_frac > 0.05 AND flagged_frac_2sig < 0.05
          (some fibers exceed 1σ, but the 2σ population criterion is still OK)
      GOOD:
        - otherwise (including the case with no 2σ table where 0 < flagged_frac <= 0.05)
    """
    if flagged_frac is None or flagged_frac_2sig is None:
        return QA2Bit.UNDEFINED
    if flagged_frac_2sig > 0.05:
        return QA2Bit.BAD
    if flagged_frac > 0.05:
        return QA2Bit.WARNING
    return QA2Bit.GOOD


@dataclass(frozen=True)
class PhotometryArm:
    name: str  # "b"|"r"|"z"
    mod_med_key: str
    mod_rms_key: str
    sci_med_key: str
    sci_rms_key: str


PHOT_ARMS = [
    PhotometryArm("b", "MODSENMB", "MODSENRB", "SCISENMB", "SCISENRB"),
    PhotometryArm("r", "MODSENMR", "MODSENRR", "SCISENMR", "SCISENRR"),
    PhotometryArm("z", "MODSENMZ", "MODSENRZ", "SCISENMZ", "SCISENRZ"),
]


def read_photometry_csv(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expnum = normalize_expnum(row.get("expnum"))
            if expnum is None:
                continue
            out[expnum] = row
    return out


def compute_metric_d_fluxcal_from_phot_row(row: Optional[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """
    Uses dr20_photometry.csv where:
      - MODSENM.. and SCISENM.. are medians
      - MODSENR.. and SCISENR.. are RMS (= uncertainty)
    """
    if not row:
        return QA2Bit.UNDEFINED, {}

    # MOD absent if any required MOD median is missing
    mod_meds = [safe_float(row.get(a.mod_med_key)) for a in PHOT_ARMS]
    if any(v is None for v in mod_meds):
        return QA2Bit.BAD, {
            "fluxcal_mod_absent": True,
        }

    ancillary: Dict[str, Any] = {"fluxcal_mod_absent": False}
    outlier_count = 0
    zscores: Dict[str, Optional[float]] = {}

    for arm in PHOT_ARMS:
        mod_med = safe_float(row.get(arm.mod_med_key))
        sci_med = safe_float(row.get(arm.sci_med_key))
        mod_rms = safe_float(row.get(arm.mod_rms_key))
        sci_rms = safe_float(row.get(arm.sci_rms_key))

        ancillary[f"fluxcal_{arm.name}_mod_med"] = mod_med
        ancillary[f"fluxcal_{arm.name}_sci_med"] = sci_med
        ancillary[f"fluxcal_{arm.name}_mod_rms"] = mod_rms
        ancillary[f"fluxcal_{arm.name}_sci_rms"] = sci_rms

        if None in (mod_med, sci_med, mod_rms, sci_rms) or (mod_rms == 0 and sci_rms == 0):
            z = None
        else:
            denom = math.sqrt((mod_rms or 0.0) ** 2 + (sci_rms or 0.0) ** 2)
            z = abs((mod_med or 0.0) - (sci_med or 0.0)) / denom if denom > 0 else None
        zscores[arm.name] = z
        ancillary[f"fluxcal_{arm.name}_z"] = z
        if z is not None and z > 1.0:
            outlier_count += 1

    if any(ancillary.get(f"fluxcal_{a.name}_z") is None for a in PHOT_ARMS):
        return QA2Bit.UNDEFINED, ancillary

    if outlier_count == 0:
        return QA2Bit.GOOD, ancillary
    if outlier_count == 1:
        return QA2Bit.WARNING, ancillary
    return QA2Bit.BAD, ancillary

def get_source_type(expnum, tileid):
    """
    Return type of the source (MW, MCs or Extragalactic) based on expnum or tileid
    Args:
        expnum:
        tileid:

    Returns:
        source type: 0 for MW, 1 for MCs and 2 for Extragalactic
    """
    if expnum in {5511, 5512, 12183}:
        return 2  # Extragalactic
    return 0 # MW by default


def read_jump2sig_csv(path: Path) -> Dict[int, Dict[str, float]]:
    """
    Optional future companion table for continuum jump metrics.

    Expected columns (header names are case-insensitive, extra columns ignored):
      - expnum
      - br_flagged_frac_2sig  (fraction of fibers exceeding 2σ threshold at the boundary)
      - rz_flagged_frac_2sig
    """
    out: Dict[int, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        fields = {name.lower(): name for name in reader.fieldnames}

        def col(*candidates: str) -> Optional[str]:
            for cand in candidates:
                if cand.lower() in fields:
                    return fields[cand.lower()]
            return None

        c_exp = col("expnum")
        c_br = col("br_flagged_frac_2sig", "br_flagged_frac2", "br_frac_2sig")
        c_rz = col("rz_flagged_frac_2sig", "rz_flagged_frac2", "rz_frac_2sig")
        if c_exp is None:
            raise SystemExit(
                "jump 2σ CSV must include an `expnum` column (see script docstring for expected layout)."
            )

        for row in reader:
            expnum = normalize_expnum(row.get(c_exp))
            if expnum is None:
                continue
            rec: Dict[str, float] = {}
            if c_br:
                v = safe_float(row.get(c_br))
                if v is not None:
                    rec["br_flagged_frac_2sig"] = float(v)
            if c_rz:
                v = safe_float(row.get(c_rz))
                if v is not None:
                    rec["rz_flagged_frac_2sig"] = float(v)
            if rec:
                out[expnum] = rec
    return out


def read_visual_qc(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expnum = normalize_expnum(row.get("expnum"))
            if expnum is None:
                continue
            out[expnum] = {
                "spectra": (row.get("Visual QC: Spectra") or "").strip(),
                "dap": (row.get("Visual QC: DAP") or "").strip(),
            }
    return out


def parse_visual_flag(value: str) -> int:
    s = (value or "").strip().lower()
    if not s:
        # I know these are good
        return QA2Bit.GOOD
    if s == "good":
        return QA2Bit.GOOD
    if s in {"minor issue", "minor issues"}:
        return QA2Bit.WARNING
    if s == "bad":
        return QA2Bit.BAD
    return QA2Bit.GOOD


def read_cont_snr_parquet(path: Path) -> Dict[int, Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyarrow (needed to read cont_snr_dr20_.parquet). "
            "Install it in your environment (e.g. `pip install pyarrow`)."
        ) from exc

    table = pq.read_table(path)
    out: Dict[int, Dict[str, Any]] = {}
    cols = set(table.column_names)
    required = {"expnum", "b_snr_med", "r_snr_med", "z_snr_med"}
    if not required.issubset(cols):
        raise SystemExit(f"Unexpected parquet schema. Missing {sorted(required - cols)}")
    data = table.to_pydict()
    n = len(data["expnum"])
    for i in range(n):
        expnum = normalize_expnum(data["expnum"][i])
        if expnum is None:
            continue
        out[expnum] = {
            "b_snr_med": data["b_snr_med"][i],
            "r_snr_med": data["r_snr_med"][i],
            "z_snr_med": data["z_snr_med"][i],
        }
    return out


def compute_metric_e_contsub(snr_row: Optional[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    if not snr_row:
        return QA2Bit.UNDEFINED, {}
    b = safe_float(snr_row.get("b_snr_med"))
    r = safe_float(snr_row.get("r_snr_med"))
    z = safe_float(snr_row.get("z_snr_med"))
    ancillary = {"cont_b_snr_med": b, "cont_r_snr_med": r, "cont_z_snr_med": z}
    if None in (b, r, z):
        return QA2Bit.UNDEFINED, ancillary
    in_1 = all(-1.0 <= v <= 1.0 for v in (b, r, z))
    if in_1:
        return QA2Bit.GOOD, ancillary
    in_2 = all(-3.0 <= v <= 3.0 for v in (b, r, z))
    if in_2:
        return QA2Bit.WARNING, ancillary
    return QA2Bit.BAD, ancillary


def sky_twilight_label(sun_alt: Optional[float]) -> str:
    if sun_alt is None:
        return "unknown"
    return "twilight" if sun_alt > -18.0 else "night"


def moon_brightness_zone(moon_fli: Optional[float]) -> str:
    if moon_fli is None:
        return "unknown"
    if moon_fli < 0.5:
        return "low"
    if moon_fli < 0.8:
        return "medium"
    return "high"


def write_qa_summary_xlsx(
    table: Table,
    path: Path,
    *,
    qa_column_names: Tuple[str, ...] = QA_METRIC_COLUMNS_ORDERED,
) -> None:
    """
    Write the summary table to XLSX. QA status columns get background colors:
      GOOD -> green, MINOR ISSUE -> gold, MAJOR ISSUE -> red, UNDEFINED -> gray.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: openpyxl (needed for XLSX export). "
            "Install it (e.g. `pip install openpyxl`)."
        ) from exc

    # ARGB (Excel / openpyxl): FF + RRGGBB
    fills = {
        "GOOD": PatternFill(fill_type="solid", start_color="FF92D050", end_color="FF92D050"),
        "WARNING": PatternFill(
            fill_type="solid", start_color="FFFFC000", end_color="FFFFC000"
        ),
        "MAJOR ISSUE": PatternFill(
            fill_type="solid", start_color="FFFF0000", end_color="FFFF0000"
        ),
        "UNDEFINED": PatternFill(
            fill_type="solid", start_color="FFBFBFBF", end_color="FFBFBFBF"
        ),
    }
    header_font = Font(bold=True)
    wrap = Alignment(vertical="top", wrap_text=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "DR20 QA"

    colnames = list(table.colnames)
    qa_set = set(qa_column_names)

    for c, name in enumerate(colnames, start=1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.font = header_font
        cell.alignment = wrap

    nrows = len(table)
    for r in range(nrows):
        excel_row = r + 2
        for c, name in enumerate(colnames, start=1):
            val = table[name][r]
            if isinstance(val, np.generic):
                val = val.item()
            cell = ws.cell(row=excel_row, column=c, value=val)
            cell.alignment = wrap
            if name in qa_set and isinstance(val, str):
                v = val.strip()
                if v in fills:
                    cell.fill = fills[v]
                    if v == "MAJOR ISSUE":
                        cell.font = Font(color="FFFFFFFF", bold=True)

    for c, name in enumerate(colnames, start=1):
        letter = get_column_letter(c)
        ws.column_dimensions[letter].width = min(48, max(10, len(name) + 2))

    ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DR20 QA bitmask FITS table.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=METRIC_DIR_DEFAULT,
        help="Directory containing the DR20 metrics tables and where output will be written.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output FITS path. Default: <metrics-dir>/dr20_qa_summary_bitmasks.fits",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=None,
        help=(
            "Output XLSX path. Default: same directory and basename as FITS with .xlsx "
            "(e.g. dr20_qa_summary_bitmasks.xlsx). Pass a path to override."
        ),
    )
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Do not write the Excel file.",
    )
    parser.add_argument(
        "--dap-flags",
        type=Path,
        default=None,
        help=(
            "DAP flux comparison flags FITS (PM_NP_flag, resid_flag per expnum). "
            "Default: <metrics-dir>/DAP_flux_comparison_flags.fits"
        ),
    )
    parser.add_argument(
        "--dap-error-comparison",
        type=Path,
        default=None,
        help=(
            "DAP flux/error/residual measurements per exposure. "
            "Default: <metrics-dir>/DAP_flux_error_comparison.fits"
        ),
    )
    parser.add_argument(
        "--dap-use-flags",
        action="store_true",
        help=(
            "Use pre-computed PM_NP_flag and resid_flag from --dap-flags for both "
            "qa_dap_npfit and qa_dap_pmfit (skip error-comparison computation)."
        ),
    )
    parser.add_argument(
        "--dap-include-siii",
        action="store_true",
        help="Include [SIII] 9069 and 9531 lines in DAP pmfit bright-line set.",
    )
    parser.add_argument(
        "--dap-pe-np-stats",
        type=Path,
        default=None,
        help=(
            "Parametric vs non-parametric flux comparison stats per exposure. "
            "Default: <metrics-dir>/DR20_pe_np_comparison_stats.fits"
        ),
    )
    parser.add_argument(
        "--knox-dir",
        type=Path,
        default=None,
        help=(
            "Knox per-exposure lvmCFrame tables (wavecal / flatfield). "
            "Default: <metrics-dir>/from_knox"
        ),
    )

    args = parser.parse_args()

    metrics_dir: Path = args.metrics_dir
    if args.output is None:
        out_path = metrics_dir / "dr20_qa_summary_bitmasks.fits"
    else:
        out_path = args.output

    if args.no_xlsx:
        xlsx_path: Optional[Path] = None
    elif args.output_xlsx is not None:
        xlsx_path = args.output_xlsx
    else:
        xlsx_path = out_path.with_suffix(".xlsx")

    pointings_path = metrics_dir / "dr20_pointings_final_v1.2.0.csv"
    sky_path = metrics_dir / "drpqa_metrics_dr20_sky_final.dat"
    jumps_tar_path = metrics_dir / "DR20_json_jumps.tar.gz"
    phot_path = metrics_dir / "dr20_photometry.csv"
    cont_path = metrics_dir / "cont_snr_dr20_.parquet"
    visual_path = metrics_dir / "DR20_visual_QC.csv"
    stars_cache_path = metrics_dir / "stars_cache.pkl"
    if args.dap_flags is not None:
        dap_flags_path = args.dap_flags
    else:
        dap_flags_path = metrics_dir / "DAP_flux_comparison_flags.fits"
    if args.dap_error_comparison is not None:
        dap_error_path = args.dap_error_comparison
    else:
        dap_error_path = metrics_dir / "DAP_flux_error_comparison.fits"
    if args.dap_pe_np_stats is not None:
        dap_pe_np_stats_path = args.dap_pe_np_stats
    else:
        dap_pe_np_stats_path = metrics_dir / "DR20_pe_np_comparison_stats.fits"
    knox_dir = args.knox_dir if args.knox_dir is not None else metrics_dir / "from_knox"

    master = read_pointings_master(pointings_path)
    expnums = sorted(master.keys())

    sky_rows = parse_pipe_dat_table(sky_path)
    cont_jumps = read_continuum_json_tar(jumps_tar_path)
    phot_rows = read_photometry_csv(phot_path)
    cont_snr = read_cont_snr_parquet(cont_path)
    visual_rows = read_visual_qc(visual_path)
    stars_cache = load_stars_cache(stars_cache_path) if stars_cache_path.exists() else {}
    dap_flags_rows = read_dap_flags_fits(dap_flags_path) if dap_flags_path.exists() else {}
    dap_pe_np_stats = read_pe_np_comparison_stats(dap_pe_np_stats_path)
    dap_error_by_expnum: Dict[int, Table] = {}
    if not args.dap_use_flags:
        dap_error_by_expnum = read_dap_error_comparison_by_expnum(
            dap_error_path, dap_flags_path
        )

    records: Dict[str, list] = {
        "expnum": [],
        "mjd": [],
        "tileid": [],
        "target_type": [],
        "qa_bitmask": [],
        # human-readable per metric
        "qa_brightstars": [],
        "qa_sky_brt": [],
        "qa_absflux": [],
        "qa_relflux_br": [],
        "qa_relflux_rz": [],
        "qa_wavecal_b": [],
        "qa_wavecal_r": [],
        "qa_wavecal_z": [],
        "qa_flatfield": [],
        "qa_skycont_sub": [],
        "qa_skyline_sub": [],
        "qa_dap_npfit": [],
        "qa_dap_pmfit": [],
        "qa_visual_drp": [],
        "qa_visual_dap": [],
    }


    for expnum in expnums:
        base = master[expnum]
        mjd = base["mjd"]
        tileid = base["tileid"]
        sun_alt = base["sun_alt"]

        source_type = get_source_type(expnum, tileid)

        qa_bright_stars, hip_anc = compute_metric_brightstars_from_cache(
            expnum=expnum, stars_cache=stars_cache, mag_limits=(6.0, 10.0)
        )

        # (a) sky brightness
        sky_row = sky_rows.get(expnum)
        if sky_row:
            moon_fli = safe_float(sky_row.get("moon_fli"))
            moon_alt = safe_float(sky_row.get("moon_alt"))
            sci_moon_sep = safe_float(sky_row.get("sci_moon_sep"))
        else:
            moon_fli = safe_float(base.get("moon_fli"))
            moon_alt = safe_float(base.get("moon_alt"))
            sci_moon_sep = safe_float(base.get("sci_moon_sep"))

        qa_sky_brightness = compute_metric_a_sky_brightness(
            moon_fli=moon_fli, sun_alt=sun_alt, moon_alt=moon_alt, sci_moon_sep=sci_moon_sep
        )

        qa_sky_linesub = compute_metric_sky_emission_line_subtraction(sky_row)
        # (b) & (c) continuum jumps (1-sigma fraction only for now)
        cj = cont_jumps.get(expnum, {})
        br_flagged_1sigma = safe_float(cj.get("br_flagged_frac_warn"))
        rz_flagged_1sigma = safe_float(cj.get("rz_flagged_frac_warn"))

        br_flagged_2sigma = safe_float(cj.get("br_flagged_frac_bad"))
        rz_flagged_2sigma = safe_float(cj.get("rz_flagged_frac_bad"))

        qa_relflux_br = compute_metric_jump_from_flagged_frac(br_flagged_1sigma, br_flagged_2sigma)
        qa_relflux_rz = compute_metric_jump_from_flagged_frac(rz_flagged_1sigma, rz_flagged_2sigma)

        # (d) flux calibration from photometry medians/RMS
        qa_absflux, phot_anc = compute_metric_d_fluxcal_from_phot_row(phot_rows.get(expnum))

        # (e) continuum subtraction from parquet (all 3 arms must satisfy band)
        qa_sky_contsub, cont_anc = compute_metric_e_contsub(cont_snr.get(expnum))

        # (f) & (g) visual flags
        vrow = visual_rows.get(expnum, {})
        spectra_raw = (vrow.get("spectra") or "").strip()
        dap_raw = (vrow.get("dap") or "").strip()
        qa_visual_drp = parse_visual_flag(spectra_raw)
        qa_visual_dap = parse_visual_flag(dap_raw)

        knox_table = read_knox_cframe_table(knox_cframe_path(knox_dir, expnum))
        qa_wavecal_b = compute_metric_wavelength_calibration_accuracy(knox_table, "b")
        qa_wavecal_r = compute_metric_wavelength_calibration_accuracy(knox_table, "r")
        qa_wavecal_z = compute_metric_wavelength_calibration_accuracy(knox_table, "z")
        qa_flatfield = compute_metric_flat_fielding_accuracy(knox_table)
        dap_row = dap_flags_rows.get(expnum)
        if args.dap_use_flags:
            qa_dap_npfit = parse_qa2bit_flag(
                dap_row.get("PM_NP_flag") if dap_row else None
            )
            qa_dap_pmfit = parse_qa2bit_flag(
                dap_row.get("resid_flag") if dap_row else None
            )
        else:
            if dap_pe_np_stats:
                qa_dap_npfit = compute_metric_dap_npfit_from_median_perc_diff(
                    dap_pe_np_stats.get(expnum)
                )
            else:
                qa_dap_npfit = parse_qa2bit_flag(
                    dap_row.get("PM_NP_flag") if dap_row else None
                )
            qa_dap_pmfit = compute_metric_dap_pmfit_from_error_row(
                dap_error_by_expnum.get(expnum),
                include_siii=args.dap_include_siii,
            )

        qa_by_metric = {
            "qa_brightstars": qa_bright_stars,
            "qa_sky_brt": qa_sky_brightness,
            "qa_absflux": qa_absflux,
            "qa_relflux_br": qa_relflux_br,
            "qa_relflux_rz": qa_relflux_rz,
            "qa_wavecal_b": qa_wavecal_b,
            "qa_wavecal_r": qa_wavecal_r,
            "qa_wavecal_z": qa_wavecal_z,
            "qa_flatfield": qa_flatfield,
            "qa_skycont_sub": qa_sky_contsub,
            "qa_skyline_sub": qa_sky_linesub,
            "qa_dap_npfit": qa_dap_npfit,
            "qa_dap_pmfit": qa_dap_pmfit,
            "qa_visual_drp": qa_visual_drp,
            "qa_visual_dap": qa_visual_dap,
        }
        qa_bitmask = pack_2bit_metrics(
            qa_by_metric[col] for col in QA_METRIC_COLUMNS_ORDERED
        )

        records["expnum"].append(expnum)
        records["mjd"].append(mjd if mjd is not None else -1)
        records["tileid"].append(tileid if tileid is not None else -1)
        records["target_type"].append(source_type)
        records["qa_bitmask"].append(bitmask_to_hex(np.uint64(qa_bitmask)))
        records["qa_brightstars"].append(QA2BIT_TO_STR[qa_bright_stars])
        records["qa_sky_brt"].append(QA2BIT_TO_STR[qa_sky_brightness])
        records["qa_absflux"].append(QA2BIT_TO_STR[qa_absflux])
        records["qa_relflux_br"].append(QA2BIT_TO_STR[qa_relflux_br])
        records["qa_relflux_rz"].append(QA2BIT_TO_STR[qa_relflux_rz])
        records["qa_wavecal_b"].append(QA2BIT_TO_STR[qa_wavecal_b])
        records["qa_wavecal_r"].append(QA2BIT_TO_STR[qa_wavecal_r])
        records["qa_wavecal_z"].append(QA2BIT_TO_STR[qa_wavecal_z])
        records["qa_flatfield"].append(QA2BIT_TO_STR[qa_flatfield])
        records["qa_skycont_sub"].append(QA2BIT_TO_STR[qa_sky_contsub])
        records["qa_skyline_sub"].append(QA2BIT_TO_STR[qa_sky_linesub])
        records["qa_dap_npfit"].append(QA2BIT_TO_STR[qa_dap_npfit])
        records["qa_dap_pmfit"].append(QA2BIT_TO_STR[qa_dap_pmfit])
        records["qa_visual_drp"].append(QA2BIT_TO_STR[qa_visual_drp])
        records["qa_visual_dap"].append(QA2BIT_TO_STR[qa_visual_dap])

    table = Table(records)
    # Column order: id columns, hex, then human-readable metrics LSB -> MSB by index (see DESC)
    qa_metric_cols = list(QA_METRIC_COLUMNS_ORDERED)
    table = table[
        ["expnum", "mjd", "tileid", "target_type", "qa_bitmask", *qa_metric_cols]
    ]
    table.meta["TARG_TYP"] = "Target types: MW=0, MCs=1, Extragalactic=2"
    table.meta["QA_ENC"] = "2-bit GOOD=00 WARNING=01 MAJOR_ISSUE=10 UNDEFINED=11"
    table.meta["QA_BITS"] = (
        "2-bit fields packed LSB-first in this order (see DESC extension): "
        "bright stars, "
        "sky brightness, "
        "absflux, "
        "relflux br, "
        "relflux rz, "
        "wavecal b, "
        "wavecal r, "
        "wavecal z, "
        "flatfield, "
        "sky contsub, "
        "sky linesub, "
        "dap npfit, "
        "dap pmfit, "
        "visual drp, "
        "visual dap"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write multi-extension FITS: primary + QA table + DESC table
    from astropy.io import fits  # local import to keep dependency surface small

    qa_hdu = fits.BinTableHDU(table, name="QA")
    desc_table = build_desc_table()
    desc_hdu = fits.BinTableHDU(desc_table, name="DESC")
    hdul = fits.HDUList([fits.PrimaryHDU(), qa_hdu, desc_hdu])
    hdul.writeto(out_path, overwrite=True)

    print(f"Wrote {len(table)} rows to {out_path} (extensions: QA, DESC)")

    if xlsx_path is not None:
        write_qa_summary_xlsx(table, xlsx_path)
        print(f"Wrote {len(table)} rows to {xlsx_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
