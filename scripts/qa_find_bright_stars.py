#!/usr/bin/env python3
"""
Build a Hipparcos bright-star cache for LVM pointings.

This script reads table with pointing coordinates and queries Hipparcos for sources
within the LVM IFU field of view (assuming regular hexagon with 1813" outer radius),
then saves a pickle cache.

Cache contents (dict):
  {
    "meta": {...},
    "by_expnum": {
       <expnum>: {
          "sci_ra": float,
          "sci_dec": float,
          "sci_pa": float,
          "band": "Vmag",
          "max_mag_in_cache": float,
          "n_stars": int,
          "brightest_mag": float|nan,
          "vmag_list": [float, ...]   # only stars inside hex and brighter than max_mag_in_cache
       },
       ...
    }
  }

"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from astroquery.vizier import Vizier
from astropy import units as u
from astropy.coordinates import SkyCoord

import numpy as np


FOV_OUTER_DIAMETER_ARCSEC = 1813.0
MAX_MAG = 6.0

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


def normalize_expnum(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None


def _rotate_xy(x: np.ndarray, y: np.ndarray, pa_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(-pa_deg)
    c, s = np.cos(theta), np.sin(theta)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def _in_regular_hexagon(x_arcsec: np.ndarray, y_arcsec: np.ndarray, r_arcsec: float) -> np.ndarray:
    x = np.abs(x_arcsec)
    y = np.abs(y_arcsec)
    y_max = (np.sqrt(3.0) / 2.0) * r_arcsec
    return (x <= r_arcsec) & (y <= y_max) & ((x + (y / np.sqrt(3.0))) <= r_arcsec)


def query_hipparcos_sources(
    ra_deg: float, dec_deg: float, *, radius_arcsec: float, max_mag: float
):
    """
    Query Hipparcos main catalog via Vizier and return the first table (or None).
    """

    viz = Vizier(columns=["RAICRS", "DEICRS", "Vmag"], column_filters={"Vmag": f"<{max_mag}"})
    try:
        res = viz.query_region(
        SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg),
        radius=radius_arcsec * u.arcsec,
        catalog="I/239/hip_main",
        )
    except Exception as e:
        return None

    if not res:
        return []
    return res[0]


def read_pointings(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expnum = normalize_expnum(row.get("expnum"))
            if expnum is None:
                continue
            ra = safe_float(row.get("sci_ra"))
            dec = safe_float(row.get("sci_dec"))
            pa = safe_float(row.get("sci_pa"))
            if ra is None or dec is None or pa is None:
                continue
            out[expnum] = {"sci_ra": ra, "sci_dec": dec, "sci_pa": pa}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Hipparcos bright-star cache pickle.")
    ap.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing dr20_pointings_final_v1.2.0.csv",
    )
    ap.add_argument(
        "--pointings",
        type=Path,
        default=None,
        help="Path to pointings CSV. Default: <metrics-dir>/dr20_pointings_final_v1.2.0.csv",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path. Default: <metrics-dir>/stars_cache.pkl",
    )
    args = ap.parse_args()

    pointings_path = args.pointings or (args.metrics_dir / "dr20_pointings_final_v1.2.0.csv")
    out_path = args.output or (args.metrics_dir / "stars_cache.pkl")

    pointings = read_pointings(pointings_path)
    r_arcsec = FOV_OUTER_DIAMETER_ARCSEC / 2.0

    # Cache per unique pointing (ra,dec,pa) to avoid repeated Vizier calls
    by_pointing: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    by_expnum: Dict[int, Dict[str, Any]] = {}

    for expnum, p in pointings.items():
        ra, dec, pa = p["sci_ra"], p["sci_dec"], p["sci_pa"]
        key = (round(ra, 6), round(dec, 6), round(pa, 3))
        if key not in by_pointing:
            tab = query_hipparcos_sources(ra, dec, radius_arcsec=r_arcsec, max_mag=MAX_MAG)
            if tab is None or len(tab) == 0:
                by_pointing[key] = {
                    "band": "Vmag",
                    "max_mag_in_cache": float(MAX_MAG),
                    "vmag_list": tab,
                }
            else:
                dra = (np.array(tab["RAICRS"], dtype=float) - ra) * np.cos(np.deg2rad(dec)) * 3600.0
                ddec = (np.array(tab["DEICRS"], dtype=float) - dec) * 3600.0
                x, y = _rotate_xy(dra, ddec, pa)
                inside = _in_regular_hexagon(x, y, r_arcsec=r_arcsec)
                vmag_in = np.array(tab["Vmag"], dtype=float)[inside]
                vmag_list = sorted(float(v) for v in vmag_in if np.isfinite(v))
                by_pointing[key] = {
                    "band": "Vmag",
                    "max_mag_in_cache": float(MAX_MAG),
                    "vmag_list": vmag_list,
                }

        rec = by_pointing[key]
        vmag_list = rec["vmag_list"]
        by_expnum[expnum] = {
            "sci_ra": ra,
            "sci_dec": dec,
            "sci_pa": pa,
            "band": rec["band"],
            "max_mag_in_cache": rec["max_mag_in_cache"],
            "n_stars": len(vmag_list),
            "brightest_mag": (min(vmag_list) if vmag_list else float("nan")),
            "vmag_list": vmag_list,
            "fov_outer_diameter_arcsec": float(FOV_OUTER_DIAMETER_ARCSEC),
        }

    payload = {
        "meta": {
            "catalog": "I/239/hip_main",
            "band": "Vmag",
            "max_mag_in_cache": float(MAX_MAG),
            "fov_outer_diameter_arcsec": float(FOV_OUTER_DIAMETER_ARCSEC),
        },
        "by_expnum": by_expnum,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote cache for {len(by_expnum)} exposures to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

