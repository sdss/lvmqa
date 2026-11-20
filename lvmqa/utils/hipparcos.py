"""
Local cache of bright Hipparcos stars (Vizier I/239/hip_main).

The cache is a single small FITS table with all Hipparcos stars brighter than
``max_mag`` (default V < 6.5, ~9000 stars). It is downloaded once with astroquery
and afterwards all bright-star checks are done offline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from astropy.table import Table

log = logging.getLogger("lvmqa")

HIP_CATALOG = "I/239/hip_main"
_LOADED: Dict[str, Table] = {}


def default_cache_path(cache_dir: Path, max_mag: float) -> Path:
    return Path(cache_dir) / f"hipparcos_hip_main_V{max_mag:.1f}.fits"


def download_catalog(path: Path, max_mag: float) -> Table:
    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("astroquery is required to build the Hipparcos cache "
                           "(pip install astroquery)") from exc
    viz = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"],
                 column_filters={"Vmag": f"<{max_mag}"}, row_limit=-1)
    res = viz.query_constraints(catalog=HIP_CATALOG, Vmag=f"<{max_mag}")
    if not res:
        raise RuntimeError("empty Vizier response for Hipparcos")
    t = res[0]
    out = Table()
    out["HIP"] = np.asarray(t["HIP"], dtype=np.int64)
    out["RAICRS"] = np.asarray(t["RAICRS"], dtype=float)
    out["DEICRS"] = np.asarray(t["DEICRS"], dtype=float)
    out["Vmag"] = np.asarray(t["Vmag"], dtype=float)
    out.meta["SOURCE"] = f"Vizier {HIP_CATALOG}"
    out.meta["MAXMAG"] = max_mag
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path, overwrite=True)
    log.info("Hipparcos cache written: %s (%d stars, V<%.1f)", path, len(out), max_mag)
    return out


def ensure_catalog(path: Path, max_mag: float, allow_download: bool = True) -> Optional[Path]:
    """Return the cache path if it exists (downloading it if allowed), else None."""
    path = Path(path)
    if path.exists():
        return path
    if not allow_download:
        log.warning("Hipparcos cache %s missing and download disabled", path)
        return None
    try:
        download_catalog(path, max_mag)
        return path
    except Exception as exc:
        log.error("could not build Hipparcos cache (%s); bright-star metric -> UNDEFINED", exc)
        return None


def load_catalog(path: Path) -> Table:
    key = str(Path(path).resolve())
    if key not in _LOADED:
        t = Table.read(path)
        t = t[np.isfinite(np.asarray(t["Vmag"], dtype=float))]
        _LOADED[key] = t
    return _LOADED[key]


def unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def angsep_arcsec(ra0, dec0, ra, dec) -> np.ndarray:
    """Angular separation (arcsec) between one position and an array of positions."""
    v0 = unit_vectors(ra0, dec0)
    v = unit_vectors(ra, dec)
    cosd = np.clip(v @ v0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosd)) * 3600.0
