"""
LVM science IFU layout: which spectrograph each fiber position belongs to.

The 1801 science fibers form a hexagon of ~35 arcmin; they are split between the three
spectrographs in three 120-degree sectors that are fixed in the IFU frame (position angle
measured from the IFU centre minus the IFU position angle sci_pa): sp1 60-180 deg, sp2 180-300,
sp3 300-60 (verified on DRP 1.2.0 and 1.3.2, early and late epochs, sci_pa 0 and 62 deg).
data/ifu_layout.npz holds the IFU-frame offsets (arcsec), spectrograph and fiber id of every
science fiber (from the SLITMAP of exposure 61411); fibers of any exposure are identified by
their nearest reference position (typical mismatch 4", pitch 37"; 0 misassignments on a
DRP 1.2.0 exposure at sci_pa = 62 deg). Used when the SLITMAP is not at hand (cached Halpha maps,
context tiles).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SECTORS = {1: (60.0, 180.0), 2: (180.0, 300.0), 3: (300.0, 60.0)}   # IFU-frame PA ranges (deg)


@lru_cache(maxsize=1)
def reference_layout():
    """(x, y [arcsec, IFU frame], spectrograph, fiberid) of the 1801 science fibers."""
    z = np.load(Path(__file__).resolve().parents[1] / "data" / "ifu_layout.npz")
    return (z["x"].astype(float), z["y"].astype(float), z["spec"].astype(int), z["fiberid"].astype(int))


def ifu_offsets(ra, dec, pa=0.0, center=None):
    """IFU-frame offsets (arcsec; x = 'east', y = 'north' for sci_pa = 0) of fiber positions
    around the IFU centre (median position unless given)."""
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    ra0, de0 = center if center is not None else (np.nanmedian(ra), np.nanmedian(dec))
    e = ((ra - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(de0)) * 3600.0
    n = (dec - de0) * 3600.0
    r = np.hypot(e, n)
    th = np.arctan2(e, n) - np.deg2rad(pa)
    return r * np.sin(th), r * np.cos(th)


def assign_fibers(ra, dec, pa=0.0, max_dist=15.0):
    """Spectrograph id, fiber id and reference IFU-frame position of every fiber (nearest
    reference fiber within max_dist arcsec; spectrograph 0 / fiberid -1 otherwise).
    Returns dict(spec, fiberid, x, y)."""
    xr, yr, sr, fr = reference_layout()
    x, y = ifu_offsets(ra, dec, pa)
    spec = np.zeros(x.size, int)
    fid = np.full(x.size, -1, int)
    xo, yo = np.full(x.size, np.nan), np.full(x.size, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.where(ok)[0]
    for i0 in range(0, idx.size, 512):
        ii = idx[i0:i0 + 512]
        d = np.hypot(x[ii, None] - xr[None, :], y[ii, None] - yr[None, :])
        k = np.argmin(d, axis=1)
        good = d[np.arange(ii.size), k] <= max_dist
        spec[ii[good]] = sr[k[good]]
        fid[ii[good]] = fr[k[good]]
        xo[ii[good]], yo[ii[good]] = xr[k[good]], yr[k[good]]
    return dict(spec=spec, fiberid=fid, x=xo, y=yo)
