"""
Extended stellar populations in the DAP results: connected groups of > ``min_region`` adjacent
fibers with stellar-continuum S/N (RSP med_flux / e_med_flux) > ``min_snr``. Fibers are
adjacent if their centres are closer than ``link_pitch`` x fiber pitch (1.5 x 37" ~ 56", i.e.
first-ring neighbours). Shared by qa_starkinematics and qa_status (status_sp).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

C_KMS = 299792.458


def _components(P: np.ndarray, link: float) -> np.ndarray:
    n = len(P)
    if n == 0:
        return np.zeros(0, int)
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    adj = d <= link
    lab = -np.ones(n, int)
    c = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = c
        while stack:
            j = stack.pop()
            for k in np.where(adj[j] & (lab < 0))[0]:
                lab[k] = c
                stack.append(k)
        c += 1
    return lab


def stellar_regions(ctx, min_snr: float = 10.0, min_region: int = 15, link_pitch: float = 1.5) -> Dict:
    """{'regions': [index arrays into PT], 'snr': array, 'n_snr': int, 'pitch': float}."""
    pt = ctx.dap_table("PT")
    rsp = ctx.dap_table("RSP")
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.asarray(rsp["med_flux"], float) / np.asarray(rsp["e_med_flux"], float)
    ra, dec = np.asarray(pt["ra"], float), np.asarray(pt["dec"], float)
    x = (ra - np.nanmedian(ra)) * np.cos(np.deg2rad(np.nanmedian(dec))) * 3600.0
    y = (dec - np.nanmedian(dec)) * 3600.0
    # fiber pitch from all fibers (sample of 300 for speed)
    k = np.arange(0, len(x), max(1, len(x) // 300))
    dd = np.hypot(x[k, None] - x[None, :], y[k, None] - y[None, :])
    dd[dd == 0] = np.inf
    pitch = float(np.median(dd.min(axis=1)))
    sel = np.where(np.isfinite(snr) & (snr > min_snr) & np.isfinite(x) & np.isfinite(y))[0]
    lab = _components(np.c_[x, y][sel], link_pitch * pitch)
    regions: List[np.ndarray] = []
    if sel.size:
        for c in np.unique(lab):
            m = sel[lab == c]
            if m.size > min_region:
                regions.append(m)
    regions.sort(key=lambda a: -a.size)
    return {"regions": regions, "snr": snr, "n_snr": int(sel.size), "pitch": pitch}


def cached_regions(ctx, **kw) -> Dict:
    key = "stellar_regions_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    return ctx.shared(key, lambda c: stellar_regions(c, **kw))
