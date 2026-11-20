"""
Spectrograph flux-scale pattern inside ONE exposure from adjacent fiber pairs across the sector
boundaries of the IFU (used by qa_gradient).

The three spectrographs observe three 120-deg sectors of the IFU (utils/ifu_layout.py). A
relative flux-scale error of spectrograph b with respect to a shows up as a jump of log F between
adjacent fibers on the two sides of the a|b boundary. Real Halpha structure also changes log F
between neighbours, so every cross-boundary pair is compared with "control" pairs: adjacent
fibers of the SAME sector, with the SAME step vector (one of the 6 hexagonal directions), whose
midpoints lie close to it (within ctrl_radius pitches). The controls measure the physical change
of log F along that step at that place; the difference cross - median(controls) is what the
physics cannot explain. Per boundary: median over its cross pairs (robust to filaments, stars,
bad fibers), error 1.2533 NMAD / sqrt(n).

Three boundaries give three differences d_12 = o2 - o1, d_23 = o3 - o2, d_31 = o1 - o3 of the
spectrograph offsets o (log10 flux scale); they are solved by weighted least squares with
sum(o) = 0. Their closure d_12 + d_23 + d_31 (should be 0) is a consistency check.
"""
from __future__ import annotations

import numpy as np


def _nmad(v):
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def adjacent_pairs(x, y, max_sep):
    """Index pairs (i < j) of fibers closer than max_sep (arcsec)."""
    n = x.size
    I, J = [], []
    for i0 in range(0, n, 512):
        xi, yi = x[i0:i0 + 512, None], y[i0:i0 + 512, None]
        d = np.hypot(xi - x[None, :], yi - y[None, :])
        a, b = np.where((d < max_sep) & (d > 0))
        a = a + i0
        m = a < b
        I.append(a[m])
        J.append(b[m])
    return np.concatenate(I) if I else np.zeros(0, int), np.concatenate(J) if J else np.zeros(0, int)


def sector_offsets(x, y, spec, logf, ok, pitch=37.0, ctrl_radius=2.2, min_pairs=10, min_ctrl=2):
    """
    x, y  : IFU-frame fiber positions (arcsec; reference layout positions recommended)
    spec  : spectrograph id (1-3) per fiber
    logf  : log10 flux per fiber; ok: usable fibers (S/N cut etc.)
    Returns dict(o (3,), e_o (3,), d {(a,b): (value, error, n)}, closure, e_closure, n_cross).
    """
    x, y, spec, logf = (np.asarray(v) for v in (x, y, spec, logf))
    use = np.asarray(ok, bool) & np.isfinite(x) & np.isfinite(y) & np.isfinite(logf) & (spec > 0)
    idx = np.where(use)[0]
    out = dict(o=np.full(3, np.nan), e_o=np.full(3, np.nan), d={}, closure=np.nan, e_closure=np.nan, n_cross=0)
    if idx.size < 20:
        return out
    I, J = adjacent_pairs(x[idx], y[idx], 1.3 * pitch)
    I, J = idx[I], idx[J]
    vx, vy = x[J] - x[I], y[J] - y[I]
    ang = np.degrees(np.arctan2(vx, vy)) % 360.0
    off = float(np.median(ang % 60.0))                       # orientation of the hexagonal grid
    k = np.round((ang - off) / 60.0).astype(int) % 6           # 6 step directions
    dl = logf[J] - logf[I]
    mx, my = 0.5 * (x[I] + x[J]), 0.5 * (y[I] + y[J])
    same = spec[I] == spec[J]
    # controls in both orientations: (direction k, dl) and (k + 3, -dl)
    cs, ck = np.r_[spec[I][same], spec[I][same]], np.r_[k[same], (k[same] + 3) % 6]
    cdl = np.r_[dl[same], -dl[same]]
    cx, cy = np.r_[mx[same], mx[same]], np.r_[my[same], my[same]]
    rows = []
    for a, b in ((1, 2), (2, 3), (3, 1)):
        m_ab = (spec[I] == a) & (spec[J] == b)
        m_ba = (spec[I] == b) & (spec[J] == a)
        # orient every cross pair from sector a to sector b
        kk = np.r_[k[m_ab], (k[m_ba] + 3) % 6]
        dd = np.r_[dl[m_ab], -dl[m_ba]]
        px, py = np.r_[mx[m_ab], mx[m_ba]], np.r_[my[m_ab], my[m_ba]]
        corr = []
        side = (cs == a) | (cs == b)
        for q in range(dd.size):
            sel = side & (ck == kk[q]) & (np.hypot(cx - px[q], cy - py[q]) <= ctrl_radius * pitch)
            if sel.sum() >= min_ctrl:
                corr.append(dd[q] - np.median(cdl[sel]))
        corr = np.asarray(corr)
        out["n_cross"] += int(corr.size)
        if corr.size >= min_pairs:
            out["d"][(a, b)] = (float(np.median(corr)), float(1.2533 * _nmad(corr) / np.sqrt(corr.size)), int(corr.size))
            rows.append((a, b))
    if len(rows) < 2:
        return out
    # weighted least squares for o with sum(o) = 0: o_b - o_a = d_ab
    A, y_, w = [], [], []
    for a, b in rows:
        v, e, _ = out["d"][(a, b)]
        r = np.zeros(3)
        r[b - 1], r[a - 1] = 1.0, -1.0
        A.append(r)
        y_.append(v)
        w.append(1.0 / max(e, 1e-4) ** 2)
    A, y_, w = np.array(A), np.array(y_), np.array(w)
    N = A.T @ (w[:, None] * A) + 1e6 * np.ones((3, 3))     # + sum(o) = 0 constraint
    cov = np.linalg.inv(N)
    o = cov @ (A.T @ (w * y_))
    out["o"], out["e_o"] = o, np.sqrt(np.abs(np.diag(cov)))
    if len(rows) == 3:
        vals = [out["d"][r_] for r_ in rows]
        out["closure"] = float(sum(v for v, _, _ in vals))
        out["e_closure"] = float(np.sqrt(sum(e ** 2 for _, e, _ in vals)))
    return out
