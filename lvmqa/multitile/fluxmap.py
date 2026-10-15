"""
qa_fluxmap - PROTOTYPE of a multi-tile flux-consistency test.

Idea
----
Adjacent LVM tiles overlap (centres 0.42 deg apart in the test set; ~45-50 fibers of one
pointing lie within 20" of a fiber of the neighbour). The same sky region observed in two
exposures must give the same Halpha surface brightness, so the flux scale of every
exposure can be checked against its neighbours without any external reference:

1. Nodes: exposures (a tile observed twice gives two nodes). Per node: DAP NP Halpha flux,
   error and fiber positions (PT ra/dec).
2. Edges: every pair of nodes with overlapping footprints. Fibers are matched to the nearest
   fiber of the other node within ``match_radius`` (20"); pairs with S/N > ``min_snr`` in
   both are kept. Edge value d_ij = robust (Huber) mean of log10(F_i / F_j) over the matched
   fibers, uncertainty from the scatter (sky-position mismatch and Halpha gradients make the
   per-fiber scatter much larger than the photon noise, but the median over ~50 pairs is
   stable), plus n_match.
3. Global model: d_ij = s_i - s_j + e_ij, with s_i the log flux-scale offset of node i.
   Weighted least squares on the graph with a gauge constraint (weighted mean of s = 0;
   finally s is shifted so that its median is 0, i.e. "the majority defines the scale").
   Iteratively re-weighted with a Tukey biweight (or Huber) loss on the standardised edge
   residuals r_ij = (d_ij - s_i + s_j) / sqrt(sigma_ij^2 + sigma_sys^2): inconsistent edges
   (bad overlap, astrophysical variability, a corrupted spectrograph on one side) lose
   their weight instead of dragging the solution, and no node is assumed to be valid a priori.
4. Per node: s_i (dex) and its uncertainty (diagonal of the inverse normal matrix), number of
   edges, number of down-weighted edges, median |residual|. Flag (prototype):
     GOOD     |s| <= 0.04 dex (10%) or not significant (< 3 sigma)
     WARNING  |s| > 0.04 dex, or > half of the node's edges rejected
     MAJOR    |s| > 0.10 dex (26%)
     UNDEFINED  node without (accepted) edges
   A node whose edges are all rejected is "inconsistent" (cannot be calibrated, not
   necessarily wrong) -> WARNING.

Limitations / next steps: a node connected by a single edge cannot be distinguished from its
neighbour (degenerate); spectrograph-specific scale errors need nodes = (exposure,
spectrograph); line-free continuum or [SII]/[NII] maps can be added as independent layers;
dithered / repeated exposures of one tile add strong edges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Node:
    name: str
    ra: np.ndarray
    dec: np.ndarray
    flux: np.ndarray
    eflux: np.ndarray
    pa: float = 0.0   # IFU position angle (drpall sci_pa, deg): edge directions in the IFU frame
    mjd: int = -1     # night: the local reference uses exposures of other nights (v1.15)

    @property
    def center(self):
        return float(np.nanmedian(self.ra)), float(np.nanmedian(self.dec))


@dataclass
class Edge:
    i: int
    j: int
    d: float          # log10(F_i/F_j)
    sigma: float
    n: int
    faint: bool = False              # v1.15: ratio-of-sums estimate (too few fibers with S/N > min_snr in both)
    ia: Optional[np.ndarray] = None  # matched fiber pairs (indices into node i / node j)
    ib: Optional[np.ndarray] = None


@dataclass
class FluxMapResult:
    names: List[str]
    s: np.ndarray
    e_s: np.ndarray
    edges: List[Edge]
    weights: np.ndarray
    resid: np.ndarray
    n_edges: np.ndarray
    n_rejected: np.ndarray
    flags: List[str] = field(default_factory=list)

    def table(self):
        from astropy.table import Table
        return Table(dict(node=self.names, s_dex=self.s, e_s_dex=self.e_s, scale=10 ** self.s,
                          n_edges=self.n_edges, n_rejected=self.n_rejected, flag=self.flags))


def _huber_mean(x, c=1.345, nit=20):
    x = x[np.isfinite(x)]
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m)) or 1e-6
    for _ in range(nit):
        r = (x - m) / s
        w = np.clip(c / np.maximum(np.abs(r), 1e-12), None, 1.0)
        m_new = np.sum(w * x) / np.sum(w)
        if abs(m_new - m) < 1e-6 * s:
            break
        m = m_new
    return m, s


def _nearest(A: np.ndarray, B: np.ndarray, chunk: int = 512):
    """For each point of A: distance and index of the nearest point of B (numpy only)."""
    dist = np.empty(len(A))
    idx = np.empty(len(A), dtype=int)
    for i0 in range(0, len(A), chunk):
        a = A[i0:i0 + chunk]
        d = np.hypot(a[:, None, 0] - B[None, :, 0], a[:, None, 1] - B[None, :, 1])
        j = np.argmin(d, axis=1)
        idx[i0:i0 + chunk] = j
        dist[i0:i0 + chunk] = d[np.arange(len(a)), j]
    return dist, idx


def _edge_value(fa, fb, ea, eb, min_snr=5.0, min_match=10, faint_snr=3.0, faint_det=5.0, floor_dex=-1.5,
                nboot=30):
    """
    Flux-scale offset d = log10(F_a / F_b) of one overlap from the matched fiber pairs.
    1) regular: pairs with S/N > min_snr in BOTH exposures; Huber mean of log10(fa / fb),
       error 1.2533 x scale / sqrt(n).
    2) v1.15 faint fallback (too few such pairs: faint fields, or one exposure much too faint, whose
       fibers then fail the per-fiber S/N cut on that side - a dark tile used to get no edge at all):
       pairs selected symmetrically on the summed S/N (fa + fb) / sqrt(ea^2 + eb^2) > faint_snr,
       k = sum(fa) / sum(fb) with 4-NMAD clipping of the normalised pair residuals
       (fa - k fb) / sqrt(ea^2 + k^2 eb^2), the brighter summed side in the denominator (which must
       be detected with summed S/N > faint_det); k floored at 10**floor_dex (an exposure without
       signal, or with negative Halpha); all matched pairs if < min_match pass the summed-S/N cut;
       error from a bootstrap over the pairs.
    Returns (d, sigma, n, faint) or None.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (fa / ea > min_snr) & (fb / eb > min_snr) & (fa > 0) & (fb > 0)
    if ok.sum() >= min_match:
        x = np.log10(fa[ok] / fb[ok])
        mu, sc = _huber_mean(x)
        return float(mu), float(1.2533 * sc / np.sqrt(ok.sum())), int(ok.sum()), False
    with np.errstate(invalid="ignore", divide="ignore"):
        fin = np.isfinite(fa) & np.isfinite(fb) & (ea > 0) & (eb > 0) & np.isfinite(ea) & np.isfinite(eb)
        sel = fin & ((fa + fb) / np.hypot(ea, eb) > faint_snr)
    if sel.sum() < min_match:
        sel = fin                  # very faint overlap: all matched pairs (the sums must still be detected)
    if sel.sum() < min_match:
        return None
    a, b, va, vb = fa[sel], fb[sel], ea[sel] ** 2, eb[sel] ** 2
    # orientation: the brighter summed side in the denominator (an exposure with ~zero or negative
    # Halpha - e.g. over-subtracted sky - must give a strongly negative d, not a failed edge)
    flip = a.sum() > b.sum()
    if flip:
        a, b, va, vb = b, a, vb, va

    def ratio(a, b, va, vb):
        keep = np.ones(a.size, bool)
        k = np.nan
        for _ in range(5):
            Sb = b[keep].sum()
            if not Sb > 0:
                return np.nan, keep
            k = max(a[keep].sum() / Sb, 10 ** floor_dex)
            r = (a - k * b) / np.sqrt(va + k * k * vb)
            med = np.median(r[keep])
            mad = 1.4826 * np.median(np.abs(r[keep] - med))
            new = np.abs(r - med) < 4 * mad if mad > 0 else keep
            if new.sum() < min_match or np.array_equal(new, keep):
                break
            keep = new
        return k, keep

    k, keep = ratio(a, b, va, vb)
    if not np.isfinite(k):
        return None
    det = b[keep].sum() / np.sqrt(vb[keep].sum())      # the brighter side must be detected
    if not det > faint_det:
        return None
    rng = np.random.default_rng(12345)
    bs = []
    for _ in range(nboot):
        j = rng.integers(0, a.size, a.size)
        kb, _ = ratio(a[j], b[j], va[j], vb[j])
        if np.isfinite(kb):
            bs.append(np.log10(kb))
    e = float(np.std(bs)) if len(bs) > 5 else 0.3
    d = float(np.log10(k))
    return (-d if flip else d), max(e, 1e-3), int(keep.sum()), True


def build_edges(nodes: Sequence[Node], match_radius: float = 20.0, min_snr: float = 5.0,
                max_sep_deg: float = 0.6, min_match: int = 10, faint: bool = True,
                faint_snr: float = 3.0, keep_pairs: bool = True) -> List[Edge]:
    edges = []
    cen = np.array([n.center for n in nodes], float).reshape(-1, 2)   # once (nanmedian per node)
    pairs = []
    for a in range(len(nodes) - 1):           # candidate pairs, vectorised per row
        dra = (cen[a + 1:, 0] - cen[a, 0] + 180.0) % 360.0 - 180.0
        sep = np.hypot(dra * np.cos(np.deg2rad(cen[a, 1])), cen[a + 1:, 1] - cen[a, 1])
        pairs += [(a, a + 1 + int(k)) for k in np.where(sep <= max_sep_deg)[0]]
    for a, b in pairs:
        (ra1, de1), (ra2, de2) = cen[a], cen[b]
        c = np.cos(np.deg2rad(0.5 * (de1 + de2)))
        ua = (nodes[a].ra - ra1 + 180.0) % 360.0 - 180.0   # RA relative to node a (wrap-safe)
        ub = (nodes[b].ra - ra1 + 180.0) % 360.0 - 180.0
        A = np.c_[ua * c * 3600.0, nodes[a].dec * 3600.0]
        B = np.c_[ub * c * 3600.0, nodes[b].dec * 3600.0]
        ga = np.isfinite(A).all(1)
        gb = np.isfinite(B).all(1)
        if not ga.any() or not gb.any():
            continue
        # v1.15: only fibers that can overlap (within the other IFU's extent + match radius): the
        # nearest-neighbour search then runs on ~100 instead of ~1800 fibers per side
        ca, cb = np.nanmedian(A[ga], axis=0), np.nanmedian(B[gb], axis=0)
        ra_ = np.nanmax(np.hypot(*(A[ga] - ca).T)) + match_radius
        rb_ = np.nanmax(np.hypot(*(B[gb] - cb).T)) + match_radius
        ga &= np.hypot(*(np.nan_to_num(A) - cb).T) <= rb_
        gb &= np.hypot(*(np.nan_to_num(B) - ca).T) <= ra_
        if ga.sum() < min_match or gb.sum() < min_match:
            continue
        dist, idx = _nearest(A[ga], B[gb])
        ia = np.where(ga)[0]
        ib = np.where(gb)[0][idx]
        m = dist < match_radius
        ia, ib = ia[m], ib[m]
        if ia.size < min_match:
            continue
        r = _edge_value(nodes[a].flux[ia], nodes[b].flux[ib], nodes[a].eflux[ia], nodes[b].eflux[ib],
                        min_snr, min_match, faint_snr if faint else np.inf)
        if r is None:
            continue
        mu, sig, n, fnt = r
        edges.append(Edge(a, b, mu, sig, n, fnt, ia if keep_pairs else None, ib if keep_pairs else None))
    return edges


def _lad(A: np.ndarray, d: np.ndarray, wt: np.ndarray, nit: int = 200, eps: float = 1e-5) -> np.ndarray:
    """Weighted L1 (least absolute deviations) solution of A s = d by IRLS (numpy only)."""
    n = A.shape[1]
    w = wt.copy()
    s = np.zeros(n)
    for _ in range(nit):
        N = A.T @ (w[:, None] * A) + 1e-9 * np.eye(n)
        s_new = np.linalg.solve(N, A.T @ (w * d))
        r = np.abs(d - A @ s_new)
        w = wt / np.maximum(r, eps)
        if np.max(np.abs(s_new - s)) < 1e-7:
            s = s_new
            break
        s = s_new
    return s


def _components(n_nodes: int, edges: Sequence[Edge]) -> np.ndarray:
    """Connected-component label per node (union-find); nodes without edges get their own label."""
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        ra_, rb_ = find(e.i), find(e.j)
        if ra_ != rb_:
            parent[ra_] = rb_
    roots = [find(a) for a in range(n_nodes)]
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def _tangent(centers: np.ndarray, idx: np.ndarray):
    """Tangent-plane coordinates (deg) of the node centres idx around their median centre."""
    ra, de = centers[idx, 0], centers[idx, 1]
    r0, d0 = np.median(ra), np.median(de)
    return ((ra - r0 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(d0)), de - d0


def edge_directions(edges: Sequence[Edge], centers: np.ndarray, pa: Optional[np.ndarray] = None) -> np.ndarray:
    """Direction of every edge i -> j (rad, east of north) in the IFU frame (sky PA minus the mean
    sci_pa of the two exposures)."""
    i = np.array([e.i for e in edges], int)
    j = np.array([e.j for e in edges], int)
    c = np.cos(np.deg2rad(0.5 * (centers[i, 1] + centers[j, 1])))
    dra = (centers[j, 0] - centers[i, 0] + 180.0) % 360.0 - 180.0
    th = np.arctan2(dra * c, centers[j, 1] - centers[i, 1])
    if pa is not None:
        th = th - np.deg2rad(0.5 * (pa[i] + pa[j]))
    return th


def solve(n_nodes: int, edges: Sequence[Edge], loss: str = "tukey", c: Optional[float] = None,
          sigma_sys="auto", sys_floor: float = 0.005, lad_floor: float = 0.03, nit: int = 100,
          centers: Optional[np.ndarray] = None, pa: Optional[np.ndarray] = None, pattern: str = "none",
          pattern_min_nodes: int = 30, pattern_min_edges: int = 50, dense_max: int = 1500) -> Dict[str, np.ndarray]:
    """
    Robust solution of d_ij = s_i - s_j [+ a cos(theta_ij) + b sin(theta_ij)] on the overlap graph.

    * Every connected component gets its own gauge (sum of s = 0; finally shifted to median 0).
    * pattern = 'dipole' (or 'auto' with >= pattern_min_edges edges and a component of
      >= pattern_min_nodes nodes): a common IFU-internal flux pattern, i.e. a dependence of the
      edge offsets on the overlap direction theta_ij in the IFU frame (a flux gradient across the
      IFU). On a mosaic such a pattern is exactly degenerate with a linear ramp of s across the
      mosaic (the reason why the uncorrected solution of a large mosaic drifts by +-0.7 dex), so the
      large components also get a "no linear trend of s" gauge: a large-scale linear trend of the
      flux scale across a mosaic is attributed to the instrument, not to the exposures.
      (A 3-fold term, e.g. spectrograph sectors, is not identifiable on a regular hexagonal mosaic:
      it is degenerate with a 3-sublattice pattern of s.)
    * Start: weighted L1 (IRLS), then Tukey (or Huber) IRLS on the standardised residuals.
      sigma_sys='auto': extra per-edge scatter = NMAD of all raw edge residuals (floor sys_floor);
      the formal edge errors (~0.002 dex) are far too small.
    Returns dict(s, e_s, w, resid, A, sigma_sys, pattern (name), dipole (dex), dipole_pa (deg, IFU
    frame: direction of the brighter IFU side), labels (components), I, J (edge node indices)).
    Graphs with more than ``dense_max`` nodes use _solve_sparse (v1.15; dense normal matrices of
    N^2 elements and N^3 solves are not feasible for 6000-40000 exposures).
    """
    if n_nodes > dense_max:
        return _solve_sparse(n_nodes, edges, loss=loss, c=c, sigma_sys=sigma_sys, sys_floor=sys_floor,
                             lad_floor=lad_floor, nit=nit, centers=centers, pa=pa, pattern=pattern,
                             pattern_min_nodes=pattern_min_nodes, pattern_min_edges=pattern_min_edges)
    E = len(edges)
    A = np.zeros((E, n_nodes))
    d = np.array([e.d for e in edges], float)
    sig_e = np.array([e.sigma for e in edges], float)
    for k, e in enumerate(edges):
        A[k, e.i], A[k, e.j] = 1.0, -1.0
    labels = _components(n_nodes, edges)
    deg = np.abs(A).sum(axis=0)
    comps = [np.where(labels == l_)[0] for l_ in np.unique(labels)]
    big = [cmp for cmp in comps if len(cmp) >= pattern_min_nodes]
    use_pat = (centers is not None and E > 0 and
               (pattern == "dipole" or (pattern == "auto" and E >= pattern_min_edges and big)))
    if use_pat:
        th = edge_directions(edges, np.asarray(centers, float), None if pa is None else np.asarray(pa, float))
        X = np.c_[np.cos(th), np.sin(th)]
    else:
        X = np.zeros((E, 0))
    M = np.c_[A, X]
    npar = M.shape[1]
    G = []
    for cmp in comps:
        row = np.zeros(npar)
        if len(cmp) == 1 or deg[cmp].sum() == 0:
            row[cmp] = 1.0          # isolated node: s = 0
            G.append(row)
            continue
        row[cmp] = 1.0
        G.append(row)
        if use_pat and len(cmp) >= pattern_min_nodes:
            x, y = _tangent(np.asarray(centers, float), cmp)
            for v in (x, y):
                r_ = np.zeros(npar)
                r_[cmp] = v
                G.append(r_)
    G = np.array(G) if G else np.zeros((0, npar))
    c = c if c is not None else (4.685 if loss == "tukey" else 1.345)

    def wls(W):
        N = M.T @ (W[:, None] * M)
        kap = 1e3 * max(float(np.max(np.diag(N))), 1.0)
        N = N + kap * (G.T @ G)
        return np.linalg.solve(N, M.T @ (W * d)), N

    def nmad(r):
        return 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else 0.0

    # robust start: weighted L1 (IRLS) with nearly uniform weights - a minority of corrupted edges
    # is out-voted by the alternative paths instead of pulling its nodes
    wt = 1.0 / np.sqrt(sig_e ** 2 + lad_floor ** 2)
    w1 = wt.copy()
    x = np.zeros(npar)
    for _ in range(200):
        x_new, _ = wls(w1)
        r = np.abs(d - M @ x_new)
        w1 = wt / np.maximum(r, 1e-5)
        done = np.max(np.abs(x_new - x)) < 1e-7
        x = x_new
        if done:
            break
    raw = d - M @ x
    ssys = max(nmad(raw), sys_floor) if sigma_sys == "auto" else float(sigma_sys)
    sig = np.sqrt(sig_e ** 2 + ssys ** 2)

    def weights(raw, sig):
        u = np.abs(raw / sig) / c
        if loss == "tukey":
            return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        return np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))

    w = weights(raw, sig)
    N = None
    for _ in range(nit):
        x_new, N = wls(w / sig ** 2)
        raw = d - M @ x_new
        if sigma_sys == "auto":
            ssys = max(nmad(raw), sys_floor)   # all edges: NMAD is robust, no collapse to the floor
        sig = np.sqrt(sig_e ** 2 + ssys ** 2)
        w_new = weights(raw, sig)
        done = np.allclose(x_new, x, atol=1e-6) and np.allclose(w_new, w, atol=1e-4)
        x, w = x_new, w_new
        if done:
            break
    if N is None:
        x, N = wls(w / sig ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.linalg.pinv(N)
    s = x[:n_nodes].copy()
    connected = (np.abs(A).T @ (w > 0.5)) > 0
    for cmp in comps:                                   # median 0 per component
        cc = cmp[connected[cmp]]
        if cc.size:
            s[cmp] -= np.median(s[cc])
    member = np.zeros(n_nodes, bool)                    # nodes of the components the dipole was fitted on
    if use_pat:
        for cmp in big:
            member[cmp] = True
    out = dict(s=s, e_s=np.sqrt(np.abs(np.diag(cov)[:n_nodes])), w=w, resid=(d - M @ x) / sig, A=A,
               I=np.array([e.i for e in edges], int), J=np.array([e.j for e in edges], int),
               pattern_term=(X @ x[n_nodes:]) if X.shape[1] else np.zeros(E),
               sigma_sys=ssys, labels=labels, pattern="dipole" if use_pat else "none",
               dipole=np.nan, dipole_pa=np.nan, e_dipole=np.nan, pattern_member=member)
    if use_pat:
        a_, b_ = x[n_nodes], x[n_nodes + 1]
        out.update(dipole=float(np.hypot(a_, b_)), dipole_pa=float(np.degrees(np.arctan2(b_, a_)) % 360.0),
                   e_dipole=float(np.sqrt(np.abs(cov[n_nodes, n_nodes] + cov[n_nodes + 1, n_nodes + 1]) / 2)))
    return out


def _cg(matvec, b, x0, dinv, tol=1e-9, maxit=1000):
    """Jacobi-preconditioned conjugate gradients (numpy only)."""
    x = x0.copy()
    r = b - matvec(x)
    z = dinv * r
    p = z.copy()
    rz = float(r @ z)
    bn = float(np.sqrt(b @ b)) or 1.0
    for _ in range(maxit):
        if np.sqrt(float(r @ r)) <= tol * bn:
            break
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x += a * p
        r -= a * Ap
        z = dinv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _solve_sparse(n_nodes: int, edges: Sequence[Edge], loss: str = "tukey", c: Optional[float] = None,
                  sigma_sys="auto", sys_floor: float = 0.005, lad_floor: float = 0.03, nit: int = 100,
                  centers: Optional[np.ndarray] = None, pa: Optional[np.ndarray] = None, pattern: str = "none",
                  pattern_min_nodes: int = 30, pattern_min_edges: int = 50, lad_nit: int = 60):
    """
    Sparse version of solve() for large graphs (v1.15), same model and outputs (A = None):
    weighted graph-Laplacian systems solved with Jacobi-preconditioned CG (edge lists, bincount),
    warm-started between the IRLS iterations (L1 start, then Tukey / Huber). The IFU dipole is
    fitted by alternation: s for the current dipole, the linear trend of s removed on the large
    components (the "no linear trend" gauge of the dense solver: dipole and ramp are degenerate),
    then the dipole from the remaining edge residuals. e_s = 1 / sqrt(sum of the node's edge
    weights) (its own information; the dense solver uses the diagonal of the inverse normal matrix,
    larger by the uncertainty of the neighbours).
    """
    E = len(edges)
    I = np.array([e.i for e in edges], int)
    J = np.array([e.j for e in edges], int)
    d = np.array([e.d for e in edges], float)
    sig_e = np.array([e.sigma for e in edges], float)
    labels = _components(n_nodes, edges)
    comps = [np.where(labels == l_)[0] for l_ in np.unique(labels)]
    big = [cmp for cmp in comps if len(cmp) >= pattern_min_nodes]
    use_pat = (centers is not None and E > 0 and
               (pattern == "dipole" or (pattern == "auto" and E >= pattern_min_edges and big)))
    if use_pat:
        th = edge_directions(edges, np.asarray(centers, float), None if pa is None else np.asarray(pa, float))
        X = np.c_[np.cos(th), np.sin(th)]
        trend = []
        for cmp in big:
            x, y = _tangent(np.asarray(centers, float), cmp)
            B = np.c_[np.ones(cmp.size), x, y]
            trend.append((cmp, B, np.linalg.pinv(B)))
    else:
        X = np.zeros((E, 0))
        trend = []
    c = c if c is not None else (4.685 if loss == "tukey" else 1.345)

    def lap(w):
        dg = np.bincount(I, w, n_nodes) + np.bincount(J, w, n_nodes)
        eps = 1e-9 * (float(dg.max()) if dg.size and dg.max() > 0 else 1.0)
        dinv = 1.0 / (dg + eps)

        def mv(x):
            t = w * (x[I] - x[J])
            return np.bincount(I, t, n_nodes) - np.bincount(J, t, n_nodes) + eps * x
        return mv, dinv

    def solve_w(w, s0, p0):
        mv, dinv = lap(w)
        p = p0.copy()
        s = s0.copy()
        for _ in range(8 if use_pat else 1):
            dd = d - (X @ p if use_pat else 0.0)
            b = np.bincount(I, w * dd, n_nodes) - np.bincount(J, w * dd, n_nodes)
            s = _cg(mv, b, s, dinv)
            if not use_pat:
                break
            for cmp, B, Bp in trend:          # no linear trend of s across the large components
                s[cmp] -= B @ (Bp @ s[cmp])
            r = d - (s[I] - s[J])
            Nx = X.T @ (w[:, None] * X)
            p_new = np.linalg.solve(Nx + 1e-12 * np.eye(2), X.T @ (w * r))
            if np.max(np.abs(p_new - p)) < 1e-6:
                p = p_new
                break
            p = p_new
        return s, p

    def resid(s, p):
        return d - (s[I] - s[J]) - (X @ p if use_pat else 0.0)

    def nmad(r):
        return 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else 0.0

    wt = 1.0 / np.sqrt(sig_e ** 2 + lad_floor ** 2)
    w1 = wt.copy()
    s = np.zeros(n_nodes)
    p = np.zeros(X.shape[1])
    for _ in range(lad_nit):
        s_new, p = solve_w(w1, s, p)
        r = np.abs(resid(s_new, p))
        w1 = wt / np.maximum(r, 1e-5)
        done = np.max(np.abs(s_new - s)) < 1e-6
        s = s_new
        if done:
            break
    raw = resid(s, p)
    ssys = max(nmad(raw), sys_floor) if sigma_sys == "auto" else float(sigma_sys)
    sig = np.sqrt(sig_e ** 2 + ssys ** 2)

    def weights(raw, sig):
        u = np.abs(raw / sig) / c
        if loss == "tukey":
            return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        return np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))

    w = weights(raw, sig)
    for _ in range(nit):
        s_new, p = solve_w(w / sig ** 2, s, p)
        raw = resid(s_new, p)
        if sigma_sys == "auto":
            ssys = max(nmad(raw), sys_floor)
        sig = np.sqrt(sig_e ** 2 + ssys ** 2)
        w_new = weights(raw, sig)
        done = np.allclose(s_new, s, atol=1e-6) and np.allclose(w_new, w, atol=1e-4)
        s, w = s_new, w_new
        if done:
            break
    info = np.bincount(I, w / sig ** 2, n_nodes) + np.bincount(J, w / sig ** 2, n_nodes)
    # x1.15: matches the dense errors (diagonal of the inverse normal matrix) on the Gum mosaic
    e_s = np.where(info > 0, 1.15 / np.sqrt(np.maximum(info, 1e-30)), np.nan)
    connected = (np.bincount(I, (w > 0.5).astype(float), n_nodes) + np.bincount(J, (w > 0.5).astype(float), n_nodes)) > 0
    for cmp in comps:
        cc = cmp[connected[cmp]]
        if cc.size:
            s[cmp] -= np.median(s[cc])
    member = np.zeros(n_nodes, bool)
    if use_pat:
        for cmp in big:
            member[cmp] = True
    out = dict(s=s, e_s=e_s, w=w, resid=resid(s, p) / sig, A=None, I=I, J=J,
               pattern_term=(X @ p) if use_pat else np.zeros(E),
               sigma_sys=ssys, labels=labels, pattern="dipole" if use_pat else "none",
               dipole=np.nan, dipole_pa=np.nan, e_dipole=np.nan, pattern_member=member)
    if use_pat:
        Nx = X.T @ ((w / sig ** 2)[:, None] * X)
        cov = np.linalg.pinv(Nx)
        out.update(dipole=float(np.hypot(*p)), dipole_pa=float(np.degrees(np.arctan2(p[1], p[0])) % 360.0),
                   e_dipole=float(np.sqrt(abs(cov[0, 0] + cov[1, 1]) / 2)))
    return out


def local_offsets(centers: np.ndarray, s: np.ndarray, e_s: np.ndarray, ok: np.ndarray,
                  radius: float = 0.97, min_ref: int = 3, group: Optional[np.ndarray] = None,
                  grow: Sequence[float] = (1.0, 1.5, 2.5), ref_ok: Optional[np.ndarray] = None,
                  return_scatter: bool = False):
    """Offset of every node relative to its surroundings: l_i = s_i - median(s_j) over the other
    connected nodes j whose centres lie within ``radius`` deg (default ~2 tile rings). Large-scale
    structure of s (residual drifts of a big mosaic) cancels, single tiles and small groups of
    offset tiles remain. v1.15: with ``group`` (night, MJD) only nodes of OTHER nights are used
    as reference (a night with a common calibration offset is compared with independent data,
    instead of with itself), and the radius grows (x ``grow``) until >= min_ref references are
    found. With fewer than min_ref reference nodes l_i = s_i (small graphs). ``ref_ok``: nodes
    allowed as reference (default ``ok``; the second iteration excludes the MAJOR nodes).
    Returns (l, e_l, n_ref, radius_used)."""
    refmask = np.asarray(ok if ref_ok is None else ref_ok, bool)
    centers = np.asarray(centers, float)
    n = len(s)
    loc, e_loc, n_ref = np.array(s, float), np.array(e_s, float), np.zeros(n, int)
    rad = np.full(n, np.nan)
    scat = np.full(n, np.nan)
    for k in range(n):
        if not ok[k]:
            continue
        c0 = np.cos(np.deg2rad(centers[k, 1]))
        dra = (centers[:, 0] - centers[k, 0] + 180.0) % 360.0 - 180.0
        dist = np.hypot(dra * c0, centers[:, 1] - centers[k, 1])
        base = refmask.copy()
        base[k] = False
        if group is not None and group[k] >= 0:
            base &= np.asarray(group) != group[k]
        for f in grow:
            ref = base & (dist <= radius * f)
            if ref.sum() >= min_ref:
                break
        n_ref[k] = int(ref.sum())
        if n_ref[k] >= min_ref:
            sr = s[ref]
            med = float(np.median(sr))
            scat[k] = 1.4826 * float(np.median(np.abs(sr - med)))
            e_ref = 1.2533 * scat[k] / np.sqrt(sr.size)
            loc[k] = s[k] - med
            e_loc[k] = float(np.hypot(e_s[k], e_ref))
            rad[k] = radius * f
    if return_scatter:
        return loc, e_loc, n_ref, rad, scat
    return loc, e_loc, n_ref, rad


def additive_sigma(loc: np.ndarray, fmed: np.ndarray, ok: np.ndarray, nbins: int = 6, min_nodes: int = 100) -> float:
    """
    v1.15: run-level additive Halpha uncertainty sigma_a (flux units per fiber) from the brightness
    dependence of the local offsets: NMAD(l)^2 ~ sigma_m^2 + (sigma_a / (ln10 F))^2. sigma_m = NMAD
    of l in the brightest quartile; sigma_a = median over quantile bins of the fainter half of
    sqrt(NMAD(l)^2 - sigma_m^2) ln10 median(F). 0 with < min_nodes usable nodes.
    MW run (DRP 1.2.1): 75 (55 for dark time only; dark-time fit: sigma_m 0.011 dex, sigma_a 56).
    """
    def nmad(x):
        return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if x.size else 0.0
    m = ok & np.isfinite(loc) & np.isfinite(fmed) & (fmed > 0)
    if m.sum() < min_nodes:
        return 0.0
    lm, Fm = loc[m], fmed[m]
    sm = nmad(lm[Fm >= np.quantile(Fm, 0.75)])
    fa = Fm < np.median(Fm)
    qs = np.quantile(Fm[fa], np.linspace(0, 1, nbins + 1))
    a = []
    for x0, x1 in zip(qs[:-1], qs[1:]):
        mm = fa & (Fm >= x0) & (Fm <= x1)
        if mm.sum() >= 20:
            a.append(np.sqrt(max(nmad(lm[mm]) ** 2 - sm ** 2, 0.0)) * np.log(10) * float(np.median(Fm[mm])))
    return float(np.median(a)) if a else 0.0


def night_offsets(group: np.ndarray, loc: np.ndarray, ok: np.ndarray):
    """Median local offset of all (connected) nodes of the same night and their number."""
    group = np.asarray(group)
    out, cnt = np.full(len(loc), np.nan), np.zeros(len(loc), int)
    for g in np.unique(group[ok]):
        if g < 0:
            continue
        m = ok & (group == g)
        out[group == g] = float(np.median(loc[m]))
        cnt[group == g] = int(m.sum())
    return out, cnt


def decompose(nodes: Sequence[Node], edges: Sequence[Edge], s: np.ndarray, w: np.ndarray,
              pattern_term: Optional[np.ndarray] = None, min_snr: float = 3.0, min_pairs: int = 40, max_bins: int = 8, per_bin: int = 15,
              range_min: float = 1.25):
    """
    v1.15: multiplicative vs additive part of every node's offset. All accepted overlaps of node i
    are pooled; the fluxes of the neighbours are put on the common scale (F_j 10^-s_j); the pairs
    (summed S/N > min_snr) are binned in quantiles of their mean flux (4-8 bins, >= 15 pairs each)
    and F_i = g F_ref + c is fitted to the bin medians (error from the bin residuals; needs a
    brightness range (bin medians) of >= range_min = 1.25x). A pure flux-scale error gives log10 g = s_i, c = 0; an additive error (e.g.
    sky-subtraction residual of Halpha: geocoronal emission, scattered moonlight with solar Halpha
    absorption) gives g ~ 1 and c != 0, and an apparent offset that grows towards faint fields.
    The brightness range of the pooled pairs separates the two (otherwise large errors).
    Returns dict of arrays: mult_dex, e_mult_dex, add, e_add (flux units of the maps), f_med
    (median F_i of the pairs), n_pairs.
    """
    n = len(nodes)
    out = {k: np.full(n, np.nan) for k in ("mult_dex", "e_mult_dex", "add", "e_add", "f_med")}
    out["n_pairs"] = np.zeros(n, int)
    per = [[] for _ in range(n)]
    for k, e in enumerate(edges):
        if e.ia is None or not w[k] > 0.5:
            continue
        pt = float(pattern_term[k]) if pattern_term is not None else 0.0   # IFU dipole term of the edge
        per[e.i].append((e.ia, e.j, e.ib, pt))
        per[e.j].append((e.ib, e.i, e.ia, -pt))
    for i in range(n):
        if not per[i]:
            continue
        Fi = np.concatenate([nodes[i].flux[a] for a, _, _, _ in per[i]])
        Ei = np.concatenate([nodes[i].eflux[a] for a, _, _, _ in per[i]])
        Fr = np.concatenate([nodes[j].flux[b] * 10 ** (pt - s[j]) for _, j, b, pt in per[i]])
        Er = np.concatenate([nodes[j].eflux[b] * 10 ** (pt - s[j]) for _, j, b, pt in per[i]])
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = np.isfinite(Fi) & np.isfinite(Fr) & (Ei > 0) & (Er > 0) & ((Fi + Fr) / np.hypot(Ei, Er) > min_snr)
        if ok.sum() < min_pairs:
            continue
        Fi, Fr = Fi[ok], Fr[ok]
        # symmetric brightness bins (quantiles of the pair mean): medians of F_i and F_ref per bin are
        # nearly noise-free, and selecting on the mean does not bias either side (a regression of the
        # noisy pairs would be diluted: g biased low, c high)
        mm = 0.5 * (Fi + Fr)
        nb = int(min(max_bins, Fi.size // per_bin))
        if nb < 4:
            continue
        q = np.quantile(mm, np.linspace(0, 1, nb + 1))
        b = np.clip(np.searchsorted(q, mm, side="right") - 1, 0, nb - 1)
        xi = np.array([np.median(Fi[b == k]) for k in range(nb)])
        xr = np.array([np.median(Fr[b == k]) for k in range(nb)])
        if not (xr.max() > range_min * max(xr.min(), 1e-30) and xr.max() > 0):
            continue                        # too little brightness range: g and c degenerate
        X = np.c_[xr, np.ones(nb)]
        coef, *_ = np.linalg.lstsq(X, xi, rcond=None)
        g, c = float(coef[0]), float(coef[1])
        res = xi - X @ coef
        s2 = max(float(res @ res) / (nb - 2), (0.005 * float(np.median(np.abs(xi)))) ** 2)
        cov = np.linalg.inv(X.T @ X) * s2
        if not g > 0:
            continue
        out["mult_dex"][i] = np.log10(g)
        out["e_mult_dex"][i] = np.sqrt(abs(cov[0, 0])) / (g * np.log(10))
        out["add"][i], out["e_add"][i] = c, np.sqrt(abs(cov[1, 1]))
        out["f_med"][i] = float(np.median(Fi))
        out["n_pairs"][i] = int(Fi.size)
    return out


def classify_node(local: float, e_local: float, n_edges: int, n_rejected: int,
                  warn_dex: float = 0.04, major_dex: float = 0.10, nsig: float = 3.0,
                  mult_rel: float = np.nan, e_mult_rel: float = np.nan, add: float = np.nan,
                  e_add: float = np.nan, soften_additive: bool = True, with_cause: bool = False,
                  soften_max_dex: float = 0.2):
    """Flag label of one node from its local offset (shared by the metric and lvmqa.recalc).
    v1.15: if the offset is explained by an ADDITIVE term (decompose(): |add| > nsig e_add, same
    sign as the local offset) while the flux scale relative to the local reference (mult_rel =
    mult_dex - (s - l)) is consistent (|mult_rel| <= warn_dex or < nsig sigma), the flag is capped
    at WARNING (cause 'additive': sky-subtraction-like Halpha residual, not a flux-calibration
    error), unless |l| > soften_max_dex (0.2 dex: such exposures are unusable for Halpha anyway). with_cause=True returns (label, cause) with cause in ('', 'scale', 'additive', 'edges')."""
    def ret(lab, cause):
        return (lab, cause) if with_cause else lab
    if n_edges == 0:
        return ret("UNDEFINED", "")
    if n_rejected == n_edges:
        return ret("WARNING", "edges")
    sig = abs(local) > nsig * e_local
    lab = "GOOD"
    if abs(local) > major_dex and sig:
        lab = "MAJOR ISSUE"
    elif abs(local) > warn_dex and sig:
        lab = "WARNING"
    if lab == "GOOD":
        return ret("WARNING", "edges") if n_rejected > 0.5 * n_edges else ret("GOOD", "")
    cause = "scale"
    if (soften_additive and abs(local) <= soften_max_dex and np.isfinite(add) and np.isfinite(e_add) and abs(add) > nsig * e_add
            and np.sign(add) == np.sign(local) and np.isfinite(mult_rel)
            and (abs(mult_rel) <= warn_dex or abs(mult_rel) < nsig * (e_mult_rel if np.isfinite(e_mult_rel) else 0))):
        cause = "additive"
        lab = "WARNING"
    return ret(lab, cause)


def fluxmap(nodes: Sequence[Node], warn_dex: float = 0.04, major_dex: float = 0.10, nsig: float = 3.0,
            local_radius: float = 0.97, local_min: int = 3, **kw) -> FluxMapResult:
    ek = {k: kw.pop(k) for k in list(kw) if k in ("match_radius", "min_snr", "max_sep_deg", "min_match")}
    edges = build_edges(nodes, **ek)
    centers = np.array([n.center for n in nodes])
    sol = solve(len(nodes), edges, centers=centers, pa=np.array([n.pa for n in nodes]), **kw)
    w, I, J = sol["w"], sol["I"], sol["J"]
    n_edges = (np.bincount(I, minlength=len(nodes)) + np.bincount(J, minlength=len(nodes))).astype(int)
    rj = (w < 0.5).astype(float)
    n_rej = (np.bincount(I, rj, len(nodes)) + np.bincount(J, rj, len(nodes))).astype(int)
    loc, e_loc, n_ref, _ = local_offsets(centers, sol["s"], sol["e_s"], n_edges > n_rej, local_radius, local_min)
    flags = [classify_node(loc[i], e_loc[i], n_edges[i], n_rej[i], warn_dex, major_dex, nsig)
             for i in range(len(nodes))]
    res = FluxMapResult([n.name for n in nodes], sol["s"], sol["e_s"], edges, w, sol["resid"],
                        n_edges, n_rej, flags)
    res.sigma_sys = sol["sigma_sys"]
    res.local, res.e_local, res.n_ref = loc, e_loc, n_ref
    res.pattern, res.dipole, res.dipole_pa = sol["pattern"], sol["dipole"], sol["dipole_pa"]
    return res


def nodes_from_dap(files: Dict[str, str], line: str = "Halpha_6562.85") -> List[Node]:
    """Nodes from DAP files {name: path} (NP flux of ``line``, PT positions)."""
    from astropy.table import Table, join
    out = []
    for name, f in files.items():
        pt = Table.read(f, "PT")
        ext = "NP_ELINES_B" if float(line.rsplit("_", 1)[1]) < 5800 else (
            "NP_ELINES_R" if float(line.rsplit("_", 1)[1]) < 7600 else "NP_ELINES_I")
        t = join(pt, Table.read(f, ext)["id", f"flux_{line}", f"e_flux_{line}"], keys="id")
        out.append(Node(str(name), np.asarray(t["ra"], float), np.asarray(t["dec"], float),
                        np.asarray(t[f"flux_{line}"], float), np.asarray(t[f"e_flux_{line}"], float)))
    return out


def neighbourhood(nodes: Sequence[Node], target: str, rings: int = 2, spacing_deg: float = 0.42) -> List[Node]:
    """Nodes within ``rings`` tile rings around ``target`` (centre distance <= (rings+0.3) x spacing)."""
    t = [n for n in nodes if n.name == target][0]
    r0, d0 = t.center
    keep = []
    for n in nodes:
        r, d = n.center
        if np.hypot((r - r0) * np.cos(np.deg2rad(d0)), d - d0) <= (rings + 0.3) * spacing_deg:
            keep.append(n)
    return keep
