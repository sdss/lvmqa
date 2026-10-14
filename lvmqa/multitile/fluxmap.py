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


def build_edges(nodes: Sequence[Node], match_radius: float = 20.0, min_snr: float = 5.0,
                max_sep_deg: float = 0.6, min_match: int = 10) -> List[Edge]:
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
        dist, idx = _nearest(A[ga], B[gb])
        ia = np.where(ga)[0]
        ib = np.where(gb)[0][idx]
        m = dist < match_radius
        ia, ib = ia[m], ib[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            fa, fb = nodes[a].flux[ia], nodes[b].flux[ib]
            ok = (fa / nodes[a].eflux[ia] > min_snr) & (fb / nodes[b].eflux[ib] > min_snr) & (fa > 0) & (fb > 0)
        if ok.sum() < min_match:
            continue
        x = np.log10(fa[ok] / fb[ok])
        mu, s = _huber_mean(x)
        edges.append(Edge(a, b, float(mu), float(1.2533 * s / np.sqrt(ok.sum())), int(ok.sum())))
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
          pattern_min_nodes: int = 30, pattern_min_edges: int = 50) -> Dict[str, np.ndarray]:
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
    frame: direction of the brighter IFU side), labels (components)).
    """
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
               sigma_sys=ssys, labels=labels, pattern="dipole" if use_pat else "none",
               dipole=np.nan, dipole_pa=np.nan, e_dipole=np.nan, pattern_member=member)
    if use_pat:
        a_, b_ = x[n_nodes], x[n_nodes + 1]
        out.update(dipole=float(np.hypot(a_, b_)), dipole_pa=float(np.degrees(np.arctan2(b_, a_)) % 360.0),
                   e_dipole=float(np.sqrt(np.abs(cov[n_nodes, n_nodes] + cov[n_nodes + 1, n_nodes + 1]) / 2)))
    return out


def local_offsets(centers: np.ndarray, s: np.ndarray, e_s: np.ndarray, ok: np.ndarray,
                  radius: float = 0.97, min_ref: int = 3):
    """Offset of every node relative to its surroundings: l_i = s_i - median(s_j) over the other
    connected nodes j whose centres lie within ``radius`` deg (default ~2 tile rings). Large-scale
    structure of s (residual drifts of a big mosaic) cancels, single tiles and small groups of
    offset tiles remain. With fewer than min_ref reference nodes l_i = s_i (small graphs).
    Returns (l, e_l, n_ref)."""
    centers = np.asarray(centers, float)
    n = len(s)
    loc, e_loc, n_ref = np.array(s, float), np.array(e_s, float), np.zeros(n, int)
    for k in range(n):
        if not ok[k]:
            continue
        c0 = np.cos(np.deg2rad(centers[k, 1]))
        dra = (centers[:, 0] - centers[k, 0] + 180.0) % 360.0 - 180.0
        dist = np.hypot(dra * c0, centers[:, 1] - centers[k, 1])
        ref = ok & (dist <= radius)
        ref[k] = False
        n_ref[k] = int(ref.sum())
        if n_ref[k] >= min_ref:
            sr = s[ref]
            med = float(np.median(sr))
            e_ref = 1.2533 * 1.4826 * float(np.median(np.abs(sr - med))) / np.sqrt(sr.size)
            loc[k] = s[k] - med
            e_loc[k] = float(np.hypot(e_s[k], e_ref))
    return loc, e_loc, n_ref


def classify_node(local: float, e_local: float, n_edges: int, n_rejected: int,
                  warn_dex: float = 0.04, major_dex: float = 0.10, nsig: float = 3.0) -> str:
    """Flag label of one node from its local offset (shared by the metric and lvmqa.recalc)."""
    if n_edges == 0:
        return "UNDEFINED"
    if n_rejected == n_edges:
        return "WARNING"
    sig = abs(local) > nsig * e_local
    if abs(local) > major_dex and sig:
        return "MAJOR ISSUE"
    if (abs(local) > warn_dex and sig) or n_rejected > 0.5 * n_edges:
        return "WARNING"
    return "GOOD"


def fluxmap(nodes: Sequence[Node], warn_dex: float = 0.04, major_dex: float = 0.10, nsig: float = 3.0,
            local_radius: float = 0.97, local_min: int = 3, **kw) -> FluxMapResult:
    ek = {k: kw.pop(k) for k in list(kw) if k in ("match_radius", "min_snr", "max_sep_deg", "min_match")}
    edges = build_edges(nodes, **ek)
    centers = np.array([n.center for n in nodes])
    sol = solve(len(nodes), edges, centers=centers, pa=np.array([n.pa for n in nodes]), **kw)
    A, w = sol["A"], sol["w"]
    n_edges = np.abs(A).sum(axis=0).astype(int)
    n_rej = (np.abs(A) * (w[:, None] < 0.5)).sum(axis=0).astype(int)
    loc, e_loc, n_ref = local_offsets(centers, sol["s"], sol["e_s"], n_edges > n_rej, local_radius, local_min)
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
