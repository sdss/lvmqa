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
    for a, b in combinations(range(len(nodes)), 2):
        (ra1, de1), (ra2, de2) = nodes[a].center, nodes[b].center
        sep = np.hypot((ra1 - ra2) * np.cos(np.deg2rad(de1)), de1 - de2)
        if sep > max_sep_deg:
            continue
        c = np.cos(np.deg2rad(0.5 * (de1 + de2)))
        A = np.c_[nodes[a].ra * c * 3600.0, nodes[a].dec * 3600.0]
        B = np.c_[nodes[b].ra * c * 3600.0, nodes[b].dec * 3600.0]
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


def solve(n_nodes: int, edges: Sequence[Edge], loss: str = "tukey", c: Optional[float] = None,
          sigma_sys="auto", sys_floor: float = 0.005, lad_floor: float = 0.03,
          nit: int = 100) -> Dict[str, np.ndarray]:
    """
    IRLS solution of d_ij = s_i - s_j with a robust loss; gauge: median(s) = 0.
    sigma_sys='auto': the extra per-edge scatter (fiber-position mismatch, PSF, Halpha
    gradients) is re-estimated in every iteration as the NMAD of the raw edge residuals of the
    accepted edges (floor ``sys_floor`` dex) - the formal edge errors (~0.002 dex) are far too
    small to standardise the residuals.
    """
    E = len(edges)
    A = np.zeros((E, n_nodes))
    d = np.array([e.d for e in edges])
    sig_e = np.array([e.sigma for e in edges])
    for k, e in enumerate(edges):
        A[k, e.i], A[k, e.j] = 1.0, -1.0
    c = c if c is not None else (4.685 if loss == "tukey" else 1.345)
    ssys = sys_floor if sigma_sys == "auto" else float(sigma_sys)
    # robust start: weighted least-absolute-deviations (L1) solution of the graph (IRLS). L1 lets a minority of edges absorb the inconsistencies, so a corrupted edge
    # is out-voted by the alternative paths (cycles) instead of pulling its nodes; starting
    # the IRLS from least squares can converge to the wrong local minimum.
    # nearly uniform L1 weights: the formal edge errors do not reflect the real edge scatter,
    # and a well-measured corrupted edge must not out-weigh two independent paths
    s = _lad(A, d, 1.0 / np.sqrt(sig_e ** 2 + lad_floor ** 2))
    raw = d - A @ s
    if sigma_sys == "auto":
        ssys = max(1.4826 * float(np.median(np.abs(raw))), sys_floor)
    sig = np.sqrt(sig_e ** 2 + ssys ** 2)
    u = np.abs(raw / sig) / c
    w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0) if loss == "tukey" else np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
    for it in range(nit):
        sig = np.sqrt(sig_e ** 2 + ssys ** 2)
        W = w / sig ** 2
        N = A.T @ (W[:, None] * A) + 1e-9 * np.eye(n_nodes)
        s_new = np.linalg.solve(N, A.T @ (W * d))
        raw = d - A @ s_new
        if sigma_sys == "auto":
            acc = w > 0.5 if (w > 0.5).sum() >= 3 else np.ones(E, bool)
            ssys = max(1.4826 * float(np.median(np.abs(raw[acc]))), sys_floor)
            sig = np.sqrt(sig_e ** 2 + ssys ** 2)
        u = np.abs(raw / sig) / c
        if loss == "tukey":
            w_new = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        else:
            w_new = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        done = np.allclose(s_new, s, atol=1e-6) and np.allclose(w_new, w, atol=1e-4)
        s, w = s_new, w_new
        if done:
            break
    sig = np.sqrt(sig_e ** 2 + ssys ** 2)
    W = w / sig ** 2
    N0 = A.T @ (W[:, None] * A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.linalg.pinv(N0)          # minimum-norm gauge (sum s = 0)
    connected = (np.abs(A).T @ (w > 0.5)) > 0
    s = s - np.median(s[connected]) if connected.any() else s
    return dict(s=s, e_s=np.sqrt(np.abs(np.diag(cov))), w=w, resid=(d - A @ s) / sig, A=A,
                sigma_sys=ssys)


def fluxmap(nodes: Sequence[Node], warn_dex: float = 0.04, major_dex: float = 0.10, nsig: float = 3.0,
            **kw) -> FluxMapResult:
    ek = {k: kw.pop(k) for k in list(kw) if k in ("match_radius", "min_snr", "max_sep_deg", "min_match")}
    edges = build_edges(nodes, **ek)
    sol = solve(len(nodes), edges, **kw)
    A, w = sol["A"], sol["w"]
    n_edges = np.abs(A).sum(axis=0).astype(int)
    n_rej = (np.abs(A) * (w[:, None] < 0.5)).sum(axis=0).astype(int)
    flags = []
    for i in range(len(nodes)):
        s, e = sol["s"][i], sol["e_s"][i]
        if n_edges[i] == 0 or n_edges[i] == n_rej[i]:
            flags.append("UNDEFINED" if n_edges[i] == 0 else "WARNING")
        elif abs(s) > major_dex and abs(s) > nsig * e:
            flags.append("MAJOR ISSUE")
        elif (abs(s) > warn_dex and abs(s) > nsig * e) or n_rej[i] > 0.5 * n_edges[i]:
            flags.append("WARNING")
        else:
            flags.append("GOOD")
    res = FluxMapResult([n.name for n in nodes], sol["s"], sol["e_s"], edges, w, sol["resid"],
                        n_edges, n_rej, flags)
    res.sigma_sys = sol["sigma_sys"]
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
