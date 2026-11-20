"""
qa_brightstars - bright Hipparcos stars falling on the science IFU.

The star list comes from a local cache of the Hipparcos main catalogue (Vizier
I/239/hip_main, V < ``max_mag_cache``), downloaded once (see ``lvmqa.utils.hipparcos``).

Footprint: if the SFrame is available the actual science-fiber positions (SLITMAP ra/dec)
are used - a star is "in the field" if it lies within ``fiber_match_radius_arcsec`` of any
science fiber center. Otherwise a regular hexagon (outer diameter 1813") rotated by the
position angle is used (DR20 geometry, but with the 30 deg orientation offset corrected -
see ``hex_pa_offset_deg``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bitmask import QAFlag
from ..registry import GROUP_OBS, Metric, ValueSpec, register
from ..utils import hipparcos as hip


def _in_regular_hexagon(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    x = np.abs(x)
    y = np.abs(y)
    return (x <= r) & (y <= np.sqrt(3.0) / 2.0 * r) & (x + y / np.sqrt(3.0) <= r)


@register
class BrightStars(Metric):
    name = "qa_brightstars"
    slot = None          # component of qa_obs (not in the bitmask)
    order = 2
    obs_weight = "hard"
    group = GROUP_OBS
    title = "Bright stars in the field"
    description = "Obs. conditions: bright Hipparcos stars within the science IFU field of view"
    requires = ("drpall", "sframe_header")
    defaults = dict(
        mag_major=4.0,                 # V < mag_major in field -> MAJOR
        mag_warn=6.0,                  # mag_major <= V < mag_warn in field -> WARNING
        max_mag_cache=6.5,             # limiting magnitude of the cached catalogue
        fov_outer_diameter_arcsec=1813.0,
        fiber_match_radius_arcsec=37.0,  # ~ fiber pitch: star counts as 'on the IFU'
        # orientation of the fallback hexagon w.r.t. the DR20 convention: +30 deg makes the
        # hexagon coincide with the SLITMAP fiber footprint (checked on 1.3.2 SFrames);
        # DR20 (qa_find_bright_stars.py) used 0, i.e. a hexagon rotated by 30 deg
        hex_pa_offset_deg=30.0,
        near_margin_arcsec=900.0,      # report stars up to this distance outside the FoV
        catalog_path=None,             # set by prepare() (default: <cache_dir>/hipparcos_*.fits)
        allow_download=True,
    )
    values = (
        ValueSpec("n_major", "i8", "", "Stars in the field with V < mag_major"),
        ValueSpec("n_warn", "i8", "", "Stars in the field with mag_major <= V < mag_warn"),
        ValueSpec("brightest_vmag", "f8", "mag", "V of the brightest catalogued star in the field"),
        ValueSpec("brightest_hip", "i8", "", "HIP id of the brightest star in the field"),
        ValueSpec("n_near", "i8", "", "Stars with V < mag_warn just outside the field (within near_margin)"),
        ValueSpec("brightest_near_vmag", "f8", "mag", "V of the brightest star just outside the field"),
        ValueSpec("geometry", "U8", "", "Footprint used: 'fibers' (SLITMAP) or 'hexagon'"),
    )
    main_value = "brightest_vmag"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": f"No Hipparcos star with V < {p['mag_warn']:g} in the field",
            "WARNING": f"Star with {p['mag_major']:g} <= V < {p['mag_warn']:g} in the field",
            "MAJOR ISSUE": f"Star with V < {p['mag_major']:g} in the field",
            "UNDEFINED": "Hipparcos cache unavailable or pointing unknown",
        }

    def prepare(self, cfg) -> None:
        p = self.params
        path = Path(p["catalog_path"]) if p["catalog_path"] else \
            hip.default_cache_path(cfg.cache_dir, p["max_mag_cache"])
        ok = hip.ensure_catalog(path, p["max_mag_cache"], allow_download=p["allow_download"])
        p["catalog_path"] = str(ok) if ok else ""

    def compute(self, ctx):
        p = self.params
        if not p["catalog_path"]:
            return self.result(QAFlag.UNDEFINED, note="no Hipparcos cache")
        cat = hip.load_catalog(Path(p["catalog_path"]))

        ra0 = ctx.value("sci_ra", ("SCIRA", "POSCIRA"))
        dec0 = ctx.value("sci_dec", ("SCIDEC", "POSCIDE"))
        pa = ctx.value("sci_pa", ("SCIPA", "POSCIPA"), 0.0)
        if ra0 is None or dec0 is None:
            return self.result(QAFlag.UNDEFINED, note="pointing unknown")
        ra0, dec0, pa = float(ra0), float(dec0), float(pa)

        r_out = 0.5 * p["fov_outer_diameter_arcsec"]
        ra_c = np.asarray(cat["RAICRS"], dtype=float)
        de_c = np.asarray(cat["DEICRS"], dtype=float)
        vmag = np.asarray(cat["Vmag"], dtype=float)
        hipid = np.asarray(cat["HIP"], dtype=np.int64)
        sep = hip.angsep_arcsec(ra0, dec0, ra_c, de_c)
        cand = np.where(sep < r_out + p["near_margin_arcsec"])[0]

        inside = np.zeros(len(cand), dtype=bool)
        geometry = "hexagon"
        if len(cand):
            fib_ok = False
            if ctx.has_sframe:
                sm = ctx.slitmap[ctx.sci_fiber_index]
                fra = np.asarray(sm["ra"], dtype=float)
                fde = np.asarray(sm["dec"], dtype=float)
                good = np.isfinite(fra) & np.isfinite(fde) & ~((fra == 0) & (fde == 0))
                if good.sum() > 100:
                    fib_ok = True
                    geometry = "fibers"
                    fv = hip.unit_vectors(fra[good], fde[good])
                    sv = hip.unit_vectors(ra_c[cand], de_c[cand])
                    cosd = np.clip(sv @ fv.T, -1, 1).max(axis=1)
                    dmin = np.rad2deg(np.arccos(cosd)) * 3600.0
                    inside = dmin <= p["fiber_match_radius_arcsec"]
            if not fib_ok:
                dra = (ra_c[cand] - ra0 + 180.0) % 360.0 - 180.0
                x = dra * np.cos(np.deg2rad(dec0)) * 3600.0
                y = (de_c[cand] - dec0) * 3600.0
                th = np.deg2rad(-(pa + p["hex_pa_offset_deg"]))
                xr = np.cos(th) * x - np.sin(th) * y
                yr = np.sin(th) * x + np.cos(th) * y
                inside = _in_regular_hexagon(xr, yr, r_out)
        else:
            geometry = "fibers" if ctx.has_sframe else "hexagon"

        v_in = vmag[cand][inside]
        h_in = hipid[cand][inside]
        v_near = vmag[cand][~inside]
        n_major = int(np.sum(v_in < p["mag_major"]))
        n_warn = int(np.sum((v_in >= p["mag_major"]) & (v_in < p["mag_warn"])))
        vals = dict(
            n_major=n_major, n_warn=n_warn,
            brightest_vmag=float(v_in.min()) if v_in.size else np.nan,
            brightest_hip=int(h_in[np.argmin(v_in)]) if v_in.size else -1,
            n_near=int(np.sum(v_near < p["mag_warn"])),
            brightest_near_vmag=float(v_near.min()) if v_near.size else np.nan,
            geometry=geometry,
        )
        if n_major:
            return self.result(QAFlag.MAJOR, note=f"V={vals['brightest_vmag']:.2f} star in field", **vals)
        if n_warn:
            return self.result(QAFlag.WARNING, note=f"V={vals['brightest_vmag']:.2f} star in field", **vals)
        return self.result(QAFlag.GOOD, **vals)
