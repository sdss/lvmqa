"""
qa_guider - guiding / sky conditions from the guider co-add keywords in the SFrame header
(GDRCOADD *: written by the guider co-add of the science telescope).

Uses the guider warnings (transparency, FWHM, Gaia match), the astrometric solution flag and
the median FWHM of the guider frames. Component of qa_obs ("hard").

The PA-drift warning (WARNPADR) is not used: the resulting displacement at the edge of the
field is small compared to the 35.3 arcsec fibers. PADRIFT is kept as a diagnostic.
FWHM limits: guider FWHM is typically 2.7-5 arcsec (16-cm telescopes); a Gaussian PSF of even
10 arcsec FWHM keeps >99.9% of a point source within a 35.3 arcsec fiber, so the FWHM criterion
flags degraded conditions (focus, clouds, guiding) rather than aperture losses.
"""

from __future__ import annotations

import math

from ..bitmask import QAFlag
from ..registry import GROUP_OBS, Metric, ValueSpec, register

PREFIX = "GDRCOADD "


def _bool(v):
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return str(v).strip().upper() in ("T", "TRUE", "1")


def _float(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@register
class Guider(Metric):
    name = "qa_guider"
    slot = None          # component of qa_obs (not in the bitmask)
    order = 3
    obs_weight = "hard"
    group = GROUP_OBS
    title = "Guiding / transparency (guider co-add)"
    description = ("Obs. conditions from the guider co-add header keywords: transparency, FWHM "
                   "and Gaia-match warnings, astrometric solution, median FWHM")
    requires = ("sframe_header",)
    defaults = dict(
        fwhm_warn=6.0,              # median guider FWHM [arcsec] above -> WARNING
        fwhm_major=10.0,            # median guider FWHM [arcsec] above -> MAJOR
        transparency_major=False,   # True: poor-transparency warning -> MAJOR (default WARNING)
        fov_radius_arcsec=906.5,    # for the diagnostic displacement at the FoV edge
    )
    values = (
        ValueSpec("solved", "bool", "", "Guider co-add astrometrically solved"),
        ValueSpec("warn_tran", "bool", "", "Poor-transparency warning (WARNTRAN)"),
        ValueSpec("warn_fwhm", "bool", "", "Co-added FWHM significantly above median (WARNFWHM)"),
        ValueSpec("warn_match", "bool", "", "Co-add could not be matched with Gaia (WARNMATC)"),
        ValueSpec("fwhm_med", "f8", "arcsec", "Median FWHM of the guider frames (FWHMMED)"),
        ValueSpec("fwhm_first", "f8", "arcsec", "FWHM in the first guider frame (FWHM0)"),
        ValueSpec("fwhm_last", "f8", "arcsec", "FWHM in the last guider frame (FWHMN)"),
        ValueSpec("padrift", "f8", "deg", "PA drift during the exposure (PADRIFT)"),
        ValueSpec("pa_edge_shift", "f8", "arcsec", "Displacement at the FoV edge caused by the PA drift"),
        ValueSpec("zeropt", "f8", "mag", "Guider instrumental zero point (ZEROPT)"),
        ValueSpec("nframes", "i8", "", "Number of guider frames (NFRAMES)"),
    )
    main_value = "fwhm_med"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        tran = "MAJOR" if p["transparency_major"] else "WARNING"
        return {
            "GOOD": f"No guider warning and median FWHM <= {p['fwhm_warn']:g} arcsec",
            "WARNING": ("Guider warning: " + ("poor transparency, " if tran == "WARNING" else "")
                        + "co-added FWHM above median"
                        + f"; or median FWHM > {p['fwhm_warn']:g} arcsec"),
            "MAJOR ISSUE": ("Guider co-add not astrometrically solved or not matched with Gaia"
                            + (", or poor transparency" if tran == "MAJOR" else "")
                            + f"; or median FWHM > {p['fwhm_major']:g} arcsec"),
            "UNDEFINED": "SFrame missing or no guider co-add keywords",
        }

    def compute(self, ctx):
        p = self.params
        if not ctx.has_sframe:
            return self.result(QAFlag.UNDEFINED, note="SFrame missing")
        h = ctx.header
        g = lambda k: h.get(PREFIX + k)  # noqa: E731
        if not any((PREFIX + k) in h for k in ("SOLVED", "FWHMMED", "WARNTRAN")):
            return self.result(QAFlag.UNDEFINED, note="no guider keywords")
        solved, tran, fwhmw = _bool(g("SOLVED")), _bool(g("WARNTRAN")), _bool(g("WARNFWHM"))
        match = _bool(g("WARNMATC"))
        fwhm = _float(g("FWHMMED"))
        padrift = _float(g("PADRIFT"))
        nfr = _float(g("NFRAMES"))
        vals = dict(solved=bool(solved), warn_tran=bool(tran), warn_fwhm=bool(fwhmw),
                    warn_match=bool(match),
                    fwhm_med=fwhm if fwhm is not None else float("nan"),
                    fwhm_first=_float(g("FWHM0")) or float("nan"),
                    fwhm_last=_float(g("FWHMN")) or float("nan"),
                    padrift=padrift if padrift is not None else float("nan"),
                    pa_edge_shift=(math.radians(abs(padrift)) * p["fov_radius_arcsec"]
                                   if padrift is not None else float("nan")),
                    zeropt=_float(g("ZEROPT")) or float("nan"),
                    nframes=int(nfr) if nfr is not None else -1)
        major, warn = [], []
        if solved is False:
            major.append("not solved")
        if match:
            major.append("no Gaia match")
        if tran:
            (major if p["transparency_major"] else warn).append("poor transparency")
        if fwhm is not None and fwhm > p["fwhm_major"]:
            major.append(f"FWHM {fwhm:.1f}\"")
        elif fwhm is not None and fwhm > p["fwhm_warn"]:
            warn.append(f"FWHM {fwhm:.1f}\"")
        if fwhmw:
            warn.append("FWHM above median")
        if major:
            return self.result(QAFlag.MAJOR, note=", ".join(major + warn), **vals)
        if warn:
            return self.result(QAFlag.WARNING, note=", ".join(warn), **vals)
        return self.result(QAFlag.GOOD, **vals)
