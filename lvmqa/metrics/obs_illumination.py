"""
qa_illumination - sky illumination by twilight and the Moon (DR20 ``qa_sky_brt`` logic).
"""

from __future__ import annotations

from ..bitmask import QAFlag
from ..registry import GROUP_OBS, Metric, ValueSpec, register


@register
class Illumination(Metric):
    name = "qa_illumination"
    slot = None          # component of qa_obs (not in the bitmask)
    order = 1
    obs_weight = "soft"  # MAJOR -> qa_obs WARNING, WARNING -> no effect
    group = GROUP_OBS
    title = "Sky illumination (twilight / Moon)"
    description = ("Obs. conditions: proximity to bright Moon or twilight "
                   "(same rules as DR20 qa_sky_brt)")
    requires = ("drpall", "sframe_header")
    defaults = dict(
        twilight_sun_alt=-18.0,   # sun above this altitude [deg] -> twilight
        moon_alt_min=0.0,         # moon below this altitude is ignored [deg]
        moon_sep_min=90.0,        # moon farther than this from the field is ignored [deg]
        fli_warn=0.5,             # FLI >= this -> WARNING (moon up & close)
        fli_major=0.8,            # FLI >= this -> MAJOR (moon up & close)
    )
    values = (
        ValueSpec("sun_alt", "f8", "deg", "Sun altitude"),
        ValueSpec("moon_alt", "f8", "deg", "Moon altitude"),
        ValueSpec("moon_fli", "f8", "", "Moon fractional illumination"),
        ValueSpec("moon_sep", "f8", "deg", "Separation between science field and Moon"),
    )
    main_value = "moon_fli"

    @property
    def criteria(self):  # type: ignore[override]
        p = self.params
        return {
            "GOOD": (f"Sun alt < {p['twilight_sun_alt']:g} and (Moon below horizon, or "
                     f"separation >= {p['moon_sep_min']:g} deg, or FLI < {p['fli_warn']:g})"),
            "WARNING": (f"Moon above horizon, separation < {p['moon_sep_min']:g} deg and "
                        f"{p['fli_warn']:g} <= FLI < {p['fli_major']:g}"),
            "MAJOR ISSUE": (f"Sun alt > {p['twilight_sun_alt']:g} (twilight), or Moon above "
                            f"horizon, separation < {p['moon_sep_min']:g} deg and "
                            f"FLI >= {p['fli_major']:g}"),
            "UNDEFINED": "Sun / Moon parameters not available in drpall or SFrame header",
        }

    def compute(self, ctx):
        p = self.params

        def get(drp_key, hdr_key):
            v = ctx.value(drp_key, (hdr_key,))
            return float(v) if v is not None else None

        sun_alt = get("sun_alt", "SKY SUN_ALT")
        moon_alt = get("moon_alt", "SKY MOON_ALT")
        moon_fli = get("moon_fli", "SKY MOON_FLI")
        moon_sep = get("sci_moon_sep", "SKY SCI_MOON_SEP")
        vals = {k: (v if v is not None else float("nan")) for k, v in
                dict(sun_alt=sun_alt, moon_alt=moon_alt, moon_fli=moon_fli,
                     moon_sep=moon_sep).items()}

        if sun_alt is None:
            return self.result(QAFlag.UNDEFINED, note="sun_alt unavailable", **vals)
        if sun_alt > p["twilight_sun_alt"]:
            return self.result(QAFlag.MAJOR, note=f"twilight (sun alt {sun_alt:.1f})", **vals)
        if moon_alt is None or moon_fli is None or moon_sep is None:
            return self.result(QAFlag.UNDEFINED, note="moon parameters unavailable", **vals)
        if moon_alt <= p["moon_alt_min"] or moon_sep >= p["moon_sep_min"] or moon_fli < p["fli_warn"]:
            return self.result(QAFlag.GOOD, **vals)
        if moon_fli < p["fli_major"]:
            return self.result(QAFlag.WARNING, note=f"moon FLI {moon_fli:.2f}, sep {moon_sep:.0f}", **vals)
        return self.result(QAFlag.MAJOR, note=f"bright moon FLI {moon_fli:.2f}, sep {moon_sep:.0f}", **vals)
