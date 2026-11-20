"""
Science-driven summary flags (group "Science summary"). Each summary is the worst
contribution of its components; a component is

    hard:  WARNING -> WARNING, MAJOR -> MAJOR
    soft:  WARNING -> GOOD,    MAJOR -> WARNING   (relevant but not decisive for this science)

UNDEFINED components are ignored; a summary without any evaluable component is UNDEFINED.
qa_incomplete enters only with the inputs the summary needs (``needs``): a WARNING caused only
by a missing DAP file / DAP output file counts as GOOD for summaries that do not need it
(e.g. qa_general_drp, qa_kinematics without DAP output).
A component may enter several summaries. Components missing from the run are skipped.

    qa_line_fluxes      emission-line fluxes (maps, ratios)
    qa_kinematics       gas (and stellar) kinematics: velocity zero point, dispersions
    qa_stars            stellar continuum / populations
    qa_abs_calibration  absolute and relative flux calibration
    qa_general_drp      instrument and reduction integrity
(qa_obs, the observational-conditions summary, is defined in obs_summary.py.)
"""

from __future__ import annotations

from ..bitmask import FLAG_LABELS, QAFlag
from ..registry import GROUP_SCI, Metric, ValueSpec, register

_SOFT = {QAFlag.GOOD: QAFlag.GOOD, QAFlag.WARNING: QAFlag.GOOD, QAFlag.MAJOR: QAFlag.WARNING}
_SEV = {QAFlag.GOOD: 0, QAFlag.WARNING: 1, QAFlag.MAJOR: 2}


class SciSummary(Metric):
    summary = True
    group = GROUP_SCI
    requires = ()
    defaults = {}
    needs: tuple = ("dap", "dap_output")
    _INPUT_NOTES = {"dap": "DAP missing", "dap_output": "DAP output missing"}
    values = (
        ValueSpec("components", "U400", "", "Component flags used (name=flag)"),
        ValueSpec("worst_component", "U32", "", "Component that determined the flag"),
        ValueSpec("n_components", "i8", "", "Number of evaluable components"),
    )

    @property
    def criteria(self):  # type: ignore[override]
        hard = ", ".join(k for k, w in self.components.items() if w == "hard") or "-"
        soft = ", ".join(k for k, w in self.components.items() if w == "soft") or "-"
        return {
            "GOOD": f"No WARNING/MAJOR in [{hard}] and no MAJOR in [{soft}]",
            "WARNING": f"WARNING in any of [{hard}], or MAJOR ISSUE in any of [{soft}]",
            "MAJOR ISSUE": f"MAJOR ISSUE in any of [{hard}]",
            "UNDEFINED": "No component could be evaluated",
        }

    def compute(self, ctx):  # pragma: no cover
        raise RuntimeError(f"{self.name} is a summary metric (use summarize)")

    def summarize(self, results, metrics):
        contrib, parts = {}, []
        for name, w in self.components.items():
            if name not in results:
                continue
            f = results[name].flag
            parts.append(f"{name[3:]}={FLAG_LABELS[f]}")
            if f == QAFlag.UNDEFINED:
                continue
            if name == "qa_incomplete" and f == QAFlag.WARNING:
                reasons = [x.strip() for x in results[name].note.split(",") if x.strip()]
                left = [x for x in reasons if not any(x == m and k not in self.needs
                                                      for k, m in self._INPUT_NOTES.items())]
                if reasons and not left:
                    f = QAFlag.GOOD
            contrib[name] = _SOFT[f] if w == "soft" else f
        vals = dict(components=";".join(parts), n_components=len(contrib))
        if not contrib:
            return self.result(QAFlag.UNDEFINED, note="no component evaluable", **vals)
        worst = max(contrib, key=lambda k: (_SEV[contrib[k]], k))
        flag = contrib[worst]
        vals["worst_component"] = worst if flag != QAFlag.GOOD else ""
        return self.result(flag, note=(f"from {worst}" if flag != QAFlag.GOOD else ""), **vals)


def _make(name, slot, title, description, components, needs=("dap", "dap_output")):
    cls = type(name, (SciSummary,), dict(name=name, slot=slot, title=title, needs=tuple(needs),
                                         description=description, components=components))
    return register(cls)


_make("qa_line_fluxes", 21, "Emission-line fluxes (summary)",
      "Reliability of emission-line fluxes, maps and line ratios",
      {"qa_fit_consistency": "hard", "qa_dap_pmfit": "hard", "qa_dap_npfluxes": "hard",
       "qa_lineratios": "hard", "qa_fluxmap": "hard", "qa_gradient": "hard", "qa_relflux_br": "hard",
       "qa_relflux_rz": "hard", "qa_absflux": "soft", "qa_skyline_sub": "soft", "qa_skycont_sub": "soft",
       "qa_flatfield": "soft", "qa_telluric": "soft", "qa_incomplete": "hard"})
_make("qa_kinematics", 22, "Kinematics (summary)",
      "Reliability of gas and stellar velocities and velocity dispersions",
      {"qa_gaskinematics": "hard", "qa_wavecal_b": "hard", "qa_wavecal_r": "hard", "qa_wavecal_z": "hard",
       "qa_lsf": "soft", "qa_starkinematics": "soft", "qa_skyline_sub": "soft", "qa_incomplete": "hard"},
      needs=("dap",))
_make("qa_stars", 23, "Stellar continuum (summary)",
      "Reliability of the stellar continuum and stellar-population analysis",
      {"qa_absorption": "soft", "qa_starkinematics": "hard", "qa_absflux": "hard", "qa_gradient": "soft",
       "qa_relflux_br": "hard", "qa_relflux_rz": "hard", "qa_skycont_sub": "hard",
       "qa_telluric": "soft", "qa_flatfield": "soft", "qa_incomplete": "hard"},
      needs=("dap",))
_make("qa_abs_calibration", 24, "Flux calibration (summary)",
      "Absolute and relative flux calibration",
      {"qa_absflux": "hard", "qa_relflux_br": "hard", "qa_relflux_rz": "hard",
       "qa_telluric": "hard", "qa_fluxmap": "hard", "qa_gradient": "hard", "qa_flatfield": "soft"})
_make("qa_general_drp", 25, "General DRP (summary)",
      "Instrument and data-reduction integrity",
      {"qa_incomplete": "hard", "qa_flatfield": "hard", "qa_skyline_sub": "hard",
       "qa_skycont_sub": "hard", "qa_lsf": "hard", "qa_wavecal_b": "hard", "qa_wavecal_r": "hard",
       "qa_wavecal_z": "hard"},
      needs=())
