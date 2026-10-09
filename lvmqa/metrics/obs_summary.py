"""
qa_obs - summary of the observational-condition flags (the only observational flag stored
in the bitmask).

Every metric of the group "Observational conditions" (except qa_obs itself) is a component.
Its contribution depends on its ``obs_weight``:

    hard  (qa_brightstars, qa_guider, ...):  WARNING -> WARNING,  MAJOR -> MAJOR
    soft  (qa_illumination, ...):            WARNING -> GOOD,     MAJOR -> WARNING

qa_obs is the worst contribution over the components that could be evaluated. New
components (e.g. saturated pixels) only need to be registered in the observational group
with the appropriate obs_weight - qa_obs picks them up automatically.
"""

from __future__ import annotations

from ..bitmask import FLAG_LABELS, QAFlag
from ..registry import GROUP_OBS, Metric, ValueSpec, register

_SOFT = {QAFlag.GOOD: QAFlag.GOOD, QAFlag.WARNING: QAFlag.GOOD, QAFlag.MAJOR: QAFlag.WARNING}
_SEV = {QAFlag.GOOD: 0, QAFlag.WARNING: 1, QAFlag.MAJOR: 2}


def _components():
    from ..registry import available_metrics
    return [c for c in available_metrics().values() if c.group == GROUP_OBS and not c.summary]


@register
class ObsSummary(Metric):
    name = "qa_obs"
    slot = 0
    summary = True
    group = GROUP_OBS
    title = "Observational conditions (summary)"
    description = ("Summary of the observational-condition flags; the individual "
                   "observational flags are not stored in the bitmask")
    requires = ()
    defaults = {}
    values = (
        ValueSpec("components", "U160", "", "Component flags used (name=flag)"),
        ValueSpec("worst_component", "U32", "", "Component that determined the flag"),
        ValueSpec("n_components", "i8", "", "Number of evaluable components"),
    )

    @property
    def criteria(self):  # type: ignore[override]
        comps = _components()
        hard = ", ".join(c.name for c in comps if c.obs_weight == "hard") or "-"
        soft = ", ".join(c.name for c in comps if c.obs_weight == "soft") or "-"
        return {
            "GOOD": f"No WARNING/MAJOR in {hard}, and no MAJOR in {soft} (WARNING there is ignored)",
            "WARNING": f"MAJOR ISSUE in any of {soft}, or WARNING in any of {hard}",
            "MAJOR ISSUE": f"MAJOR ISSUE in any of {hard}",
            "UNDEFINED": "No component flag could be evaluated",
        }

    def compute(self, ctx):  # pragma: no cover - summary metric
        raise RuntimeError("qa_obs is a summary metric (use summarize)")

    def summarize(self, results, metrics):
        contrib, parts = {}, []
        for m in metrics:
            if m.group != GROUP_OBS or m.summary or m.name not in results:
                continue
            f = results[m.name].flag
            parts.append(f"{m.short}={FLAG_LABELS[f]}")
            if f == QAFlag.UNDEFINED:
                continue
            contrib[m.name] = _SOFT[f] if m.obs_weight == "soft" else f
        vals = dict(components=";".join(parts), n_components=len(contrib))
        if not contrib:
            return self.result(QAFlag.UNDEFINED, note="no observational flag evaluable", **vals)
        worst = max(contrib, key=lambda k: (_SEV[contrib[k]], k))
        flag = contrib[worst]
        vals["worst_component"] = worst if flag != QAFlag.GOOD else ""
        return self.result(flag, note=(f"from {worst}" if flag != QAFlag.GOOD else ""), **vals)
