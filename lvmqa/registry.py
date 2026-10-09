"""
Metric base class and registry.

A metric is a small class living in its own module under ``lvmqa/metrics`` (or in an
external module loaded with ``--metric-module``). It declares

* ``name``   : flag column name, e.g. ``qa_incomplete``
* ``slot``   : fixed position in the packed bitmask (never reuse / renumber!)
* ``group``  : one of ``GROUPS``
* ``criteria``: human-readable WARNING / MAJOR ISSUE (and optional GOOD / UNDEFINED) rules
* ``values`` : the numeric/string measurements it stores (``ValueSpec``)
* ``defaults``: tunable parameters (overridable per run via ``metric_options``)

and implements ``compute(ctx) -> MetricResult``.  ``ctx`` is an
:class:`lvmqa.context.ExposureContext` giving lazy access to the drpall row, the SFrame
(header, SLITMAP, data) and the DAP tables.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from .bitmask import MAX_SLOTS, QAFlag

log = logging.getLogger("lvmqa")

GROUP_OBS = "Observational conditions"
GROUP_DRP = "DRP quality"
GROUP_DAP = "DAP quality"
GROUP_GEN = "General"
# order used for sorting metrics in tables and in the HTML report
GROUPS: Tuple[str, ...] = (GROUP_OBS, GROUP_GEN, GROUP_DRP, GROUP_DAP)
GROUP_KEYS = {GROUP_OBS: "obs", GROUP_DRP: "drp", GROUP_DAP: "dap", GROUP_GEN: "gen"}

INPUTS = ("drpall", "sframe_header", "sframe_data", "dap")


class MissingInputError(RuntimeError):
    """Raised by the context when a required input product is not available."""


@dataclass(frozen=True)
class ValueSpec:
    name: str                 # short name; column becomes "<metric short>_<name>"
    dtype: str = "f8"         # numpy dtype string ('f8', 'i8', 'bool', 'U16', ...)
    unit: str = ""
    description: str = ""

    def fill_value(self) -> Any:
        k = self.dtype[0]
        if k == "f":
            return float("nan")
        if k in "iu":
            return -1
        if k == "b":
            return False
        return ""


@dataclass
class MetricResult:
    flag: QAFlag
    values: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


class Metric:
    name: str = ""
    # bitmask slot; None = not stored in the bitmask (observational-condition components,
    # which enter the bitmask only through the qa_obs summary)
    slot: Optional[int] = None
    summary: bool = False      # summary metrics are computed from other metrics' results
    # for components of qa_obs (group "Observational conditions"):
    #   "hard": WARNING -> WARNING, MAJOR -> MAJOR ;  "soft": MAJOR -> WARNING, WARNING -> GOOD
    obs_weight: str = "hard"
    order: int = 0             # sort position among metrics without a slot
    group: str = ""
    title: str = ""
    description: str = ""
    criteria: Dict[str, str] = {}
    requires: Tuple[str, ...] = ()
    values: Tuple[ValueSpec, ...] = ()
    main_value: Optional[str] = None       # value used for summary plots
    defaults: Dict[str, Any] = {}

    def __init__(self, **options: Any) -> None:
        unknown = set(options) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown option(s) {sorted(unknown)}; "
                             f"known: {sorted(self.defaults)}")
        self.params: Dict[str, Any] = {**self.defaults, **options}

    # ------------------------------------------------------------------ helpers
    @property
    def short(self) -> str:
        return self.name[3:] if self.name.startswith("qa_") else self.name

    def column(self, value_name: str) -> str:
        return f"{self.short}_{value_name}"

    def empty_values(self) -> Dict[str, Any]:
        return {v.name: v.fill_value() for v in self.values}

    def result(self, flag: QAFlag, note: str = "", **values: Any) -> MetricResult:
        vals = self.empty_values()
        unknown = set(values) - set(vals)
        if unknown:
            raise KeyError(f"{self.name}: undeclared value(s) {sorted(unknown)}")
        vals.update(values)
        return MetricResult(QAFlag(flag), vals, note)

    # ------------------------------------------------------------------ hooks
    def prepare(self, cfg) -> None:
        """Called once in the main process before exposures are processed
        (e.g. to download / build caches). May update ``self.params``."""

    def compute(self, ctx) -> MetricResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def summarize(self, results: Dict[str, "MetricResult"], metrics: List["Metric"]) -> MetricResult:
        """Summary metrics: combine the results of the other metrics of one exposure."""
        raise NotImplementedError  # pragma: no cover

    def safe_summarize(self, results, metrics, expnum=None) -> MetricResult:
        try:
            res = self.summarize(results, metrics)
        except Exception as exc:
            log.error("%s failed for expnum %s: %s\n%s", self.name, expnum, exc,
                      traceback.format_exc())
            return self.result(QAFlag.UNDEFINED, note=f"error: {type(exc).__name__}: {exc}")
        full = self.empty_values()
        full.update(res.values)
        res.values = full
        return res

    @property
    def in_bitmask(self) -> bool:
        return self.slot is not None

    @property
    def role(self) -> str:
        if self.summary:
            return "summary"
        if self.group == GROUP_OBS:
            return f"component of qa_obs ({self.obs_weight})"
        return "bitmask"

    def safe_compute(self, ctx) -> MetricResult:
        try:
            res = self.compute(ctx)
        except MissingInputError as exc:
            return self.result(QAFlag.UNDEFINED, note=str(exc))
        except Exception as exc:  # never let one metric kill the run
            log.error("%s failed for expnum %s: %s\n%s", self.name, ctx.expnum, exc,
                      traceback.format_exc())
            return self.result(QAFlag.UNDEFINED, note=f"error: {type(exc).__name__}: {exc}")
        # make sure all declared values are present
        full = self.empty_values()
        full.update(res.values)
        res.values = full
        return res

    def params_string(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.params.items())


_REGISTRY: Dict[str, Type[Metric]] = {}


def register(cls: Type[Metric]) -> Type[Metric]:
    """Class decorator adding a metric to the registry (validates declaration)."""
    if not cls.name.startswith("qa_"):
        raise ValueError(f"metric name must start with 'qa_': {cls.name!r}")
    if cls.group not in GROUPS:
        raise ValueError(f"{cls.name}: group must be one of {GROUPS}, got {cls.group!r}")
    if cls.slot is not None and not 0 <= cls.slot < MAX_SLOTS:
        raise ValueError(f"{cls.name}: slot must be None or in [0, {MAX_SLOTS})")
    if cls.group == GROUP_OBS and not cls.summary and cls.slot is not None:
        raise ValueError(f"{cls.name}: observational-condition components must have slot=None "
                         f"(they enter the bitmask through qa_obs)")
    if cls.group != GROUP_OBS and not cls.summary and cls.slot is None:
        raise ValueError(f"{cls.name}: a bitmask slot is required")
    if cls.obs_weight not in ("hard", "soft"):
        raise ValueError(f"{cls.name}: obs_weight must be 'hard' or 'soft'")
    for other in _REGISTRY.values():
        if other is cls or other.name == cls.name:
            continue
        if cls.slot is not None and other.slot == cls.slot:
            raise ValueError(f"{cls.name}: slot {cls.slot} already used by {other.name}")
    bad = set(cls.requires) - set(INPUTS)
    if bad:
        raise ValueError(f"{cls.name}: unknown inputs {bad}; allowed {INPUTS}")
    if cls.main_value and cls.main_value not in {v.name for v in cls.values}:
        raise ValueError(f"{cls.name}: main_value {cls.main_value!r} not in values")
    _REGISTRY[cls.name] = cls
    return cls


def sort_key(m) -> tuple:
    """Group order (GROUPS), summary first within a group, then slot, then name."""
    return (GROUPS.index(m.group), 0 if m.summary else 1,
            m.slot if m.slot is not None else 100 + m.order, m.name)


def available_metrics() -> Dict[str, Type[Metric]]:
    from . import metrics  # noqa: F401  (populates the registry)
    return dict(sorted(_REGISTRY.items(), key=lambda kv: sort_key(kv[1])))


def load_metric_module(spec: str) -> None:
    """Import an external metric module (dotted name or path to a .py file)."""
    import importlib
    import importlib.util
    from pathlib import Path

    p = Path(spec)
    if p.suffix == ".py" and p.exists():
        mod_name = f"lvmqa_ext_{p.stem}"
        sp = importlib.util.spec_from_file_location(mod_name, p)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        importlib.import_module(spec)


def build_metrics(names: Optional[Sequence[str]] = None,
                  options: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Metric]:
    """Instantiate metrics (all registered ones by default), ordered by group (see sort_key)."""
    reg = available_metrics()
    options = options or {}
    if names:
        missing = [n for n in names if n not in reg]
        if missing:
            raise KeyError(f"unknown metric(s) {missing}; available: {list(reg)}")
        selected = [reg[n] for n in names]
        # a summary metric needs the components of its group
        for cls in list(selected):
            if cls.summary:
                selected += [c for c in reg.values() if c.group == cls.group and not c.summary
                             and c not in selected]
    else:
        selected = list(reg.values())
    unknown_opts = set(options) - set(reg)
    if unknown_opts:
        raise KeyError(f"options given for unknown metric(s): {sorted(unknown_opts)}")
    return sorted((cls(**options.get(cls.name, {})) for cls in selected), key=sort_key)
