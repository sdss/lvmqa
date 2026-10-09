"""
QA metric modules. Each module registers one (or more) metric classes with
``@register``. Add new metrics by creating a module here and importing it below
(or load an external module with ``--metric-module``).

Bitmask slots (never renumber released slots). Observational-condition components have no
slot: they enter the bitmask only through the summary flag qa_obs.

    slot  metric             group
    0     qa_obs             Observational conditions (summary of the components below)
    -     qa_illumination    Observational conditions (component, soft)
    -     qa_brightstars     Observational conditions (component, hard)
    -     qa_guider          Observational conditions (component, hard)
    1     qa_incomplete      General
    2     qa_lineratios      General
    3     qa_absflux         DRP quality
    4     qa_relflux_br      DRP quality
    5     qa_relflux_rz      DRP quality
    6     qa_skyline_sub     DRP quality
    7     qa_skycont_sub     DRP quality
    8     qa_flatfield       DRP quality
    9     qa_dap_npfit       DAP quality
    10    qa_dap_pmfit       DAP quality
"""

from . import obs_summary  # noqa: F401
from . import obs_illumination  # noqa: F401
from . import obs_brightstars  # noqa: F401
from . import obs_guider  # noqa: F401
from . import gen_incomplete  # noqa: F401
from . import gen_lineratios  # noqa: F401
from . import drp_absflux  # noqa: F401
from . import drp_relflux  # noqa: F401
from . import drp_skyline  # noqa: F401
from . import drp_skycont  # noqa: F401
from . import drp_flatfield  # noqa: F401
from . import dap_npfit  # noqa: F401
from . import dap_pmfit  # noqa: F401
