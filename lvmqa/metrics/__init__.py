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
    9     qa_fit_consistency DAP quality (formerly qa_dap_npfit)
    10    qa_dap_pmfit       DAP quality
    11    qa_wavecal_b       DRP quality
    12    qa_wavecal_r       DRP quality
    13    qa_wavecal_z       DRP quality
    14    qa_lsf             DRP quality
    15    qa_telluric        DRP quality
    16    qa_gaskinematics   DAP quality (prototype)
    17    qa_dap_npfluxes    DAP quality (prototype; formerly qa_faintlines)
    18    qa_absorption      DAP quality (prototype)
    19    qa_fluxmap         DAP quality (prototype, multi-exposure)
    20    qa_starkinematics  DAP quality (prototype)
    21    qa_line_fluxes     Science summary
    22    qa_kinematics      Science summary
    23    qa_stars           Science summary
    24    qa_abs_calibration Science summary
    25    qa_general_drp     Science summary
    -     qa_status          Tile status (information: status_auroral/bright/faint/sp)
"""

from . import sci_summary  # noqa: F401
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
from . import drp_wavecal  # noqa: F401
from . import drp_lsf  # noqa: F401
from . import drp_telluric  # noqa: F401
from . import dap_fit_consistency  # noqa: F401
from . import dap_pmfit  # noqa: F401
from . import dap_npfluxes  # noqa: F401
from . import dap_gaskinematics  # noqa: F401
from . import dap_absorption  # noqa: F401
from . import dap_starkinematics  # noqa: F401
from . import dap_fluxmap  # noqa: F401
from . import status  # noqa: F401
