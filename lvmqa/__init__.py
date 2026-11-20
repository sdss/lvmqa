"""
lvmqa - quality-assurance pipeline for SDSS-V LVM (and future LVM2) data.

Typical use from Python::

    from lvmqa import run_qa
    res = run_qa("1.3.2", "1.2.0.251218", tileids=[1056679])
    res.table          # astropy Table, one row per exposure

and from the command line::

    python -m lvmqa 1.3.2 1.2.0.251218 --tileids 1056679
"""

__version__ = "0.1.0.dev0"

from .bitmask import QAFlag, decode_bitmask  # noqa: E402,F401
from .pipeline import QAResult, run_qa  # noqa: E402,F401
