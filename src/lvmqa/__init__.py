"""
lvmqa — Quality assurance tools for LVM data products.

Provides routines to compute QA metrics from SFrame, CFrame, DAP, and drpall
FITS files and to produce per-exposure diagnostic plots and a summary FITS table.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lvmqa")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
