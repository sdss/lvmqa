"""
Saturated-pixel QA (requirement 16).

16.  Detection of saturated pixels (use CFrames or drpall).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Row

__all__ = ["detect_saturated_pixels"]

# Default saturation mask bit.  LVM uses a bitmask in the MASK extension;
# bit 1 (value 2) is typically the saturation flag.
_SAT_BIT = 2


def detect_saturated_pixels(
    hdul: fits.HDUList,
    mask_extname: str = "MASK",
    sat_bit: int = _SAT_BIT,
    sat_flux_threshold: float | None = None,
) -> dict[str, object]:
    """Count saturated pixels and report their distribution (req. 16).

    Two detection strategies are tried in order:

    1. **Bitmask** — if a ``MASK`` extension exists, pixels with ``sat_bit``
       set are flagged as saturated.
    2. **Flux threshold** — if *sat_flux_threshold* is given, pixels with
       ``FLUX > sat_flux_threshold`` are flagged.

    Parameters
    ----------
    hdul:
        HDUList of a CFrame FITS file.
    mask_extname:
        Name of the bitmask extension (default ``"MASK"``).
    sat_bit:
        Bitmask bit that indicates saturation.
    sat_flux_threshold:
        Optional flux level above which pixels are considered saturated.
        Only used when no bitmask extension is present.

    Returns
    -------
    dict with:
    - ``"n_saturated"``      — total number of saturated pixels.
    - ``"frac_saturated"``   — fraction of all pixels that are saturated.
    - ``"n_fibers_affected"``— number of fibers containing ≥1 saturated pixel.
    - ``"fiber_sat_counts"`` — 1-D array of saturated-pixel counts per fiber.
    - ``"method"``           — ``"bitmask"`` or ``"flux_threshold"``.
    """
    if mask_extname in hdul:
        mask = hdul[mask_extname].data.astype(int)
        saturated = (mask & sat_bit) != 0
        method = "bitmask"
    elif sat_flux_threshold is not None:
        flux = hdul["FLUX"].data.astype(float)
        saturated = flux > sat_flux_threshold
        method = "flux_threshold"
    else:
        # No bitmask and no threshold — return zeros
        n_pix = hdul["FLUX"].data.size
        n_fibers = hdul["FLUX"].data.shape[0]
        return {
            "n_saturated": 0,
            "frac_saturated": 0.0,
            "n_fibers_affected": 0,
            "fiber_sat_counts": np.zeros(n_fibers, dtype=int),
            "method": "none",
        }

    n_saturated = int(np.sum(saturated))
    n_total = saturated.size
    frac_saturated = n_saturated / n_total if n_total > 0 else 0.0

    fiber_sat_counts = np.sum(saturated, axis=1).astype(int)
    n_fibers_affected = int(np.sum(fiber_sat_counts > 0))

    return {
        "n_saturated": n_saturated,
        "frac_saturated": float(frac_saturated),
        "n_fibers_affected": n_fibers_affected,
        "fiber_sat_counts": fiber_sat_counts,
        "method": method,
    }
