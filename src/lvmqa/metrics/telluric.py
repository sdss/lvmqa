"""
Telluric absorption QA (requirement 11).

11.  Equivalent width (EW) of the uncorrected telluric absorptions (use SFrames).

The main telluric absorption bands measured here are:
  * O₂ B-band  ~6860–6930 Å
  * O₂ A-band  ~7590–7680 Å
  * H₂O band   ~7150–7350 Å
  * H₂O band   ~8100–8360 Å
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from lvmqa.utils import get_wavelength_array

__all__ = ["telluric_ew"]

# Telluric band definitions: (label, band_min, band_max, cont_blue, cont_red)
_TELLURIC_BANDS: list[tuple[str, float, float, float, float]] = [
    ("O2_B",  6860.0, 6930.0, 6820.0, 6860.0, ),
    ("H2O_1", 7150.0, 7350.0, 7100.0, 7150.0, ),
    ("O2_A",  7590.0, 7680.0, 7540.0, 7590.0, ),
    ("H2O_2", 8100.0, 8360.0, 8050.0, 8100.0, ),
]
# Re-format as (label, band_min, band_max, cont_blue_min, cont_blue_max,
#               cont_red_min, cont_red_max)
_BANDS: list[tuple[str, float, float, float, float, float, float]] = [
    ("O2_B",  6860.0, 6930.0, 6820.0, 6860.0, 6930.0, 6970.0),
    ("H2O_1", 7150.0, 7350.0, 7100.0, 7150.0, 7350.0, 7400.0),
    ("O2_A",  7590.0, 7680.0, 7540.0, 7590.0, 7680.0, 7730.0),
    ("H2O_2", 8100.0, 8360.0, 8050.0, 8100.0, 8360.0, 8420.0),
]


def telluric_ew(hdul: fits.HDUList) -> dict[str, dict[str, float]]:
    """Measure the mean EW of telluric absorption bands per fiber (req. 11).

    The continuum is estimated from flanking windows on each side of the band,
    linearly interpolated across the band, and the EW is computed as:

        EW = ∫ (1 − F_obs / F_cont) dλ

    Parameters
    ----------
    hdul:
        HDUList of an SFrame FITS file with ``FLUX`` extension.

    Returns
    -------
    dict
        Outer key is the band label (e.g. ``"O2_A"``).  Each value is a sub-dict
        with ``"ew_median"`` (float, Å), ``"ew_std"`` (float, Å), and
        ``"ew_per_fiber"`` (1-D ``numpy.ndarray``).
    """
    wave = get_wavelength_array(hdul)
    flux = hdul["FLUX"].data.astype(float)

    results: dict[str, dict[str, float]] = {}

    for label, bmin, bmax, cb_min, cb_max, cr_min, cr_max in _BANDS:
        band_mask = (wave >= bmin) & (wave <= bmax)
        cont_blue_mask = (wave >= cb_min) & (wave <= cb_max)
        cont_red_mask = (wave >= cr_min) & (wave <= cr_max)

        if not (np.any(band_mask) and np.any(cont_blue_mask) and np.any(cont_red_mask)):
            results[label] = {"ew_median": np.nan, "ew_std": np.nan, "ew_per_fiber": np.array([])}
            continue

        # Median continuum in each flanking window per fiber
        f_blue = np.nanmedian(flux[:, cont_blue_mask], axis=1)  # (n_fibers,)
        f_red = np.nanmedian(flux[:, cont_red_mask], axis=1)

        # Linear interpolation of continuum level across the band
        w_blue = float(np.nanmean(wave[cont_blue_mask]))
        w_red = float(np.nanmean(wave[cont_red_mask]))
        w_band = wave[band_mask]

        # For each fiber, interpolate a continuum level at each pixel
        # f_cont[f, p] = f_blue[f] + (w_band[p] - w_blue) / (w_red - w_blue) * (f_red[f] - f_blue[f])
        if w_red == w_blue:
            f_cont = 0.5 * (f_blue + f_red)[:, np.newaxis] * np.ones(int(np.sum(band_mask)))[np.newaxis, :]
        else:
            frac = (w_band - w_blue) / (w_red - w_blue)
            f_cont = f_blue[:, np.newaxis] + frac[np.newaxis, :] * (f_red - f_blue)[:, np.newaxis]

        f_obs = flux[:, band_mask]  # (n_fibers, n_band)
        dwave = np.diff(w_band)
        dwave = np.append(dwave, dwave[-1]) if dwave.size > 0 else np.ones(w_band.size)

        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(f_cont != 0.0, 1.0 - f_obs / f_cont, 0.0)
        ew_per_fiber = np.nansum(integrand * dwave[np.newaxis, :], axis=1)

        results[label] = {
            "ew_median": float(np.nanmedian(ew_per_fiber)),
            "ew_std": float(np.nanstd(ew_per_fiber)),
            "ew_per_fiber": ew_per_fiber,
        }

    return results
