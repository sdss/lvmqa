"""
Stellar-population and DAP cross-comparison QA (requirements 19, 20, 21).

19.  Comparison of parametric and non-parametric analysis between them and with
     independent measurements (use DAP outputs and SFrames).
20.  Comparisons with the existing compilation of emission-line maps (use DAP
     outputs and external sources).
21.  Stellar properties: Compare DAP results with Gaia values (velocity,
     stellar properties, Av); compare with MWM-derived quantities (use DAP
     outputs and external Gaia catalogs).
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

__all__ = [
    "parametric_nonparametric_comparison",
    "emission_line_map_comparison",
    "stellar_gaia_comparison",
]


def parametric_nonparametric_comparison(
    dap_table: Table,
    param_flux_col: str = "PARAM_FLUX",
    nonparam_flux_col: str = "NONPARAM_FLUX",
    param_vel_col: str = "PARAM_VEL",
    nonparam_vel_col: str = "NONPARAM_VEL",
) -> dict[str, object]:
    """Compare parametric and non-parametric DAP results (req. 19).

    Computes the residuals and statistics between the two analysis approaches
    for emission-line fluxes and stellar velocities.

    Parameters
    ----------
    dap_table:
        DAP output table containing both parametric and non-parametric columns.
    param_flux_col, nonparam_flux_col:
        Column names for parametric and non-parametric line fluxes.
    param_vel_col, nonparam_vel_col:
        Column names for parametric and non-parametric stellar velocities.

    Returns
    -------
    dict with sub-dicts ``"flux"`` and ``"velocity"``, each containing
    ``"delta"`` (array), ``"mean_delta"``, ``"std_delta"``, and
    ``"n_valid"`` entries.
    """
    results: dict[str, object] = {}

    for key, col_a, col_b in [
        ("flux", param_flux_col, nonparam_flux_col),
        ("velocity", param_vel_col, nonparam_vel_col),
    ]:
        if col_a in dap_table.colnames and col_b in dap_table.colnames:
            a = np.asarray(dap_table[col_a], dtype=float)
            b = np.asarray(dap_table[col_b], dtype=float)
            delta = a - b
            valid = np.isfinite(delta)
            results[key] = {
                "delta": delta,
                "mean_delta": float(np.nanmean(delta)),
                "std_delta": float(np.nanstd(delta)),
                "n_valid": int(np.sum(valid)),
            }
        else:
            results[key] = {
                "delta": np.array([np.nan]),
                "mean_delta": np.nan,
                "std_delta": np.nan,
                "n_valid": 0,
            }

    return results


def emission_line_map_comparison(
    dap_table: Table,
    catalogue_table: Table,
    dap_flux_col: str = "LINE_FLUX",
    cat_flux_col: str = "CAT_FLUX",
    match_col: str = "FIBER_ID",
) -> dict[str, object]:
    """Compare DAP emission-line fluxes with an external catalogue (req. 20).

    Both tables are matched on *match_col* (e.g. a fiber or source ID).

    Parameters
    ----------
    dap_table:
        DAP output table.
    catalogue_table:
        External compilation table to compare against.
    dap_flux_col:
        Flux column in the DAP table.
    cat_flux_col:
        Flux column in the catalogue table.
    match_col:
        Column used to cross-match rows.

    Returns
    -------
    dict with ``"flux_ratio"`` (array), ``"mean_flux_ratio"``,
    ``"std_flux_ratio"``, and ``"n_matched"``.
    """
    if match_col not in dap_table.colnames or match_col not in catalogue_table.colnames:
        return {
            "flux_ratio": np.array([np.nan]),
            "mean_flux_ratio": np.nan,
            "std_flux_ratio": np.nan,
            "n_matched": 0,
        }
    if dap_flux_col not in dap_table.colnames or cat_flux_col not in catalogue_table.colnames:
        return {
            "flux_ratio": np.array([np.nan]),
            "mean_flux_ratio": np.nan,
            "std_flux_ratio": np.nan,
            "n_matched": 0,
        }

    # Cross-match by ID
    dap_ids = np.asarray(dap_table[match_col])
    cat_ids = np.asarray(catalogue_table[match_col])

    dap_flux = np.asarray(dap_table[dap_flux_col], dtype=float)
    cat_flux = np.asarray(catalogue_table[cat_flux_col], dtype=float)

    matched_dap, matched_cat = [], []
    for i, did in enumerate(dap_ids):
        idx = np.where(cat_ids == did)[0]
        if idx.size > 0:
            matched_dap.append(dap_flux[i])
            matched_cat.append(cat_flux[idx[0]])

    if not matched_dap:
        return {
            "flux_ratio": np.array([np.nan]),
            "mean_flux_ratio": np.nan,
            "std_flux_ratio": np.nan,
            "n_matched": 0,
        }

    f_dap = np.asarray(matched_dap, dtype=float)
    f_cat = np.asarray(matched_cat, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(f_cat != 0.0, f_dap / f_cat, np.nan)

    return {
        "flux_ratio": ratio,
        "mean_flux_ratio": float(np.nanmean(ratio)),
        "std_flux_ratio": float(np.nanstd(ratio)),
        "n_matched": len(matched_dap),
    }


def stellar_gaia_comparison(
    dap_table: Table,
    gaia_table: Table,
    dap_vel_col: str = "STELLAR_VEL",
    gaia_vel_col: str = "GAIA_RV",
    dap_av_col: str = "STELLAR_AV",
    gaia_av_col: str = "GAIA_AV",
    match_col: str = "SOURCE_ID",
) -> dict[str, object]:
    """Compare DAP stellar parameters with Gaia catalogue values (req. 21).

    Parameters
    ----------
    dap_table:
        DAP output table with stellar parameters.
    gaia_table:
        Gaia catalogue table.
    dap_vel_col, gaia_vel_col:
        Stellar radial velocity columns (km/s).
    dap_av_col, gaia_av_col:
        V-band extinction columns (mag).
    match_col:
        Column used to cross-match rows (e.g. Gaia source ID).

    Returns
    -------
    dict with sub-dicts ``"velocity"`` and ``"extinction"``, each containing
    ``"delta"`` (array), ``"mean_delta"``, and ``"std_delta"``.
    """
    results: dict[str, object] = {}

    if match_col not in dap_table.colnames or match_col not in gaia_table.colnames:
        for key in ("velocity", "extinction"):
            results[key] = {"delta": np.array([np.nan]), "mean_delta": np.nan, "std_delta": np.nan, "n_matched": 0}
        return results

    dap_ids = np.asarray(dap_table[match_col])
    gaia_ids = np.asarray(gaia_table[match_col])

    for key, col_dap, col_gaia in [
        ("velocity", dap_vel_col, gaia_vel_col),
        ("extinction", dap_av_col, gaia_av_col),
    ]:
        if col_dap not in dap_table.colnames or col_gaia not in gaia_table.colnames:
            results[key] = {"delta": np.array([np.nan]), "mean_delta": np.nan, "std_delta": np.nan, "n_matched": 0}
            continue

        dap_vals = np.asarray(dap_table[col_dap], dtype=float)
        gaia_vals = np.asarray(gaia_table[col_gaia], dtype=float)

        deltas = []
        for i, did in enumerate(dap_ids):
            idx = np.where(gaia_ids == did)[0]
            if idx.size > 0:
                deltas.append(dap_vals[i] - gaia_vals[idx[0]])

        if deltas:
            d_arr = np.asarray(deltas, dtype=float)
            results[key] = {
                "delta": d_arr,
                "mean_delta": float(np.nanmean(d_arr)),
                "std_delta": float(np.nanstd(d_arr)),
                "n_matched": len(deltas),
            }
        else:
            results[key] = {"delta": np.array([np.nan]), "mean_delta": np.nan, "std_delta": np.nan, "n_matched": 0}

    return results
