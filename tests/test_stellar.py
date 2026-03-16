"""Tests for lvmqa.metrics.stellar."""

import numpy as np
import pytest
from astropy.table import Table

from lvmqa.metrics.stellar import (
    emission_line_map_comparison,
    parametric_nonparametric_comparison,
    stellar_gaia_comparison,
)


def test_parametric_nonparametric_keys(dap_table):
    result = parametric_nonparametric_comparison(dap_table)
    assert "flux" in result
    assert "velocity" in result
    for sub in (result["flux"], result["velocity"]):
        assert "mean_delta" in sub
        assert "std_delta" in sub
        assert "n_valid" in sub


def test_parametric_nonparametric_n_valid(dap_table):
    result = parametric_nonparametric_comparison(dap_table)
    assert result["flux"]["n_valid"] == len(dap_table)


def test_parametric_nonparametric_missing_cols():
    t = Table({"A": [1.0]})
    result = parametric_nonparametric_comparison(t)
    assert result["flux"]["n_valid"] == 0
    assert result["velocity"]["n_valid"] == 0


def test_emission_line_map_comparison(dap_table_with_fiber_id, catalogue_table):
    result = emission_line_map_comparison(
        dap_table_with_fiber_id,
        catalogue_table,
        dap_flux_col="LINE_FLUX",
        cat_flux_col="CAT_FLUX",
        match_col="FIBER_ID",
    )
    assert result["n_matched"] == len(dap_table_with_fiber_id)
    assert np.isfinite(result["mean_flux_ratio"])
    # catalogue is 1.05× the dap flux → ratio ≈ 1/1.05
    assert result["mean_flux_ratio"] == pytest.approx(1.0 / 1.05, rel=0.05)


def test_emission_line_map_comparison_no_match():
    dap = Table({"FIBER_ID": [99], "LINE_FLUX": [1.0]})
    cat = Table({"FIBER_ID": [0], "CAT_FLUX": [1.0]})
    result = emission_line_map_comparison(dap, cat)
    assert result["n_matched"] == 0


def test_stellar_gaia_comparison(dap_stellar_table, gaia_table):
    result = stellar_gaia_comparison(dap_stellar_table, gaia_table)
    assert "velocity" in result
    assert "extinction" in result
    assert result["velocity"]["n_matched"] == len(gaia_table)


def test_stellar_gaia_comparison_missing_match_col():
    dap = Table({"STELLAR_VEL": [10.0]})
    gaia = Table({"GAIA_RV": [8.0]})
    result = stellar_gaia_comparison(dap, gaia)
    assert result["velocity"]["n_matched"] == 0
