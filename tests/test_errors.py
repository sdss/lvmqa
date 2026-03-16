"""Tests for lvmqa.metrics.errors."""

import numpy as np
import pytest

from lvmqa.metrics.errors import error_propagation_accuracy


def test_error_propagation_keys(sframe_hdul, dap_model_hdul):
    result = error_propagation_accuracy(sframe_hdul, dap_model_hdul)
    for key in ("chi_mean", "chi_std", "ks_stat", "ks_pvalue", "chi_values"):
        assert key in result


def test_error_propagation_chi_near_zero(sframe_hdul, dap_model_hdul):
    """Model ≈ flux → chi should be near zero."""
    result = error_propagation_accuracy(sframe_hdul, dap_model_hdul)
    if np.isfinite(result["chi_mean"]):
        assert abs(result["chi_mean"]) < 1.0


def test_error_propagation_ks_pvalue_range(sframe_hdul, dap_model_hdul):
    result = error_propagation_accuracy(sframe_hdul, dap_model_hdul)
    if np.isfinite(result["ks_pvalue"]):
        assert 0.0 <= result["ks_pvalue"] <= 1.0


def test_error_propagation_chi_values_array(sframe_hdul, dap_model_hdul):
    result = error_propagation_accuracy(sframe_hdul, dap_model_hdul)
    assert isinstance(result["chi_values"], np.ndarray)
