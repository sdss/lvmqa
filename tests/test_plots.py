"""Tests for lvmqa.plots — verify figures are created without errors."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lvmqa.plots import (
    plot_arm_color_jumps,
    plot_calibration_offsets,
    plot_continuum_snr,
    plot_emission_rms_over_flux,
    plot_error_distribution,
    plot_ha_snr,
    plot_saturation_map,
    plot_sky_brightness,
    plot_sky_line_residuals,
    plot_spectral_resolution,
    plot_summary_dashboard,
    plot_telluric_ew,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_arm_color_jumps():
    data = {"blue_red": np.array([0.95, 1.02, 0.98, np.nan])}
    fig = plot_arm_color_jumps(data)
    assert fig is not None


def test_plot_continuum_snr():
    snr = np.array([10.0, 12.0, 9.5, np.nan])
    fig = plot_continuum_snr(snr)
    assert fig is not None


def test_plot_ha_snr():
    snr = np.array([5.0, 7.0, 3.5])
    fig = plot_ha_snr(snr)
    assert fig is not None


def test_plot_sky_brightness():
    result = {"sky_median": 1e-17, "sky_std": 1e-18}
    fig = plot_sky_brightness(result)
    assert fig is not None


def test_plot_sky_line_residuals():
    result = {
        "line_wave": np.array([5577.3, 6300.3]),
        "rms_over_flux": np.array([0.05, 0.08]),
    }
    fig = plot_sky_line_residuals(result)
    assert fig is not None


def test_plot_calibration_offsets():
    result = {
        "line_wave": np.array([5577.3, 6300.3]),
        "delta_lambda": np.array([[0.1, -0.2, 0.05], [0.3, 0.1, -0.1]]),
    }
    fig = plot_calibration_offsets(result)
    assert fig is not None


def test_plot_spectral_resolution():
    result = {
        "line_wave": np.array([5577.3, 6300.3]),
        "mean_fwhm": np.array([2.5, 2.7]),
        "mean_R": np.array([2200.0, 2330.0]),
    }
    fig = plot_spectral_resolution(result)
    assert fig is not None


def test_plot_telluric_ew():
    result = {
        "O2_B": {"ew_median": 5.0, "ew_std": 0.5},
        "O2_A": {"ew_median": 12.0, "ew_std": 1.2},
    }
    fig = plot_telluric_ew(result)
    assert fig is not None


def test_plot_error_distribution():
    rng = np.random.default_rng(0)
    chi = rng.standard_normal(1000)
    result = {"chi_values": chi, "chi_mean": 0.0, "chi_std": 1.0}
    fig = plot_error_distribution(result)
    assert fig is not None


def test_plot_saturation_map():
    result = {
        "fiber_sat_counts": np.array([0, 0, 5, 0, 2]),
        "n_saturated": 7,
        "frac_saturated": 0.001,
    }
    fig = plot_saturation_map(result)
    assert fig is not None


def test_plot_emission_rms_over_flux():
    result = {
        "rms_over_flux": np.array([0.05, 0.08, 0.12]),
        "line_names": ["Hα", "Hβ", "OI"],
    }
    fig = plot_emission_rms_over_flux(result)
    assert fig is not None


def test_plot_summary_dashboard():
    metrics = {
        "arm_color_jumps": {"blue_red": np.array([1.0, 0.98])},
        "continuum_snr": np.array([12.0, 10.0]),
        "ha_snr": np.array([5.0, 6.0]),
        "sky_brightness": {"sky_median": 1e-17, "sky_std": 1e-18},
        "sky_continuum_residuals": {
            "cont_flux_before": 1e-17, "cont_rms_after": 5e-19, "cont_rms_over_flux": 0.05
        },
        "saturation": {"n_saturated": 0, "frac_saturated": 0.0, "n_fibers_affected": 0},
        "wavelength_calibration": {"mean_delta_lambda": np.array([0.1, -0.05])},
        "error_propagation": {"chi_mean": 0.01, "chi_std": 1.02, "ks_pvalue": 0.6},
        "telluric_ew": {
            "O2_A": {"ew_median": 12.0},
            "H2O_1": {"ew_median": 30.0},
        },
    }
    fig = plot_summary_dashboard(metrics, exposure_id="test_001")
    assert fig is not None
