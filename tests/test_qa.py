"""Tests for lvmqa.qa (orchestrator and table builder)."""

import numpy as np
import pytest
from astropy.table import Table

from lvmqa.qa import build_qa_table, run_qa


def test_run_qa_no_files():
    """run_qa should not crash when no files are provided."""
    metrics = run_qa("test_exposure_0001")
    assert metrics["exposure_id"] == "test_exposure_0001"


def test_run_qa_with_sframe_and_cframe(tmp_path, sframe_hdul, cframe_hdul):
    """run_qa should compute metrics when SFrame and CFrame paths are given."""
    sf_path = tmp_path / "sframe.fits"
    cf_path = tmp_path / "cframe.fits"
    sframe_hdul.writeto(str(sf_path))
    cframe_hdul.writeto(str(cf_path))

    metrics = run_qa("test_001", sframe_path=sf_path, cframe_path=cf_path)

    # Key metrics should be present and not None
    assert metrics.get("continuum_snr") is not None
    assert metrics.get("ha_snr") is not None
    assert metrics.get("arm_color_jumps") is not None
    assert metrics.get("sky_brightness") is not None
    assert metrics.get("saturation") is not None
    assert metrics.get("telluric_ew") is not None


def test_build_qa_table_single_row(sframe_hdul, cframe_hdul, tmp_path):
    sf_path = tmp_path / "sframe.fits"
    cf_path = tmp_path / "cframe.fits"
    sframe_hdul.writeto(str(sf_path))
    cframe_hdul.writeto(str(cf_path))

    metrics = run_qa("exp_001", sframe_path=sf_path, cframe_path=cf_path)
    table = build_qa_table([metrics])

    assert isinstance(table, Table)
    assert len(table) == 1
    assert "EXPOSURE_ID" in table.colnames
    assert "CONT_SNR_MEDIAN" in table.colnames
    assert "HA_SNR_MEDIAN" in table.colnames


def test_build_qa_table_multiple_rows(tmp_path, sframe_hdul, cframe_hdul):
    sf_path = tmp_path / "sframe.fits"
    cf_path = tmp_path / "cframe.fits"
    sframe_hdul.writeto(str(sf_path))
    cframe_hdul.writeto(str(cf_path))

    all_metrics = [
        run_qa(f"exp_{i:04d}", sframe_path=sf_path, cframe_path=cf_path)
        for i in range(3)
    ]
    table = build_qa_table(all_metrics)
    assert len(table) == 3


def test_write_qa_table(tmp_path, sframe_hdul, cframe_hdul):
    from lvmqa.qa import write_qa_table

    sf_path = tmp_path / "sframe.fits"
    cf_path = tmp_path / "cframe.fits"
    sframe_hdul.writeto(str(sf_path))
    cframe_hdul.writeto(str(cf_path))

    metrics = run_qa("exp_w", sframe_path=sf_path, cframe_path=cf_path)
    table = build_qa_table([metrics])

    out_path = tmp_path / "qa_table.fits"
    write_qa_table(table, out_path)
    assert out_path.exists()

    # Verify the file can be read back
    from astropy.table import Table as T
    t2 = T.read(str(out_path))
    assert len(t2) == 1
