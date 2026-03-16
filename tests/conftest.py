"""
Shared pytest fixtures: synthetic FITS HDULists and Astropy Tables that
mimic the structure of real LVM SFrame, CFrame, and DAP files without
requiring actual data on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
N_FIBERS = 30
N_WAVE = 4001        # covers 3600–9601 Å at 1.5 Å/pixel
WAVE_START = 3600.0  # Å
WAVE_STEP = 1.5      # Å / pixel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wave() -> np.ndarray:
    return WAVE_START + np.arange(N_WAVE) * WAVE_STEP


def _make_continuum_flux(seed: int = 0) -> np.ndarray:
    """Smooth, featureless continuum (n_fibers × n_wave)."""
    rng = np.random.default_rng(seed)
    base = 1e-16 * (1.0 + 0.1 * rng.standard_normal((N_FIBERS, N_WAVE)))
    return base.astype(np.float32)


def _make_ivar(flux: np.ndarray, noise_level: float = 5e-18) -> np.ndarray:
    ivar = 1.0 / noise_level**2 * np.ones_like(flux)
    return ivar.astype(np.float32)


def _make_mask(flux: np.ndarray) -> np.ndarray:
    """Zero mask — no bad or saturated pixels."""
    return np.zeros(flux.shape, dtype=np.int32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wave() -> np.ndarray:
    return _make_wave()


@pytest.fixture
def sframe_hdul() -> fits.HDUList:
    """Minimal SFrame HDUList."""
    wave = _make_wave()
    flux = _make_continuum_flux(seed=1)
    ivar = _make_ivar(flux)
    mask = _make_mask(flux)

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=flux, name="FLUX"),
        fits.ImageHDU(data=ivar, name="IVAR"),
        fits.ImageHDU(data=mask, name="MASK"),
        fits.ImageHDU(data=wave, name="WAVE"),
    ])
    return hdul


@pytest.fixture
def cframe_hdul() -> fits.HDUList:
    """Minimal CFrame HDUList (includes sky emission lines)."""
    wave = _make_wave()
    rng = np.random.default_rng(42)

    flux = _make_continuum_flux(seed=2)

    # Add a synthetic OI sky line at 5577.3 Å
    oi_idx = np.argmin(np.abs(wave - 5577.3))
    sigma_pix = 2  # pixels
    pix = np.arange(N_WAVE)
    sky_line = 5e-15 * np.exp(-0.5 * ((pix - oi_idx) / sigma_pix) ** 2)
    flux = flux + sky_line[np.newaxis, :]

    ivar = _make_ivar(flux)
    mask = _make_mask(flux)
    sky = _make_continuum_flux(seed=3) * 0.1  # sky-only fibers

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=flux, name="FLUX"),
        fits.ImageHDU(data=ivar, name="IVAR"),
        fits.ImageHDU(data=mask, name="MASK"),
        fits.ImageHDU(data=wave, name="WAVE"),
        fits.ImageHDU(data=sky, name="SKY"),
    ])
    return hdul


@pytest.fixture
def cframe_with_saturation() -> fits.HDUList:
    """CFrame HDUList with some saturated pixels flagged in MASK."""
    wave = _make_wave()
    flux = _make_continuum_flux(seed=5)
    ivar = _make_ivar(flux)
    mask = _make_mask(flux)

    # Flag 10 pixels in first fiber as saturated (bit 2)
    mask[0, :10] = 2

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=flux, name="FLUX"),
        fits.ImageHDU(data=ivar, name="IVAR"),
        fits.ImageHDU(data=mask, name="MASK"),
        fits.ImageHDU(data=wave, name="WAVE"),
    ])
    return hdul


@pytest.fixture
def dap_table() -> Table:
    """Minimal DAP emission-line table."""
    n = 10
    rng = np.random.default_rng(7)
    return Table({
        "LINE_NAME": [f"L{i}" for i in range(n)],
        "LINE_WAVE": WAVE_START + rng.uniform(50, N_WAVE * WAVE_STEP - 50, n),
        "LINE_FLUX": rng.uniform(1e-17, 1e-15, n),
        "LINE_FLUX_ERR": rng.uniform(1e-18, 1e-16, n),
        "LINE_SNR": rng.uniform(1, 20, n),
        "LINE_RMS": rng.uniform(1e-18, 1e-16, n),
        "PARAM_FLUX": rng.uniform(1e-17, 1e-15, n),
        "NONPARAM_FLUX": rng.uniform(1e-17, 1e-15, n),
        "PARAM_VEL": rng.uniform(-50, 50, n),
        "NONPARAM_VEL": rng.uniform(-50, 50, n),
    })


@pytest.fixture
def dap_model_hdul(sframe_hdul: fits.HDUList) -> fits.HDUList:
    """DAP model HDUList — model is slightly perturbed SFrame flux."""
    flux = sframe_hdul["FLUX"].data.astype(float)
    rng = np.random.default_rng(99)
    model = flux + 1e-19 * rng.standard_normal(flux.shape)
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=model.astype(np.float32), name="MODEL"),
    ])


@pytest.fixture
def drpall_row() -> object:
    """Minimal drpall-like row (plain dict with colnames attribute)."""

    class _Row:
        def __init__(self, data: dict) -> None:
            self._data = data
            self.colnames = list(data.keys())

        def __getitem__(self, key: str) -> float:
            return self._data[key]

    return _Row({
        "GAIA_GMAG": 18.5,
        "GAIA_COLOR_GBP_GRP": 0.8,
    })


@pytest.fixture
def gaia_table() -> Table:
    n = 5
    return Table({
        "SOURCE_ID": [f"src{i}" for i in range(n)],
        "GAIA_RV": np.linspace(-30, 30, n),
        "GAIA_AV": np.linspace(0.1, 1.5, n),
    })


@pytest.fixture
def dap_stellar_table(gaia_table: Table) -> Table:
    return Table({
        "SOURCE_ID": gaia_table["SOURCE_ID"],
        "STELLAR_VEL": gaia_table["GAIA_RV"] + np.random.default_rng(11).uniform(-5, 5, len(gaia_table)),
        "STELLAR_AV": gaia_table["GAIA_AV"] + np.random.default_rng(12).uniform(-0.1, 0.1, len(gaia_table)),
    })


@pytest.fixture
def catalogue_table(dap_table: Table) -> Table:
    """External emission-line catalogue matching dap_table by FIBER_ID."""
    n = len(dap_table)
    return Table({
        "FIBER_ID": np.arange(n),
        "CAT_FLUX": dap_table["LINE_FLUX"] * 1.05,
    })


@pytest.fixture
def dap_table_with_fiber_id(dap_table: Table) -> Table:
    dap_table["FIBER_ID"] = np.arange(len(dap_table))
    return dap_table
