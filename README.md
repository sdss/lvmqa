# lvmqa
QA tools for LVM 

## Overview

`lvmqa` is a Python package that performs quality control of [LVM](https://www.sdss.org/surveys/local-volume-mapper/) data products.  It analyses row-stacked spectra (RSS) from **SFrame.fits** and **CFrame.fits** files, emission-line and stellar-population measurements from **DAP** outputs, and summary data from the **drpall** FITS table.

### Measurements (21 QA metrics)

| # | Description | Inputs |
|---|-------------|--------|
| 1 | Colour/jumps in the continuum between spectrograph arms | CFrames |
| 2 | Colour comparisons with PanSTARRS / Gaia | drpall + SFrames |
| 3 | Absolute photometry comparison with Gaia | drpall + SFrames |
| 4 | S/N of Hα | SFrames |
| 5 | S/N of the continuum | SFrames |
| 6 | Weakest detected emission line (data depth) | SFrames + DAP |
| 7 | Weakest detected continuum level | SFrames + DAP |
| 8 | Sky brightness during the observations | CFrames + drpall |
| 9 | Residual RMS / Flux of sky emission lines after subtraction | CFrames + drpall |
| 10 | Residual RMS / Flux of sky continuum | CFrames + drpall |
| 11 | EW of uncorrected telluric absorptions | SFrames |
| 12 | Night-sky extinction | drpall + CFrames |
| 13 | Wavelength calibration accuracy Δλ (per fiber + average) | CFrames + DAP |
| 14 | Spectral resolution / LOSVD accuracy from sky lines | CFrames + DAP |
| 15 | Error-propagation accuracy: RMS / error, Gaussian test | DAP + SFrames |
| 16 | Detection of saturated pixels | CFrames / drpall |
| 17 | RMS_residual / Flux for DAP emission lines | DAP |
| 18 | Compare reported emission-line errors with residual spectra | DAP + SFrames |
| 19 | Parametric vs non-parametric DAP comparison | DAP + SFrames |
| 20 | Comparison with external emission-line map compilations | DAP + external |
| 21 | Stellar properties comparison with Gaia | DAP + external Gaia |

All measurements are stored in a single FITS binary table (one row per exposure).  Per-exposure diagnostic plots can be generated alongside the table.

## Package structure

```
src/lvmqa/
├── __init__.py
├── qa.py               # Orchestrator: run_qa(), build_qa_table(), write_qa_table()
├── utils.py            # Common helpers: load_rss(), load_drpall(), snr_per_fiber(), …
├── plots.py            # Diagnostic figures for each metric
└── metrics/
    ├── continuum.py    # Reqs 1, 5, 7
    ├── photometry.py   # Reqs 2, 3
    ├── emission_lines.py # Reqs 4, 6, 17, 18
    ├── sky.py          # Reqs 8, 9, 10, 12
    ├── telluric.py     # Req 11
    ├── calibration.py  # Reqs 13, 14
    ├── errors.py       # Req 15
    ├── saturation.py   # Req 16
    └── stellar.py      # Reqs 19, 20, 21
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from lvmqa.qa import run_qa, build_qa_table, write_qa_table

# Run all metrics for a single exposure
metrics = run_qa(
    exposure_id="00012345",
    sframe_path="path/to/SFrame-00012345.fits",
    cframe_path="path/to/CFrame-00012345.fits",
    dap_path="path/to/dap-00012345.fits",
)

# Build a summary FITS table from multiple exposures
table = build_qa_table([metrics])

# Save to disk
write_qa_table(table, "qa_summary.fits")
```

## Running tests

```bash
pytest tests/
```
