# lvmqa

Quality assurance tools for LVM. 

## QA pipeline (`lvmqa` package)

Per-exposure QA metrics and 2-bit flags from SFrames, DAP output and the drpall file.

```bash
pip install -e .            # optional; or run from the repo root with python -m lvmqa
lvmqa 1.3.2 1.2.0.251218 --tileids 1056679 --mjds 61235 61293 -j 4
lvmqa 1.3.2 1.2.0.251218 --expnums 62131-62136 9068,9069 --existing-only
lvmqa --list-metrics
lvmqa 1.2.0 none --tileids 1027689   # no DAP version: legacy (DR20) layout analysis/<drpver>/<tilegrp>/...
lvmqa 1.3.2 1.2.0.251218 --set qa_dap_npfit.min_snr=3 --metrics qa_dap_npfit --tag snr3
```

```python
from lvmqa import run_qa
res = run_qa("1.3.2", "1.2.0.251218", tileids=[1056679], nproc=4)
res.table          # astropy Table (also written to FITS / CSV / HTML)
```

Paths: `--redux-dir`, `--analysis-dir`, `--output-dir` (defaults: `$LVM_SPECTRO_REDUX`,
`$LVM_SPECTRO_ANALYSIS`, `$LVM_QA_OUTPUT_DIR`, else the development test paths in
`lvmqa/config.py`).

Outputs (`lvmqa-<drpver>-<dapver>[-tag]*`):

* `.fits` / `.xlsx` (final QA, DR20-like): `QA` = expnum, tileid, mjd, object, `qa_bitmask`
  (int64) and one flag column per metric (GOOD / WARNING / MAJOR ISSUE / UNDEFINED; coloured
  cells in the xlsx), `DESC` = metric group, slot/bits, criteria and parameters
* `_supl.fits` / `_supl.xlsx`: `SUPL` = ids, observing info (exptime, obstime, dpos,
  coordinates, airmass, fluxcal), all values measured by the metrics, processing times and
  notes; `COLUMNS` = description and unit of every column
* `.html`: plotly report (status overview, sky map, trends, processing-time statistics)

Encoding: 2 bits per metric at bits `2*slot, 2*slot+1`: GOOD=00, WARNING=01,
MAJOR ISSUE=10, UNDEFINED=11. Slots are fixed per metric (see `lvmqa/metrics/__init__.py`).
Observational-condition flags (`qa_illumination`, `qa_brightstars`, `qa_guider`, ...) are not in
the bitmask; they enter it only through the summary flag `qa_obs` (slot 0). New observational
components only need to be registered in the "Observational conditions" group with
`slot = None` and `obs_weight = "hard"` or `"soft"`.

### Adding a metric

Create `lvmqa/metrics/<group>_<name>.py` with a `Metric` subclass decorated by `@register`
(declare `name`, unused `slot`, `group` - one of "Observational conditions", "DRP quality",
"DAP quality", "General" -, `criteria`, `values`, `defaults`) and implement
`compute(ctx)` returning `self.result(flag, note, **values)`. `ctx` gives lazy access to
`ctx.row` (drpall), `ctx.header`, `ctx.slitmap`, `ctx.sframe_data("FLUX")`,
`ctx.dap_table("PM_KEL")`, `ctx.dap_np_columns([...])`, ... Import the module in
`lvmqa/metrics/__init__.py`, or load it at run time with `--metric-module path/to/file.py`.

### Notebooks

* `notebooks/dap_npfit_IC1613.ipynb`: investigation of `qa_dap_npfit` for IC 1613 (5351-5353):
  spectra with parametric / non-parametric line models, sensitivity to S/N threshold and line subset.




## List of scripts (outdated, used for DR20)

Each script produces a table of results with an `expnum` column, which can be used to join the outputs.

| Script              | Description                       | Input   | Output             |
| ------------------- | --------------------------------- | ------- | ------------------ |
| `photometry.py`     | for point 2 and 3 from the DRP QA | SFrames | photometry.csv     |
| `weakest_elines.py` | for point 6 from DRP QA          | DAP     | weakest_elines.csv |
| `weakest_continuum.py` | for Point 7 from DRP QA | SFrames | weakest_continuum.csv |
| `ew_telluric.py` | for Point 11 from DRP QA | SFrames | ew_telluric.csv |
| `drpall_compare.py` | stats and plots of drp_all file   | drpall  | drp summary plots  |
| `get_DAP_files.py`  | downloads DAP frames              | DR20 list | dap_file_list.txt  |
| `DR20_DAP_flux_comp.py` | for point 7 of DAP QA         | dap_file_list | fits tables and plots  |
| `DAPflux_comp_tools.py` | functions used in `DR20_DAP_flux_comp.py` |  |  |
| `DR20_DAP_flux_comp_plots.py` | generate plots from tables saved in `DR20_DAP_flux_comp.py` | fits tables | various plots |





## LVM DRP QA


1. Color/jumps in the continuum between arms : spectrophotometry stability.
2. Colors comparisons with existing datasets (Panstarrs/Gaia).
3. Absolute photometry comparison with existing datasets (GAIA).
4. S/N of the Ha 
5. S/N of continuum
6. Weakest detected emission line: data depths.
7. Weakest detected continuum level. 
8. Sky brightness during the observations.
9. Residual RMS / FLUX of a set of night-sky emission lines, after removal (RMS), vs before (FLUX)
10. Residual RMS/FLUX of the night-sky continuum. 
11. EW of the uncorrected (so far) Telluric Absorptions.
12. Night-sky extinction. 
13. Accuracy of the wavelength calibration: Delta_lambda from the night-sky spectra (lvmCFrame) fiber by fiber and in average for a set of prominent emission lines (covering the full wavelength range). 
14. Accuracy of the LOSVD, spectral resolution: Measure it in the night-sky lines.
15. Accuracy of the propagated error: RMS spectra-DAP model / propagated error, does it follow a Gaussian function? 



## LVM DAP QA

1. RMS_res / Flux for different emission lines (measured by DAP).
2. Compare the reported errors in the emission lines with a simple analysis of the residual spectra. 
3. Comparison with independent measurements: I.e., the DAP runs over the integrated spectra, that are 15,000. Run independent tools for these very same spectra and compare the results, for parametric and non-parametric analysis, or the emission lines.
4. Comparisons with the existing compilation of emission line maps. 
5. Stellar properties: Compare the DAP results with GAIA values (velocity? Stellar properties?). Compare with MWM derived quantities? Av along the line-of-sight? 
6. Analysis of the residuals for a subset of continuum ranges and the emission lines? Are the Chi-sq provided informative?
7. Parametric vs non-parametric derivations: The differences are driven by noise (how faint are the lines), the relative noise/contrast (EW of lines) and/or systematics (double lines)?

