# lvmqa

Quality assurance tools for LVM. 



## List of scripts

Each script produces a table of results with an `expnum` column, which can be used to join the outputs.

| Script              | Description                       | Input   | Output             |
| ------------------- | --------------------------------- | ------- | ------------------ |
| `photometry.py`     | for point 2 and 3 from the DRP QA | SFrames | photometry.csv     |
| `weakest_elines.py` | for point 6 form DRP QA           | DAP     | weakest_elines.csv |
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
