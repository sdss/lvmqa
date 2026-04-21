#!/usr/bin/env python3
'''
DAP_flux_errors.py
Author: Maren Cosens
Date: 2024-04-15
Description: 
This script compares reported flux errors from the DAP with residual spectra to evaluate the accuracy of the DAP flux error estimates. It calculates the ratio of the residual flux to the reported flux error for each spaxel and emission line, and analyzes the distribution of these ratios to identify any systematic trends or discrepancies in the DAP error estimates.
'''
##import packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from dap_tools import read_DAP_file

c=2.99792e5 #[km/s]

##get DAP table and model files in the DR20 file list
with open('dap_file_list_test.txt', 'r') as f:
    dap_files = f.read().splitlines()
with open('dap_file_list_mod_test.txt', 'r') as f:
    dap_mod_files = f.read().splitlines()

dap_files = [entry.replace('/lvm', 'sdsswork/lvm') for entry in dap_files]
dap_mod_files = [entry.replace('/lvm', 'sdsswork/lvm') for entry in dap_mod_files] #may need to run on Binder hub since these files take a lot of space
#test with subset

##table of emission lines to evaluate
tab_elines = Table.read('QAemission_lines.LVM', format='ascii.no_header', names=('wl', 'line'))

##initialize table with exposure numbers to store overall flux comparison stats
dr20_file_list = Table.read('dr20_pointings_final_v1.2.0.csv', format='csv')

#function to get DAP line flux, error, and width for a given line and DAP file
def get_dap_line_fit(line, wl, dap_tab):
    '''get DAP parametric line flux, error, center and width for a given line and DAP file
    Parameters:
    line (str): name of emission line (e.g. 'Halpha')
    wl (float): wavelength of emission line
    dap_file (str): path to DAP file
    Returns:
    tuple: line flux, line flux error, line width (A), and line center (A)
    '''
    try:
        line_flux = dap_tab[f'flux_pek_{wl}']
        line_flux_err = dap_tab[f'e_flux_pek_{wl}']
        line_width = dap_tab[f'disp_pek_{wl}']
        line_cen = dap_tab[f'vel_pek_{wl}']
        return line_flux.value, line_flux_err.value, line_width.value, line_cen.value
    except KeyError:
        id=dap_tab['id']
        ret_arr = np.full(len(id), np.nan)
        return ret_arr, ret_arr, ret_arr, ret_arr
    

#generate table to store results
flux_err_table = Table()
#flux_err_table['expnum'] = dr20_file_list['expnum']
for line, wl in zip(tab_elines['line'], tab_elines['wl']):
    flux_err_table.add_columns([np.full(len(dap_files), np.nan), np.full(len(dap_files), np.nan), np.full(len(dap_files), np.nan), np.full(len(dap_files), np.nan), np.full(len(dap_files), np.nan)], names=[f'{line}_{wl}_flux', f'{line}_{wl}_flux_err', f'{line}_{wl}_width', f'{line}_{wl}_cen', f'{line}_{wl}_resid_rms'])
flux_err_table.add_column(np.full(len(dap_files), np.nan), name='average_error_to_resid_ratio')

all_fluxes = []
all_snr = []
all_ratios = []
#get DAP fluxes and errors plus line widths
for i in range(len(dap_files)):
    dap_tab = read_DAP_file(dap_files[i]) #fit results
    #residual from model
    hdu=fits.open(dap_mod_files[i])
    model_spectra=hdu[0].data
    residual = model_spectra[0,:,:] - model_spectra[1,:,:] #data - model to get residuals
    hdr=hdu[0].header
    wave = hdr['crval1']+hdr['cdelt1']*(np.arange(0,model_spectra.shape[2])-(hdr['crpix1']-1))
    file_errs=[]
    file_rms=[]
    for j in range(len(tab_elines)):
        line = tab_elines['line'][j]
        wl = tab_elines['wl'][j]
        #fit results
        flux, flux_err, width, cen = get_dap_line_fit(line, wl, dap_tab)
        file_errs.append(flux_err)
        all_fluxes.append(flux)
        all_snr.append(flux/flux_err)
        #residuals
        wcen=((cen/c) * wl) + wl #convert to A
        residual_rms=np.full(len(flux), np.nan)
        for k in range(len(flux)): #loop through each fiber
            chan_start, chan_end = np.argmin(np.abs(wave-(wcen[k]-3*width[k]))), np.argmin(np.abs(wave-(wcen[k]+3*width[k])))
            residual_rms[k] = np.sqrt(np.nanmean(residual[k, chan_start:chan_end]**2))
        file_rms.append(residual_rms)
        all_ratios.append((flux_err*1e-16)/residual_rms)
        flux_err_table[f'{line}_{wl}_flux'][i] = np.nanmean(flux)
        flux_err_table[f'{line}_{wl}_flux_err'][i] = np.nanmean(flux_err)
        flux_err_table[f'{line}_{wl}_width'][i] = np.nanmean(width) #[A]
        flux_err_table[f'{line}_{wl}_cen'][i] = np.nanmean(cen) #[km/s]
        flux_err_table[f'{line}_{wl}_resid_rms'][i] = np.nanmean(residual_rms)
    del hdu #save memory by deleting hdu after use
    flux_err_table['average_error_to_resid_ratio'][i] = np.nanmean(np.array(file_errs)*1e-16/np.array(file_rms))
all_fluxes = np.array(all_fluxes)
all_snr = np.array(all_snr)
all_ratios = np.array(all_ratios)

flux_err_table.write('DAP_flux_error_comparison.fits', overwrite=True)

##Generate comparison plots
#plot by line
inds=np.arange(len(tab_elines))
plt.figure(figsize=(12, 6))
for i in inds:
    line = tab_elines['line'][i]
    wl = tab_elines['wl'][i]
    quartile1, median, quartile3 = np.percentile(all_ratios[i,:], [25, 50, 75], axis=0)
    n_points=np.nansum(np.isfinite(all_ratios[i,:]), axis=0)
    plt.violinplot(all_ratios[i,:], positions=[i], widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    plt.scatter(i, median, marker='o', color='deepskyblue', s=10, zorder=3, label='median')
    plt.vlines(i, quartile1, quartile3, color='k', linestyle='-', lw=3, label='IQR')
plt.hlines(1, inds[0]-1, inds[-1]+1, color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
plt.ylabel('Reported Error / Residual')
plt.xticks(inds, [f'{line}_{wl}' for line, wl in zip(tab_elines['line'], tab_elines['wl'])], rotation=90)
plt.ylim(-20,20)
plt.xlim(inds[0]-1, inds[-1]+1)
#generate legend without duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.savefig('DR20_error_comparison.png', bbox_inches='tight')
plt.close()

#plot by line flux
plt.figure(figsize=(8, 6))
plt.plot(all_fluxes.flatten(), all_ratios.flatten(), '.', c='skyblue', alpha=0.1)
plt.hlines(1, plt.xlim()[0], plt.xlim()[1], color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
plt.xlabel('Parametric Flux [1e-16 erg/s/cm^2]')
plt.ylabel('Reported Error / Residual')
plt.ylim(-5,5)
plt.xscale('log')
plt.xlim(1, 1e8)
plt.savefig('DR20_error_comparison_by_flux.png', bbox_inches='tight')
plt.close()

#plot by DAP reported S/N
plt.figure(figsize=(8, 6))
plt.plot(all_snr.flatten(), all_ratios.flatten(), '.', c='skyblue', alpha=0.1)
plt.hlines(1, plt.xlim()[0], plt.xlim()[1], color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
plt.xlabel('Parametric Flux S/N')
plt.ylabel('Reported Error / Residual')
plt.ylim(-5,5)
plt.xlim(0, plt.xlim()[1])
plt.savefig('DR20_error_comparison_by_snr.png', bbox_inches='tight')
plt.close()

#report stats:
print('Overall median ratio of reported error to residual:', np.nanmedian(all_ratios))
for i in range(len(tab_elines)):
    line = tab_elines['line'][i]
    wl = tab_elines['wl'][i]
    median_ratio = np.nanmedian(flux_err_table[f'{line}_{wl}_flux_err']*1e-16/flux_err_table[f'{line}_{wl}_resid_rms'])
    print(f'Median ratio for {line}_{wl}: {median_ratio}')