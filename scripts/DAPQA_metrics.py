#!/usr/bin/env python3
'''
DAPQA_metrics.py

Author: Maren Cosens
Date: 2024-04-04
Description: 
    This script produces quality metrics for the LVM DAP emission line fitting. 
    1. It compares the parametric and non-parametric fluxes and saves flags based on the agreement between the two methods (00 - Good: differences <1sigma for >95% of fibers; 01 - Warning: excesses <3sigma for 5% of fibers (rest <1sigma); 10 - Bad: excesses in >5% of fibers; 11 - Undefined: not analyzed)
    2. It compares the residuals to the reported uncertainties on the Gaussian (parametric) fits (00 - Good: sum(residuals) <1sigma for >95% of fibers; 01- Warning: sum(residuals) <3sigma for >95% of fibers; 10 - Bad: sum(residuals) >3sigma for >5% of fibers; 11 - Undefined: not analyzed)
'''
##import packages
import os
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from dap_tools import read_DAP_file

c=2.99792e5 #[km/s]

##get DAP files in the DR20 file list
with open('dap_file_list.txt', 'r') as f:
    dap_files = f.read().splitlines()
with open('dap_file_list_mod.txt', 'r') as f:
    dap_mod_files = f.read().splitlines()

dap_files = [entry.replace('/lvm', 'sdsswork/lvm') for entry in dap_files]
dap_mod_files = [entry.replace('/lvm', 'sdsswork/lvm') for entry in dap_mod_files]
line_list='QAemission_lines.LVM'
tab_elines = Table.read(line_list, format='ascii.no_header', names=('wl', 'line'))

##initialize table with exposure numbers to store overall flux comparison stats
dr20_file_list = Table.read('dr20_pointings_final_v1.2.0.csv', format='csv')

flux_stat_table = Table()
flux_stat_table['expnum'] = dr20_file_list['expnum']
flux_stat_table['PM_NP_flag'] = np.full(len(flux_stat_table), '11') #initialize with '11' for undefined
flux_stat_table['resid_flag'] = np.full(len(flux_stat_table), '11')

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
        pm_flux = dap_tab[f'flux_pek_{wl}']
        pm_flux_err = dap_tab[f'e_flux_pek_{wl}']
        np_flux= dap_tab[f'flux_{line}_{wl}']
        np_flux_err = dap_tab[f'e_flux_{line}_{wl}']
        line_width = dap_tab[f'disp_pek_{wl}']
        line_cen = dap_tab[f'vel_pek_{wl}']
        return pm_flux.value, pm_flux_err.value, np_flux.value, np_flux_err.value, line_width.value, line_cen.value
    except KeyError:
        id=dap_tab['id']
        ret_arr = np.full(len(id), np.nan)
        return ret_arr, ret_arr, ret_arr, ret_arr, ret_arr, ret_arr

#change to function and call in batches with download of model files before call and deleting them after
def evaluate_DAP_flags(index, dap_file, dap_mod_file):
    i=index
    #files for DAP fits
    dap_file = dap_files[i]
    print(f'Comparing fluxes for file: {dap_file}')
    dap_data = read_DAP_file(dap_file)
    hdu=fits.open(dap_mod_file)
    model_spectra=hdu[0].data
    residual = model_spectra[0,:,:] - model_spectra[8,:,:] #data - joint model to get residuals
    hdr=hdu[0].header
    wave = hdr['crval1']+hdr['cdelt1']*(np.arange(0,model_spectra.shape[2])-(hdr['crpix1']-1))
    hdu.close()
    del hdu
    #set up arrays to store fluxes, errors, and residuals for each line and fiber
    flux_diff = np.full((len(tab_elines), len(dap_data)), np.nan)
    flux_diff_err = np.full((len(tab_elines), len(dap_data)), np.nan)
    pm_err = np.full((len(tab_elines), len(dap_data)), np.nan) 
    residual_rss = np.full((len(tab_elines), len(dap_data)), np.nan)
    n_entries = len(tab_elines)*len(dap_data)
    for j in range(len(tab_elines)):
        line, wl = tab_elines['line'][j], tab_elines['wl'][j]
        #get DAP fit parameters
        pm_flux, pm_flux_err, np_flux, np_flux_err, line_width, line_cen = get_dap_line_fit(line, wl, dap_data)
        flux_diff[j,:] = np.abs(pm_flux - np_flux)
        flux_diff_err[j,:] = np.sqrt(pm_flux_err**2 + np_flux_err**2)
        pm_err[j,:] = pm_flux_err
        #get residuals
        wcen=((line_cen/c) * wl) + wl #convert to A
        sigma_hwhm = np.sqrt(2*np.log(2)) #conversion factor from sigma to HWHM for Gaussian line width
        for k in range(len(pm_flux)): #loop through each fiber and get RSS of residuals over line FWHM
            chan_start, chan_end = np.argmin(np.abs(wave-(wcen[k]-sigma_hwhm*line_width[k]))), np.argmin(np.abs(wave-(wcen[k]+sigma_hwhm*line_width[k])))
            residual_rss[j,k] = np.sqrt(np.nansum(residual[k, chan_start:chan_end]**2))*1e16 #match units of DAP errors (1e-16 erg/s/cm^2)
    #flag PM vs NP flux comparison
    if np.sum(flux_diff < flux_diff_err) / n_entries > 0.95:
        flux_stat_table['PM_NP_flag'][i] = '00' #Good
    elif np.sum(flux_diff < 3*flux_diff_err) / n_entries > 0.95:
        flux_stat_table['PM_NP_flag'][i] = '01' #Warning
    else:
        flux_stat_table['PM_NP_flag'][i] = '10' #Bad
    #flag residuals vs uncertainties
    if np.sum(residual_rss < pm_err) / n_entries > 0.95:
        flux_stat_table['resid_flag'][i] = '00' #Good
    elif np.sum(residual_rss < 3*pm_err) / n_entries > 0.95:
        flux_stat_table['resid_flag'][i] = '01' #Warning
    else:
        flux_stat_table['resid_flag'][i] = '10' #Bad

nbatches=10
nfiles_batch=round(len(dap_files)/nbatches) #number of files per batch (except last batch will contain remainder)
for n in range(nbatches):
    mod_list=dap_mod_files[nfiles_batch*n:nfiles_batch*n+nfiles_batch]
    mod_list = [entry.replace('sdsswork/lvm', '/lvm') for entry in mod_list]
    #make list of these files, download
    with open('temp_file_list.txt', 'w') as f:
        for item in mod_list:
            f.write("%s\n" % item)
    os.system("rsync -avz --no-motd --files-from=temp_file_list.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork 2>temp_rsync-errors.txt")
    try: #if there are errors, try getting zipped version of the model
        f= open('temp_rsync-errors.txt', 'r')
        fail_list=list(f)
        new_list=[]
        for i in range(len(fail_list)):
            try:
                new_list.append(fail_list[i].split('"')[1].replace('.fits', '.fits.gz')) #replace with zipped version
            except:
                #do not add row to new list to avoid errors in subsequent steps 
                pass
        with open('temp_list_gz.txt', 'w') as f2: #write zipped versions to new file
            for item in new_list:
                f2.write("%s\n" % item)
        os.system("rsync -avz --no-motd --files-from=temp_list_gz.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork") #downlad zipped versions
        for f in new_list:
            os.system(f"gunzip sdsswork/{f}") #unzip locally
        os.system(f"rm temp_rsync-errors.txt") #remove temporary files
        os.system(f"rm temp_list_gz.txt")
    except:
        pass
    for i in range(nfiles_batch*n,nfiles_batch*n+nfiles_batch):
        try:
            evaluate_DAP_flags(i, dap_files[i], dap_mod_files[i])
            os.system(f"rm {dap_mod_files[i]}")
        except:
            pass
# delete list
os.system("rm temp_file_list.txt")

##save table with exposure number and flags
flux_stat_table.write('DAP_flux_comparison_flags.fits', overwrite=True)

##plot histogram of flags to check
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('PEK vs. NP flux')
plt.hist(flux_stat_table['PM_NP_flag'].astype('int'), bins=[0,0.5,1,1.5,10,10.5,11,11.5], align='left') #uneven bins for discrete flag options
plt.xlabel('Flag')
plt.ylabel('N exposures')
plt.xticks([0,1,10,11], ['Good', 'Warning', 'Bad', 'Undefined'], rotation=-45)
plt.xlim(-0.5, 11.5)
plt.subplot(1,2,2)
plt.title('Residuals vs. uncertainties')
plt.hist(flux_stat_table['resid_flag'].astype('int'), bins=[0,0.5,1,1.5,10,10.5,11,11.5], align='left')
plt.xlabel('Flag')
plt.ylabel('N exposures')
plt.xticks([0,1,10,11], ['Good', 'Warning', 'Bad', 'Undefined'], rotation=-45)
plt.xlim(-0.5, 11.5)
plt.tight_layout()
plt.savefig('DAP_flux_comparison_flags_histogram.png')
plt.close()