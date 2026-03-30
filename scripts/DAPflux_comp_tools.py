'''
DAPflux_comp_tools.py

Author: Maren Cosens
Date: 2024-03-20

Description:
    This script contains functions to compare the DAP fluxes from the parametric and non-parametric fits for a specified file. It generates plots to visualize the differences between the two sets of fluxes and identifies any significant discrepancies. The script also calculates statistical metrics to quantify the agreement between the two methods.
'''
##import necessary libraries
from astropy.table import Table
import numpy as np
from dap_tools import read_DAP_file
from matplotlib import pyplot as plt

def get_DAP_vals(dap_file, line_file, EW=False, snr=False):
    """Read DAP file and emission-line list, return specified parameters from parametric and non-parametric fits. Default to fluxes only; if EW=True also return EW of non-parametric fit. If snr=True also return S/N of parametric flux"""
    dap_data = read_DAP_file(dap_file)
    tab_elines = Table.read(line_file, format='ascii.no_header', names=('wl', 'line'))
    n_fibers = len(dap_data['id'])
    p_fluxes = np.empty((n_fibers, len(tab_elines)))
    np_fluxes = np.empty((n_fibers, len(tab_elines)))
    if EW:
        ew=np.empty((n_fibers, len(tab_elines)))
    if snr:
        s_n=np.empty((n_fibers, len(tab_elines)))
        s_n_np=np.empty((n_fibers, len(tab_elines)))
    for i in range(len(tab_elines)):
        try:
            p_fluxes[:, i] = dap_data[f'flux_pek_{tab_elines["wl"][i]}']
            np_fluxes[:, i] = dap_data[f'flux_{tab_elines["line"][i]}_{tab_elines["wl"][i]}']
            if snr:
                s_n[:,i] = p_fluxes[:, i]/dap_data[f'e_flux_pek_{tab_elines["wl"][i]}']
                s_n_np[:,i] = np_fluxes[:, i]/dap_data[f'e_flux_{tab_elines["line"][i]}_{tab_elines["wl"][i]}']
        except KeyError:
            p_fluxes[:, i] = np.full(n_fibers, np.nan)
            np_fluxes[:, i] = np.full(n_fibers, np.nan)
            if snr:
                s_n[:,i] = np.full(n_fibers, np.nan)
                s_n_np[:,i] = np.full(n_fibers, np.nan)
        if EW:
            ew[:,i] = dap_data[f'EW_{tab_elines["line"][i]}_{tab_elines["wl"][i]}']
    arr_return = [tab_elines, p_fluxes, np_fluxes]
    if EW:
        arr_return.append(ew)
    if snr:
        arr_return.append(s_n)
        arr_return.append(s_n_np)
    return arr_return


def compare_by_line(dap_file, line_file):
    '''
    This function compares the DAP fluxes from the parametric and non-parametric fits for a specified file. It generates plots to visualize the differences between the two sets of fluxes and identifies any significant discrepancies. The function also calculates statistical metrics to quantify the agreement between the two methods.
    
    Parameters:
    dap_file (str): The path to the DAP file to be analyzed.
    line_file (str): The path to the table of emission lines to evaluate, which should be a tab-separated file with columns 'wl' and 'line'.
    
    Returns:
    flux_stats: A list containing the median difference and standard deviation of the flux differences.
    Saves violin plots comparing the fluxes for each emission line to the DAP file directory and a FITS table with the calculated statistics for this exposure.
    '''
    ##retrieve DAP and line flux arrays via helper function
    tab_elines, p_fluxes, np_fluxes = get_DAP_vals(dap_file, line_file)
    ##generate violin plots to compare fluxes
    ratios = np_fluxes / p_fluxes
    inds=np.arange(len(tab_elines))
    quartile1, medians, quartile3 = np.percentile(ratios, [25, 50, 75], axis=0)
    n_points=np.nansum(np.isfinite(ratios), axis=0)
    plt.figure(figsize=(12, 6))
    plt.violinplot(ratios, positions=inds, widths=0.75, showmeans=False, showmedians=False,
        showextrema=False)
    plt.scatter(inds, medians, marker='o', color='deepskyblue', s=10, zorder=3, label='median')
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3, label='IQR')
    plt.hlines(1, inds[0]-1, inds[-1]+1, color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.xticks(range(len(tab_elines)), [f'{line}_{wl}' for line, wl in zip(tab_elines['line'], tab_elines['wl'])], rotation=90)
    plt.ylim(-20,20)
    plt.xlim(inds[0]-1, inds[-1]+1)
    plt.legend()
    #save plot to DAP file directory
    plt.savefig(dap_file.replace('.dap.fits', '_flux_comparison.png'), bbox_inches='tight')
    plt.close()
    
    ##save individual line ratio statistics to a fits table for this exposure
    lines_str=[f'{line}_{wl}' for line, wl in zip(tab_elines['line'], tab_elines['wl'])]
    ratio_tab=Table([lines_str, quartile1, medians, quartile3, n_points], names=['line', 'quartile1', 'median', 'quartile3', 'n_good'])
    ratio_tab.write(dap_file.replace('.dap.fits', '_flux_comparison_stats.fits'), overwrite=True)

    ##calculate overall statistical metrics
    median_percent_diff = np.nanmedian((p_fluxes - np_fluxes) / p_fluxes)*100
    std_diff = np.nanstd((p_fluxes - np_fluxes) / p_fluxes)*100

    print(f'Median percent difference between fluxes: {median_percent_diff:.2f}%')
    print(f'Standard deviation of percent difference: {std_diff:.2e}%')
    return [median_percent_diff, std_diff, np.nansum(n_points)]

def compare_by_line_strength(dap_file, line_file, bins):
    '''
    This function compares the DAP fluxes from the parametric and non-parametric fits for a specified file, grouping by line strength. It generates plots to visualize the differences between the two sets of fluxes and calculates statistical metrics to quantify the agreement between the two methods for flux bins.
    
    Parameters:
    dap_file (str): The path to the DAP file to be analyzed.
    line_file (str): The path to the table of emission lines to evaluate, which should be a tab-separated file with columns 'wl' and 'line'.
    bins (array-like): The edges of the bins to use for grouping by parametric flux strength. Should be in the same units as the DAP fluxes (1e-16 erg/s/cm^2).
    
    Returns:
    An array containing the number of data points (fibers*lines), median, and standard deviation of the flux differences for each line strength bin.
    Saves plots comparing the fluxes for each emission line strength bin to the DAP file directory.
    '''
    ##retrieve DAP and line flux arrays via helper function
    tab_elines, p_fluxes, np_fluxes = get_DAP_vals(dap_file, line_file)
    ratios= np_fluxes / p_fluxes
    
    ##calculate median and standard deviation of the flux ratio in bins of parametric flux
    n_points= np.histogram(p_fluxes.flatten(), bins=bins)[0]
    medians= np.array([np.nanmedian(ratios.flatten()[(p_fluxes.flatten() > bins[i-1]) & (p_fluxes.flatten() <= bins[i])]) for i in range(1, len(bins))])
    stds= np.array([np.nanstd(ratios.flatten()[(p_fluxes.flatten() > bins[i-1]) & (p_fluxes.flatten() <= bins[i])]) for i in range(1, len(bins))])

    #generate figure with individual values and bin median/std
    plt.figure(figsize=(8, 6))
    plt.plot(p_fluxes.flatten(), ratios.flatten(), '.', c='skyblue', alpha=0.1)
    plt.errorbar(0.5*(bins[1:]+bins[:-1]), medians, yerr=stds, fmt='ok')
    plt.hlines(1, plt.xlim()[0], plt.xlim()[1], color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.xlabel('Parametric Flux [1e-16 erg/s/cm^2]')
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.ylim(-20,20)
    plt.xscale('log')
    plt.xlim(1, plt.xlim()[1])
    plt.savefig(dap_file.replace('.dap.fits', '_flux_comparison_by_strength.png'), bbox_inches='tight')
    plt.close()

    ##return statistics as array for this exposure
    return [n_points, medians, stds]


def compare_by_line_EW(dap_file, line_file, bins):
    '''
    This function compares the DAP fluxes from the parametric and non-parametric fits for a specified file, grouping by line equivalent width (EW). It generates plots to visualize the differences between the two sets of fluxes and calculates statistical metrics to quantify the agreement between the two methods for EW bins.
    
    Parameters:
    dap_file (str): The path to the DAP file to be analyzed.
    line_file (str): The path to the table of emission lines to evaluate, which should be a tab-separated file with columns 'wl' and 'line'.
    bins (array-like): The edges of the bins to use for grouping by non-parametric EW. Should be in the same units as the DAP EWs (Angstroms).
    
    Returns:
    An array containing the number of data points (fibers*lines), median, and standard deviation of the flux differences for each EW bin.
    Saves plots comparing the fluxes for each emission line EW bin to the DAP file directory.
    '''
    ##retrieve DAP and line flux arrays via helper function
    tab_elines, p_fluxes, np_fluxes, ew_vals = get_DAP_vals(dap_file, line_file, EW=True)
    ratios= np_fluxes / p_fluxes
    ew_vals= -1*ew_vals #multiply by -1 so that stronger lines have higher EW values for easier visualization (since EWs are negative in DAP for emission lines)

    ##calculate median and standard deviation of the flux ratio in bins of no-parametric EW
    n_points= np.histogram(p_fluxes.flatten(), bins=bins)[0]
    medians= np.array([np.nanmedian(ratios.flatten()[(p_fluxes.flatten() > bins[i-1]) & (p_fluxes.flatten() <= bins[i])]) for i in range(1, len(bins))])
    stds= np.array([np.nanstd(ratios.flatten()[(p_fluxes.flatten() > bins[i-1]) & (p_fluxes.flatten() <= bins[i])]) for i in range(1, len(bins))])

    #generate figure with individual values and bin median/std
    plt.figure(figsize=(8, 6))
    plt.plot(ew_vals.flatten(), ratios.flatten(), '.', c='skyblue', alpha=0.1)
    plt.errorbar(0.5*(bins[1:]+bins[:-1]), medians, yerr=stds, fmt='ok')
    plt.hlines(1, plt.xlim()[0], plt.xlim()[1], color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.xlabel('Equivalent Width [$\\AA$]')
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.ylim(-10,10)
    plt.xscale('log')
    plt.xlim(1, plt.xlim()[1])
    plt.savefig(dap_file.replace('.dap.fits', '_flux_comparison_byEW.png'), bbox_inches='tight')
    plt.close()

    ##return statistics as array for this exposure
    return [n_points, medians, stds]

def compare_by_line_snr(dap_file, line_file, bins):
    '''
    This function compares the DAP fluxes from the parametric and non-parametric fits for a specified file, grouping by parametric or non-parametricline fit signal-to-noise. It generates plots to visualize the differences between the two sets of fluxes and calculates statistical metrics to quantify the agreement between the two methods for S/N bins.
    
    Parameters:
    dap_file (str): The path to the DAP file to be analyzed.
    line_file (str): The path to the table of emission lines to evaluate, which should be a tab-separated file with columns 'wl' and 'line'.
    bins (array-like): The edges of the bins to use for grouping by parametric flux S/N. Should be unitless.
    
    Returns:
    An array containing the number of data points (fibers*lines), median, and standard deviation of the flux differences for each S/N bin (x2 bins for parametric and non-parametric S/N).
    Saves plots comparing the fluxes for each emission S/N bin to the DAP file directory.
    '''
    ##retrieve DAP and line flux arrays via helper function
    tab_elines, p_fluxes, np_fluxes, p_snr, np_snr = get_DAP_vals(dap_file, line_file, snr=True) #add an S/N flag to this
    ratios= np_fluxes / p_fluxes
    
    ##calculate median and standard deviation of the flux ratio in bins of parametric flux
    n_points_p= np.histogram(p_snr.flatten(), bins=bins)[0]
    medians_p= np.array([np.nanmedian(ratios.flatten()[(p_snr.flatten() > bins[i-1]) & (p_snr.flatten() <= bins[i])]) for i in range(1, len(bins))])
    stds_p= np.array([np.nanstd(ratios.flatten()[(p_snr.flatten() > bins[i-1]) & (p_snr.flatten() <= bins[i])]) for i in range(1, len(bins))])
    ##calculate median and standard deviation of the flux ratio in bins of non-parametric flux
    n_points_np= np.histogram(np_snr.flatten(), bins=bins)[0]
    medians_np= np.array([np.nanmedian(ratios.flatten()[(np_snr.flatten() > bins[i-1]) & (np_snr.flatten() <= bins[i])]) for i in range(1, len(bins))])
    stds_np= np.array([np.nanstd(ratios.flatten()[(np_snr.flatten() > bins[i-1]) & (np_snr.flatten() <= bins[i])]) for i in range(1, len(bins))])

    #generate figure with individual values and bin median/std
    plt.figure(figsize=(8, 12))
    plt.subplot(2,1,1)
    plt.plot(p_snr.flatten(), ratios.flatten(), '.', c='skyblue', alpha=0.1)
    plt.errorbar(0.5*(bins[1:]+bins[:-1]), medians_p, yerr=stds_p, fmt='ok')
    plt.hlines(1, -5, 200, color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.xlabel('Parametric Fit S/N')
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.ylim(-10,10)
    plt.xlim(-5, 200)
    plt.subplot(2,1,2)
    plt.plot(np_snr.flatten(), ratios.flatten(), '.', c='skyblue', alpha=0.1)
    plt.errorbar(0.5*(bins[1:]+bins[:-1]), medians_np, yerr=stds_np, fmt='ok')
    plt.hlines(1, -5, 200, color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.xlabel('Non-Parametric Fit S/N')
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.ylim(-10,10)
    plt.xlim(-5, 200)
    plt.tight_layout()
    plt.savefig(dap_file.replace('.dap.fits', '_flux_comparison_by_snr.png'), bbox_inches='tight')
    plt.close()

    ##return statistics as array for this exposure
    return [n_points_p, medians_p, stds_p, n_points_np, medians_np, stds_np]