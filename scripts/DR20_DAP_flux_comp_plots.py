#!/usr/bin/env python3
"""
DR20_DAP_flux_comp.py

Author: Maren Cosens
Date: 2024-03-24

Reads output FITS tables from DR20_DAP_flux_comp.py and reproduces frame, line, line-strength, and EW plots.

Description: reads in tables saved in DR20_DAP_flux_comp.py to generate overall diagnostic plots of interest. Each function below produces specific plots: 
    -plot_by_frame('dr20_pointings_finale-1.csv', 'DR20_pe_np_comparison_stats.fits') 
        'DR20_flux_comparison_by_frame.png': distribution of percent difference for each frame side by side (average over all lines/fibers) 
    -plot_by_line_strength('DR20_pe_np_comparison_by_line_strength.fits') 
        'DR20_flux_comparison_by_line_strength.png': overplot np/pek distribution per pek flux bin for every tile and print full data set medians and IQR 
    -plot_by_EW('DR20_pe_np_comparison_by_EW.fits') 
        'DR20_flux_comparison_by_EW.png': same as above but for bins of EW 
    -plot_by_snr('DR20_pe_np_comparison_by_snr.fits') 
        'DR20_flux_comparison_by_snr.png': same as above but for bins of EW 
    -plot_by_line_name('../lvmdap/_legacy/emission_lines_strong.LVM', 'dap_file_list.txt') 
        'DR20_flux_comparison_by_line.png': overplot np/pek distribution for each line for every tile plus full data set median and standard deviation 
"""
##import necessary libraries
import os
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def load_binned_array(stat_table):
    n_exps = len(stat_table)
    # Determine bins from table columns
    bin_low_cols = [col for col in stat_table.colnames if col.startswith('bin_') and col.endswith('_lower_edge')]
    bin_high_cols = [col for col in stat_table.colnames if col.startswith('bin_') and col.endswith('_upper_edge')]
    # Extract bin values from first row into a 1-d array
    bin_left_edges = np.array([stat_table[col][0] for col in bin_low_cols], dtype=float)
    bin_right_edges = np.array([stat_table[col][0] for col in bin_high_cols], dtype=float)
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    bin_widths=bin_right_edges-bin_left_edges
    if ((len(stat_table.columns)-1)/9)==5: #for line strength and EW tables with 1 set of bins
        arr = np.full((n_exps, 3, len(bin_left_edges)), np.nan)
        for i in range(len(bin_left_edges)):
            arr[:, 0, i] = stat_table[f'bin_{i}_n']
            arr[:, 1, i] = stat_table[f'bin_{i}_median']
            arr[:, 2, i] = stat_table[f'bin_{i}_std']
    elif ((len(stat_table.columns)-1)/9)==8: #for snr table with separate parametric and non-parametric S/N bins
        arr = np.full((n_exps, 6, len(bin_left_edges)), np.nan)
        for i in range(len(bin_left_edges)):
            arr[:, 0, i] = stat_table[f'pek_bin_{i}_n']
            arr[:, 1, i] = stat_table[f'pek_bin_{i}_median']
            arr[:, 2, i] = stat_table[f'pek_bin_{i}_std']
            arr[:, 3, i] = stat_table[f'np_bin_{i}_n']
            arr[:, 4, i] = stat_table[f'np_bin_{i}_median']
            arr[:, 5, i] = stat_table[f'np_bin_{i}_std']
    return bin_centers, bin_widths, arr

def plot_by_frame(file_list, table_file):
    dr20_file_list = Table.read(file_list, format='csv')
    flux_stat_table = Table.read(table_file, format='fits')

    plt.figure(figsize=(30, 6))
    inds = np.arange(len(dr20_file_list['expnum']))
    plt.errorbar(inds, flux_stat_table['median_perc_diff'], yerr=flux_stat_table['std_perc_diff'], fmt='ok',  label='frame distribution')
    plt.hlines(0, inds[0] - 1, inds[-1] + 1, color='grey', linestyle='--', alpha=0.5, zorder=0,)
    plt.xticks(range(len(inds)), [f'{expnum}' for expnum in dr20_file_list['expnum']], rotation=90)
    plt.ylim(-20, 20)
    plt.xlim(inds[0] - 1, inds[-1] + 1)
    plt.legend()
    plt.ylabel('Median Percent Difference (Non-parametric - Parametric) / Parametric')
    plt.savefig('DR20_flux_comparison_by_frame.png', bbox_inches='tight')
    plt.close()

def plot_by_line_strength(table_file):
    line_strength_table = Table.read(table_file, format='fits')
    bin_centers, bin_widths, line_strength_array = load_binned_array(line_strength_table)
    max_points=np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        max_points[i] = np.nanmax(line_strength_array[:, 0, i])
    #replace np.nan with 0 in number of points per bin to avoid issues with scaling alpha of ellipses
    bad_inds=np.where(np.isnan(line_strength_array[:,0,:]))
    line_strength_array[bad_inds[0],0,bad_inds[1]]=0

    '''
    #now using ellipse to show distribution within bins more clearly
    plt.figure(figsize=(12, 6))
    for i in range(line_strength_array.shape[0]):
        plt.scatter(bin_centers, line_strength_array[i, 1, :], marker='o', c='skyblue', facecolor=None, s=line_strength_array[i, 2, :], alpha=line_strength_array[i, 0, :] / np.nanmax(line_strength_array[:, 0, :]))
    '''
    fig, ax=plt.subplots(figsize=(12, 6))
    for i in range(line_strength_array.shape[0]):
        for j in range(line_strength_array.shape[2]):
            patch = Ellipse((bin_centers[j], line_strength_array[i,1,j]), width=0.7*bin_widths[j], height=line_strength_array[i,2,j], facecolor='skyblue', alpha=line_strength_array[i,0,j]/max_points[j])
            ax.add_artist(patch)
        plt.scatter(bin_centers, line_strength_array[i, 1, :], marker='.', color='deepskyblue')#, label='frame median')  
    #x_lims=plt.xlim()
    plt.hlines(1, 1, 1e6, color='grey', linestyle='--', alpha=0.5, zorder=0)
    for i in range(len(bin_centers)):
        quartile1, master_median, quartile3 = np.nanpercentile(line_strength_array[:, 1, i], [25, 50, 75])
        upper = quartile3 - master_median
        lower = master_median - quartile1
        plt.text(bin_centers[i], 10, s=f'${master_median:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$', horizontalalignment='center')
    plt.ylim(-20, 20)
    plt.xscale('log')
    plt.xlim(1,1e6)
    #dummy entry for legend
    plt.scatter(0,0, marker='o', c='skyblue', facecolor=None, s=50, alpha=0.5, label='frame distributions')
    plt.scatter(0,0, marker='o', c='deepskyblue', facecolor=None, s=10, alpha=0.5, label='frame medians')
    plt.legend()
    plt.xlabel('Parametric Flux Strength Bin (1e-16 erg/s/cm^2)')
    plt.ylabel('Median of Non-parametric/Parametric Flux')
    plt.savefig('DR20_flux_comparison_by_line_strength.png', bbox_inches='tight')
    plt.close()

def plot_by_EW(table_file):
    ew_table = Table.read(table_file, format='fits')
    # Load arrays for plotting
    bin_centers, bin_widths, ew_array = load_binned_array(ew_table)
    max_points=np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        max_points[i] = np.nanmax(ew_array[:, 0, i])
    #replace np.nan with 0 in number of points per bin to avoid issues with scaling alpha of ellipses
    bad_inds=np.where(np.isnan(ew_array[:,0,:]))
    ew_array[bad_inds[0],0,bad_inds[1]]=0

    '''
    #now using ellipse to show distribution within bins more clearly
    plt.figure(figsize=(12, 6))
    for i in range(len(ew_array)):
        plt.scatter(bin_centers, ew_array[i, 1, :], marker='o', c='skyblue', facecolor=None, s=ew_array[i, 2, :], alpha=ew_array[i, 0, :] / np.nanmax(ew_array[:, 0, :]))
    '''
    fig, ax=plt.subplots(figsize=(12, 6))
    for i in range(ew_array.shape[0]):
        for j in range(ew_array.shape[2]):
            patch = Ellipse((bin_centers[j], ew_array[i,1,j]), width=0.7*bin_widths[j], height=ew_array[i,2,j], facecolor='skyblue', alpha=ew_array[i,0,j]/max_points[j])
            ax.add_artist(patch)
        plt.scatter(bin_centers, ew_array[i, 1, :], marker='.', color='deepskyblue')#, label='frame median')  
    plt.errorbar(bin_centers, np.nanmedian(ew_array[:, 1, :], axis=0), yerr=np.nanstd(ew_array[:, 1, :], axis=0), fmt='ok', capsize=2, label='median distribution') #add error bars showing 1sigma range of medians for all lines
    for i in range(len(bin_centers)):
        quartile1, master_median, quartile3 = np.nanpercentile(ew_array[:, 1, i], [25, 50, 75])
        upper = quartile3 - master_median
        lower = master_median - quartile1
        plt.text(bin_centers[i], 10, s=f'${master_median:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$', horizontalalignment='center')
    #x_lims=plt.xlim()
    plt.hlines(1, 1, 1e7, color='grey', linestyle='--', alpha=0.5, zorder=0)
    plt.ylim(-20, 20)
    plt.xscale('log')
    plt.xlim(1, 1e7)
    #dummy entry for legend
    plt.scatter(0,0, marker='o', c='skyblue', facecolor=None, s=50, alpha=0.5, label='frame distributions')
    plt.scatter(0,0, marker='o', c='deepskyblue', facecolor=None, s=10, alpha=0.5, label='frame medians')
    plt.legend()
    plt.xlabel('Non-parametric Equivalent Width Bin (Å)')
    plt.ylabel('Median of Non-parametric/Parametric Flux')
    plt.savefig('DR20_flux_comparison_by_EW.png', bbox_inches='tight')
    plt.close()

def plot_by_snr(table_file):
    snr_table = Table.read(table_file, format='fits')
    # Load arrays for plotting
    bin_centers, bin_widths, snr_array = load_binned_array(snr_table)
    bin_width=bin_centers[1]-bin_centers[0]
    max_points_p=np.full(len(bin_centers), np.nan)
    max_points_np=np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        max_points_p[i] = np.nanmax(snr_array[:, 0, i])
        max_points_np[i] = np.nanmax(snr_array[:, 3, i])
    #replace np.nan with 0 in number of points per bin to avoid issues with scaling alpha of ellipses
    bad_inds=np.where(np.isnan(snr_array[:,0,:]))
    snr_array[bad_inds[0],0,bad_inds[1]]=0
    bad_inds2=np.where(np.isnan(snr_array[:,3,:]))
    snr_array[bad_inds2[0],3,bad_inds2[1]]=0

    fig, ax =plt.subplots(nrows=2, ncols=1,figsize=(12, 12))
    #plt.subplot(2,1,1)
    for i in range(snr_array.shape[0]):
        for j in range(snr_array.shape[2]):
            p_patch = Ellipse((bin_centers[j], snr_array[i,1,j]), width=0.7*bin_widths[j], height=snr_array[i,2,j], facecolor='skyblue', alpha=snr_array[i,0,j]/max_points_p[j])
            np_patch = Ellipse((bin_centers[j], snr_array[i,4,j]), width=0.7*bin_widths[j], height=snr_array[i,5,j], facecolor='skyblue', alpha=snr_array[i,3,j]/max_points_np[j])
            ax[0].add_artist(p_patch)
            ax[1].add_artist(np_patch)
        ax[0].scatter(bin_centers, snr_array[i, 1, :], marker='.', color='deepskyblue') #medians
        ax[1].scatter(bin_centers, snr_array[i, 4, :], marker='.', color='deepskyblue')
    ax[0].errorbar(bin_centers, np.nanmedian(snr_array[:, 1, :], axis=0), yerr=np.nanstd(snr_array[:, 1, :], axis=0), fmt='ok', capsize=2, label='median distribution') #add error bars showing 1sigma range of medians for all lines
    ax[1].errorbar(bin_centers, np.nanmedian(snr_array[:, 4, :], axis=0), yerr=np.nanstd(snr_array[:, 4, :], axis=0), fmt='ok', capsize=2)
    x_lims=ax[0].get_xlim()
    ax[0].hlines(1, x_lims[0], x_lims[1], color='grey', linestyle='--', alpha=0.5, zorder=0)
    ax[1].hlines(1, x_lims[0], x_lims[1], color='grey', linestyle='--', alpha=0.5, zorder=0) #reference line
    #print bin stats
    for i in range(len(bin_centers)):
        p_quartile1, p_master_median, p_quartile3 = np.nanpercentile(snr_array[:, 1, i], [25, 50, 75])
        upper = p_quartile3 - p_master_median
        lower = p_master_median - p_quartile1
        ax[0].text(bin_centers[i], 1.2, s=f'${p_master_median:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$', horizontalalignment='center')
        np_quartile1, np_master_median, np_quartile3 = np.nanpercentile(snr_array[:, 4, i], [25, 50, 75])
        upper = np_quartile3 - np_master_median
        lower = np_master_median - np_quartile1
        ax[1].text(bin_centers[i], 1.2, s=f'${np_master_median:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}$', horizontalalignment='center')
    ax[0].set_ylim(0.25,1.5)
    ax[0].set_xlim(x_lims[0], x_lims[1])
    ax[0].set_xlabel('Parametric Fit S/N')
    ax[1].set_ylabel('Median of Non-Parametric/Parametric Flux')
    ax[1].set_ylim(0.25,1.5)
    ax[1].set_xlim(x_lims[0], x_lims[1])
    ax[1].set_xlabel('Non-Parametric Fit S/N')
    ax[1].set_ylabel('Median of Non-Parametric/Parametric Flux')
    #dummy entries for legend
    ax[0].scatter(0,-1000, marker='o', c='skyblue', facecolor=None, s=50, alpha=0.5, label='frame distributions')
    ax[0].scatter(0,-1000, marker='o', c='deepskyblue', facecolor=None, s=10, alpha=0.5, label='frame medians')
    ax[0].legend() 
    plt.savefig('DR20_flux_comparison_by_snr.png', bbox_inches='tight')
    plt.close()

def plot_by_line_name(line_list_file, file_list_file):
    tab_elines=tab_elines = Table.read(line_list_file, format='ascii.no_header', names=('wl', 'line'))
    inds= np.arange(len(tab_elines))
    with open(file_list_file, 'r') as f:
        file_list = f.read().splitlines()
    #generate plot, read in results from each DAP file, and overplot individual median and IQR for each line
    #read results from each DAP file and add to arrays below (needs to be separate step to scale alpha of each ellipse by n_points/max_points per line)
    medians_all = np.full((len(file_list), len(tab_elines)), np.nan)
    centers_all = np.full((len(file_list), len(tab_elines)), np.nan)
    IQR_all = np.full((len(file_list), len(tab_elines)), np.nan)
    n_points_all = np.full((len(file_list), len(tab_elines)), np.nan)
    for i in range(len(file_list)):
        file=file_list[i].replace('/lvm', 'sdsswork/lvm').replace('.dap.fits', '_flux_comparison_stats.fits')
        if os.path.exists(file):
            ratio_tab = Table.read(file_list[i].replace('/lvm', 'sdsswork/lvm').replace('.dap.fits', '_flux_comparison_stats.fits'), format='fits')
            quartile1, medians_all[i,:], quartile3, n_points_all[i,:] = ratio_tab['quartile1'], ratio_tab['median'], ratio_tab['quartile3'], ratio_tab['n_good']
            centers_all[i,:]=(quartile1+quartile3) / 2
            IQR_all[i,:] = quartile3 - quartile1
    max_points = np.nanmax(n_points_all, axis=0) #currently all are 1754 (fit either fails for all or none of a line which seems unrealistic)
    #generate plot
    fig, ax=plt.subplots(figsize=(12, 6))
    for i in range(len(file_list)):
        file=file_list[i].replace('/lvm', 'sdsswork/lvm').replace('.dap.fits', '_flux_comparison_stats.fits')
        if os.path.exists(file):
            for j in range(len(tab_elines)):
                patch = Ellipse((inds[j], centers_all[i,j]), width=0.7, height=IQR_all[i,j], facecolor='skyblue', alpha=0.2, label='IQR')#n_points_all[i,j]/max_points[j])
                ax.add_artist(patch)
        plt.scatter(inds, medians_all[i,:], marker='.', color='deepskyblue', label='frame median')
    plt.hlines(1, inds[0]-1, inds[-1]+1, color='grey', linestyle='--', alpha=0.5, zorder=0) #add horizontal line at ratio=1 for reference
    plt.errorbar(inds, np.nanmedian(medians_all, axis=0), yerr=np.nanstd(medians_all, axis=0), fmt='ok', capsize=2, label='median distribution') #add error bars showing 1sigma range of medians for all lines
    plt.ylabel('Non-Parametric Flux / Parametric Flux')
    plt.xticks(range(len(tab_elines)), [f'{line}_{wl}' for line, wl in zip(tab_elines['line'], tab_elines['wl'])], rotation=90)
    plt.ylim(-20,20)
    plt.xlim(inds[0]-1, inds[-1]+1)
    #generate legend without duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('DR20_flux_comparison_by_line.png', bbox_inches='tight')
    plt.close()
        
if __name__ == '__main__':
    # Example usage: update file list / line list as needed
    # Uncomment and adjust these for your workflow
    plot_by_frame('dr20_pointings_final_v1.2.0.csv', 'DR20_pe_np_comparison_stats.fits')
    plot_by_line_strength('DR20_pe_np_comparison_by_line_strength.fits')
    plot_by_EW('DR20_pe_np_comparison_by_EW.fits')
    plot_by_snr('DR20_pe_np_comparison_by_snr.fits')
    plot_by_line_name('../lvmdap/_legacy/emission_lines_strong.LVM', 'dap_file_list.txt')

##bin plots show standard deviation, but print IQR which is much smaller and less sensitive to outliers; may want to plot this as well?