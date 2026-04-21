#!/usr/bin/env python3
'''
DR20_DAP_flux_comp.py

Author: Maren Cosens
Date: 2024-03-20
Description: 
    This script uses functions in DAPflux_comp_tools.py to compare the DAP fluxes from the parametric and non-parametric fits for a sample of DR20 tiles. These functions generate plots to visualize the differences between the two sets of fluxes and identifies any significant discrepancies in individual tiles while returning values used to look for overall trends here. Results are stored in FITS tables for later access.

    Saves 3 summary tables of diagnostics: 
        -DR20_pe_np_comparison_stats.fits: 
            Stats on difference between fits for each frame. Values from function compare_by_line()
            Columns: 
                expnum: exposure number 
                mean_perc_diff: median of the percent difference (pek-np)/pek for all line fits evaluated 
                std_perc_diff: standard deviation of percent difference 
                n_good: number of data points evaluated for this tile (fibers*lines) 
                
        -DR20_pe_np_comparison_by_line_strength.fits
            Stats on np/pek flux ratio for defined bins of pek flux. Values from compare_by_line_strength()
            Columns:
                expnum: exposure number
                bin_{n}_lower_edge: lower bound of bin n
                bin_{n}_upper_edge: upper bound of bin n
                bin_n_n: number of points in bin n (fibers*lines)
                bin_{n}_median: median of np/pek ratios in bin n
                bin_{n}_std:  standard deviation of np/pek ratios in bin n 
        -DR20_pe_np_comparison_by_line_EW.fits
            Same as above but for bins of np EW. Values from compare_by_line_EW() 
        -DR20_pe_np_comparison_by_line_snr.fits 
            Same as above but for bins of pek S/N. Values from compare_by_line_snr() 
'''
##import necessary libraries
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from dap_tools import read_DAP_file
import DAPflux_comp_tools as comp_tools

##get DAP files in the DR20 file list
with open('dap_file_list.txt', 'r') as f:
    dap_files = f.read().splitlines()

dap_files = [entry.replace('/lvm', 'sdsswork/lvm') for entry in dap_files]

##initialize table with exposure numbers to store overall flux comparison stats
dr20_file_list = Table.read('dr20_pointings_final_v1.2.0.csv', format='csv')
flux_stat_table = Table()
flux_stat_table['expnum'] = dr20_file_list['expnum']
flux_stat_table['median_perc_diff'] = np.full(len(flux_stat_table), np.nan)
flux_stat_table['std_perc_diff'] = np.full(len(flux_stat_table), np.nan)
flux_stat_table['n_good'] = np.full(len(flux_stat_table), np.nan)

##initialize array to store flux comparison stats by line strength for each bin
line_strength_bins = np.logspace(0, 6, num=10, endpoint=True)
line_strength_array= np.full((len(dap_files), 3, len(line_strength_bins)-1), np.nan) #3 is for n_points, median, std in each bin

##initialize array to store flux comparison stats by EW for each bin
ew_bins = np.logspace(0, 7, num=10, endpoint=True)
ew_array= np.full((len(dap_files), 3, len(ew_bins)-1), np.nan) #3 is for n_points, median, std in each bin

##initialize array to store flux comparison stats by line S/N for each bin
snr_bins = np.linspace(-5, 200, num=10, endpoint=True)
snr_array= np.full((len(dap_files), 6, len(snr_bins)-1), np.nan) #6 is for n_points, median, std in each bin x2 for parametric and non-parametric S/N bins

##loop through DAP files and extract fluxes for comparison
for i in range(len(dap_files)):
    dap_file = dap_files[i]
    print(f'Comparing fluxes for file: {dap_file}')
    #comparison by emission line (individual line comparisons saved at location of of 'dap_file', average compiled saved here)
    flux_stats = comp_tools.compare_by_line(dap_file, '../lvmdap/_legacy/emission_lines_strong.LVM')
    flux_stat_table['median_perc_diff'][i] = flux_stats[0]
    flux_stat_table['std_perc_diff'][i] = flux_stats[1]
    flux_stat_table['n_good'][i] = flux_stats[2]
    #comparison by parametric flux strength
    line_strength_array[i,:,:] = comp_tools.compare_by_line_strength(dap_file, '../lvmdap/_legacy/emission_lines_strong.LVM', bins=line_strength_bins)
    #comparison by non-parametric EW
    ew_array[i,:,:] = comp_tools.compare_by_line_EW(dap_file, '../lvmdap/_legacy/emission_lines_strong.LVM', bins=ew_bins)
    #comparison by parametric fit S/N
    snr_array[i,:,:] = comp_tools.compare_by_line_snr(dap_file, '../lvmdap/_legacy/emission_lines_strong.LVM', bins=snr_bins)

##write overall flux statistics to a FITS table
flux_stat_table.write('DR20_pe_np_comparison_stats.fits', overwrite=True) #saved to fits extension 1

##write line strength and EW binned statistics to FITS tables
#make line strength table from array
line_strength_stat_table = Table()
line_strength_stat_table['expnum'] = dr20_file_list['expnum']
for i in range(len(line_strength_bins)-1):
    line_strength_stat_table[f'bin_{i}_lower_edge'] = line_strength_bins[i]
    line_strength_stat_table[f'bin_{i}_upper_edge'] = line_strength_bins[i+1]
    line_strength_stat_table[f'bin_{i}_n'] = line_strength_array[:,0,i]
    line_strength_stat_table[f'bin_{i}_median'] = line_strength_array[:,1,i]
    line_strength_stat_table[f'bin_{i}_std'] = line_strength_array[:,2,i]
#save to fits
line_strength_stat_table.write('DR20_pe_np_comparison_by_line_strength.fits', overwrite=True) #saved to fits extension 1
#make EW table from array
ew_stat_table = Table()
ew_stat_table['expnum'] = dr20_file_list['expnum']
for i in range(len(ew_bins)-1):
    ew_stat_table[f'bin_{i}_lower_edge'] = ew_bins[i]
    ew_stat_table[f'bin_{i}_upper_edge'] = ew_bins[i+1]
    ew_stat_table[f'bin_{i}_n'] = ew_array[:,0,i]
    ew_stat_table[f'bin_{i}_median'] = ew_array[:,1,i]
    ew_stat_table[f'bin_{i}_std'] = ew_array[:,2,i]
#save to fits
ew_stat_table.write('DR20_pe_np_comparison_by_EW.fits', overwrite=True) #saved to fits extension 1

#make S/N table from array
snr_stat_table = Table()
snr_stat_table['expnum'] = dr20_file_list['expnum']
for i in range(len(snr_bins)-1):
    snr_stat_table[f'bin_{i}_lower_edge'] = snr_bins[i]
    snr_stat_table[f'bin_{i}_upper_edge'] = snr_bins[i+1]
    snr_stat_table[f'pek_bin_{i}_n'] = snr_array[:,0,i]
    snr_stat_table[f'pek_bin_{i}_median'] = snr_array[:,1,i]
    snr_stat_table[f'pek_bin_{i}_std'] = snr_array[:,2,i]
    snr_stat_table[f'np_bin_{i}_n'] = snr_array[:,3,i]
    snr_stat_table[f'np_bin_{i}_median'] = snr_array[:,4,i]
    snr_stat_table[f'np_bin_{i}_std'] = snr_array[:,5,i]

#save to fits
snr_stat_table.write('DR20_pe_np_comparison_by_snr.fits', overwrite=True) #saved to fits extension 1

print('Data processing complete: saved DR20_pe_np_comparison_stats.fits, DR20_pe_np_comparison_by_line_strength.fits, DR20_pe_np_comparison_by_EW.fits, DR20_pe_np_comparison_by_snr.fits')

##identify exposures that are outliers from median percent difference across all lines
outlier_exposures = flux_stat_table[flux_stat_table['median_perc_diff'] > np.nanmedian(flux_stat_table['median_perc_diff']) + 1 * np.nanstd(flux_stat_table['median_perc_diff'])]
print('Outlier exposures with median percent difference > 1 sigma from median (>20% difference):')
print(outlier_exposures['expnum'].data)
#updated list [12183 14114 15660 20486 20573 20980 22929 30171 30813 30868 31087 31090] # NGC4945, outskirts of Rossette
outlier_exposures2 = flux_stat_table[flux_stat_table['median_perc_diff'] > np.nanmedian(flux_stat_table['median_perc_diff']) + 10]
print('Outlier exposures with median percent difference > 10% :')
print(outlier_exposures2['expnum'].data)
#[ 5511  5512 12183 14114 15660 20486 20573 20980 22929 30171 30173 30197 30198 30199 30758 30759 30813 30868 30870 31087 31090] #M33, NGC4945, outskirts of Rossette