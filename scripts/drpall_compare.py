#!/usr/bin/env python
# coding: utf-8

'''
                    Space Telescope Science Institute

Synopsis:

Create some summary plots and statistics from
a drp_all file


Command line usage (if any):

    usage: drpall_compare.py [-h] [-outroot name] filename

Description:

Primary routines:

    doit

Notes:

History:

260326 ksl Coding begun

'''

import sys
from astropy.io import ascii,fits
from astropy.table import Table,join
import numpy as np
import matplotlib.pyplot as plt
import os




def drp_stats(filename='drpall-1.2.0.fits'):
    """Print summary statistics for a drpall FITS table.

    Prints the total number of exposures, the number with exposure time
    >= 900 s, the MJD and exposure-number ranges, and a breakdown of
    flux-calibration types (``MOD``, ``STD``, ``SCI``, ``NONE``).

    :param filename: Path to the drpall FITS table.
    :type filename: str
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return
    ztab=xtab[xtab['exptime']>=899]
    print('There are %6d (%6d>=900s) exposures in the file' % (len(xtab),len(ztab)))
    print('The data spans MJDs      : %8d to %8d' % (np.min(xtab['mjd']),np.max(xtab['mjd'])))
    print('The data spans exposures : %8d to %8d' % (np.min(xtab['expnum']),np.max(xtab['expnum'])))
    # print(np.unique(xtab['drpqual'], return_counts=True))
    try:
        values,counts=np.unique(xtab['fluxcal'], return_counts=True)
        values900,counts900=np.unique(ztab['fluxcal'], return_counts=True)
        i=0
        print('Calibration Type, Total Number, Number for 900 s exposures')
        while i<len(values):
            print(    '%10s        %5d        (%5d)' % (values[i],counts[i],counts900[i]))
            i+=1
    except:
        print('There is no fluxcal entry in this file')




def calibration_used(filename,outroot='drpall'):
    """Plot the flux-calibration method used for each exposure as a function of MJD.

    Exposures are colour-coded by calibration type (``MOD``, ``STD``,
    ``SCI``, ``NONE``) and plotted at slightly offset y-levels so that
    all four categories are visible simultaneously.  The figure is saved
    to ``Plot_drpall/<outroot>_mjd_cal_method.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.  Defaults to
        ``'drpall'``; when called from :func:`steer` this is derived
        from the input filename.
    :type outroot: str
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return



    xmod=xtab[xtab['fluxcal']=='MOD']
    xstd=xtab[xtab['fluxcal']=='STD']
    xsci=xtab[xtab['fluxcal']=='SCI']
    xnone=xtab[xtab['fluxcal']=='NONE']
    xmod['xlevel']=1
    xstd['xlevel']=1.1
    xsci['xlevel']=1.2
    xnone['xlevel']=1.3
    plt.close(1)
    plt.figure(1,(12,6))
    plt.clf()

    plt.plot(xmod['mjd'],xmod['xlevel'],'.',label='Mod')
    plt.plot(xstd['mjd'],xstd['xlevel'],'.',label='Std')
    plt.plot(xsci['mjd'],xsci['xlevel'],'.',label='Sci')
    plt.plot(xnone['mjd'],xnone['xlevel'],'.',label='None')
    plt.ylim(0.9,1.4)
    plt.legend()
    plt.suptitle(filename)
    plt.savefig('Plot_drpall/%s_mjd_cal_method.png' % outroot)




def plot_sensitivities(filename,outroot='drpall'):
    """Plot the mean sensitivity curves and their differences over time.

    Produces a four-panel figure (one panel per quantity) for exposures
    calibrated with the ``MOD`` method:

    * Panel 1: ``mod_mean_sen{b,r,z}`` vs MJD.
    * Panel 2: ``std_mean_sen{b,r,z}`` vs MJD.
    * Panel 3: ``mod_mean_sen`` − ``std_mean_sen`` vs MJD (B, R, Z).
    * Panel 4: ``mod_mean_sen`` − ``sci_mean_sen`` vs MJD (B, R, Z).

    The figure is saved to ``Plot_drpall/<outroot>_mjd_senssitivity.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.
    :type outroot: str
    """
    try:
        qtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return


    print(np.median(qtab['mod_mean_senb']))
    mod_tab=qtab[qtab['fluxcal']=='MOD']
    plt.close(1)
    plt.figure(1,(12,12))
    plt.clf()
    plt.subplot(4,1,1)
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senb'],label='mod_mean_senb')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senr'],label='mod_mean_senr')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senz'],label='mod_mean_senz')
    plt.legend()
    plt.ylim(1e-14,1e-13)
    plt.subplot(4,1,2)
    plt.plot(mod_tab['mjd'],mod_tab['std_mean_senb'],label='std_mean_senb')
    plt.plot(mod_tab['mjd'],mod_tab['std_mean_senr'],label='std_mean_senr')
    plt.plot(mod_tab['mjd'],mod_tab['std_mean_senz'],label='std_mean_senz')
    plt.legend()
    plt.ylim(1e-14,1e-13)
    plt.subplot(4,1,3)
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senb']-mod_tab['std_mean_senb'],label='delta b')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senr']-mod_tab['std_mean_senr'],label='delta r')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senz']-mod_tab['std_mean_senz'],label='delta z')
    plt.legend()
    plt.ylim(-1e-14,1e-14)
    plt.subplot(4,1,4)
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senb']-mod_tab['sci_mean_senb'],label='delta b')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senr']-mod_tab['sci_mean_senr'],label='delta r')
    plt.plot(mod_tab['mjd'],mod_tab['mod_mean_senz']-mod_tab['sci_mean_senz'],label='delta z')
    plt.legend()
    plt.ylim(-1e-13,1e-13)
    plt.suptitle(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plot_drpall/%s_mjd_senssitivity.png' % outroot)



def compare_sensitivity_methods(filename='drpall-1.2.1.fits',outroot='drpall'):
    """Plot histograms of the sensitivity difference between calibration methods.

    For exposures with exposure time >= 900 s, compares ``MOD`` vs ``STD``
    and ``MOD`` vs ``SCI`` sensitivities for each of the B, R, and Z bands
    as log-scale normalised histograms.  The figure is saved to
    ``Plot_drpall/<outroot>_calibration_method.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.
    :type outroot: str
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return
    ztab=xtab[xtab['exptime']>=899]
    plt.close(1)
    plt.figure(1,(12,8))
    plt.clf()
    plt.subplot(3,2,1)
    foo=plt.hist(ztab['mod_mean_senb']-ztab['std_mean_senb'],1000,range=(-3e-15,3e-15),log=True,density=True,label='B (Mod-Std)')
    plt.xlabel(r'$\Delta$ Sensitivity')
    plt.legend()
    plt.subplot(3,2,2)
    foo=plt.hist(ztab['mod_mean_senb']-ztab['sci_mean_senb'],1000,range=(-3e-14,3e-14),log=True,density=True,label='B (Mod-Sci)')
    plt.xlabel(r'$\Delta$ Sensitivity')
    plt.legend()
    plt.subplot(3,2,3)
    foo=plt.hist(ztab['mod_mean_senr']-ztab['std_mean_senr'],1000,range=(-3e-15,3e-15),log=True,density=True,label='R (Mod-Std)')
    plt.xlabel(r'$\Delta$ Sensitivity')
    plt.legend()
    plt.subplot(3,2,4)
    foo=plt.hist(ztab['mod_mean_senr']-ztab['sci_mean_senr'],1000,range=(-3e-14,3e-14),log=True,density=True,label='R (Mod-Sci)')
    plt.legend()
    plt.subplot(3,2,5)

    foo=plt.hist(ztab['mod_mean_senz']-ztab['std_mean_senz'],1000,range=(-3e-15,3e-15),log=True,density=True,label='Z (Mod-Std)')
    plt.xlabel(r'$\Delta$ Sensitivity')
    plt.legend()
    plt.subplot(3,2,6)
    foo=plt.hist(ztab['mod_mean_senz']-ztab['sci_mean_senz'],1000,range=(-3e-14,3e-14),log=True,density=True,label='Z (Mod-Sci)')
    plt.xlabel(r'$\Delta$ Sensitivity')
    plt.legend()
    plt.suptitle(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plot_drpall/%s_calibration_method.png' % outroot)




def compare_sensitivity_ratio(filename='drpall-1.2.1.fits',outroot='drpall'):
    """Plot histograms of the sensitivity ratio between calibration methods.

    For exposures with exposure time >= 900 s, compares ``STD:MOD`` and
    ``SCI:MOD`` sensitivity ratios for each of the B, R, and Z bands as
    log-scale normalised histograms.  The figure is saved to
    ``Plot_drpall/<outroot>_sensitivity_ratio.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.
    :type outroot: str
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return
    ztab=xtab[xtab['exptime']>=899]
    plt.close(1)
    plt.figure(1,(12,8))
    plt.clf()
    plt.subplot(3,2,1)
    foo=plt.hist(ztab['std_mean_senb']/ztab['mod_mean_senb'],1000,range=(0.9,1.1),log=True,density=True,label='B (Std:Mod)')
    plt.xlabel(r'Sensitivity Ratio')
    plt.xlim(0.9,1.1)
    plt.legend()
    plt.subplot(3,2,2)
    foo=plt.hist(ztab['sci_mean_senb']/ztab['mod_mean_senb'],1000,range=(0.5,1.5),log=True,density=True,label='B (Sci:Mod)')
    plt.xlabel(r'Sensitivity Ratio')
    plt.xlim(0.5,1.5)
    plt.legend()
    plt.subplot(3,2,3)
    foo=plt.hist(ztab['std_mean_senr']/ztab['mod_mean_senr'],1000,range=(0.9,1.1),log=True,density=True,label='R (Std:Mod)')
    plt.xlabel(r'Sensitivity Ratio')
    plt.xlim(0.9,1.1)
    plt.legend()
    plt.subplot(3,2,4)
    foo=plt.hist(ztab['sci_mean_senr']/ztab['mod_mean_senr'],1000,range=(0.5,1.5),log=True,density=True,label='R (Sci:Mod)')
    plt.xlim(0.5,1.5)
    plt.legend()
    plt.subplot(3,2,5)

    foo=plt.hist(ztab['std_mean_senz']/ztab['mod_mean_senz'],1000,range=(0.9,1.1),log=True,density=True,label='Z (Std:Mod)')
    plt.xlabel(r'Sensitivity Ratio')
    plt.xlim(0.9,1.1)
    plt.legend()
    plt.subplot(3,2,6)
    foo=plt.hist(ztab['sci_mean_senz']/ztab['mod_mean_senz'],1000,range=(0.5,1.5),log=True,density=True,label='Z (Sci:Mod)')
    plt.xlabel(r'Sensitivity Ratio')
    plt.xlim(0.5,1.5)
    plt.legend()
    plt.suptitle(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plot_drpall/%s_sensitivity_ratio.png' % outroot)




def mjd_stability(filename,outroot='drpall',nexp=5):
    """Plot night-to-night stability of the MOD sensitivity as absolute scatter.

    For each MJD with at least ``nexp`` exposures, computes the standard
    deviation of ``mod_mean_sen{b,r,z}`` within that night.  The results
    are displayed as inverse-cumulative histograms (fraction of nights
    with scatter less than a given value) for B, R, and Z bands.  A
    histogram of the per-night exposure counts is shown in the first
    panel.  The figure is saved to
    ``Plot_drpall/<outroot>_mjd_stability.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.
    :type outroot: str
    :param nexp: Minimum number of exposures per MJD required for a
        night to be included in the stability statistics.  Defaults to 5.
    :type nexp: int
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return


    xtab=xtab[xtab['exptime']>=899]
    mjd,counts=np.unique(xtab['mjd'],return_counts=True)
    plt.close(1)
    plt.figure(1,(12,6))
    plt.clf()
    plt.subplot(1,4,1)
    plt.hist(counts)
    delta_b=[]
    delta_r=[]
    delta_z=[]
    i=0
    while i<len(mjd):
        if counts[i]>=nexp:
            xfoo=xtab[xtab['mjd']==mjd[[i]]]
            delta_b.append(np.std(xfoo['mod_mean_senb']))
            delta_r.append(np.std(xfoo['mod_mean_senr']))
            delta_z.append(np.std(xfoo['mod_mean_senz']))
        i+=1
    plt.subplot(1,4,2)
    plt.hist(delta_b,1000,range=(0,4e-14),cumulative=-1,density=True,log=True,label='B')
    plt.ylabel('Frac < STD(sens)')
    plt.xlabel('STD(sens)')
    plt.xlim(0,4e-14)
    plt.legend()

    plt.subplot(1,4,3)
    plt.hist(delta_r,1000,range=(0,4e-14),cumulative=-1,density=True,log=True,label='R')
    plt.ylabel('Frac < STD(sens)')
    plt.xlabel('STD(sens)')
    plt.xlim(0,4e-14)
    plt.legend()
    plt.subplot(1,4,4)
    plt.hist(delta_z,1000,range=(0,4e-14),cumulative=-1,density=True,log=True,label='Z')
    plt.ylabel('Frac < STD(sens)')
    plt.xlabel('STD(sens)')
    plt.xlim(0,4e-14)
    plt.legend()

    plt.suptitle(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plot_drpall/%s_mjd_stability.png' % outroot)



def mjd_stability_ratio(filename,outroot='drpall',nexp=5):
    """Plot night-to-night stability of the MOD sensitivity as fractional scatter.

    Identical in structure to :func:`mjd_stability` but uses the
    coefficient of variation (``STD / mean``) rather than the absolute
    standard deviation, so the y-axis is dimensionless and comparable
    across bands.  The figure is saved to
    ``Plot_drpall/<outroot>_mjd_stability_ratio.png``.

    :param filename: Path to the drpall FITS table.
    :type filename: str
    :param outroot: Prefix for the output filename.
    :type outroot: str
    :param nexp: Minimum number of exposures per MJD required for a
        night to be included in the stability statistics.  Defaults to 5.
    :type nexp: int
    """
    try:
        xtab=Table.read(filename)
    except:
        print('Could not read: ',filename)
        return

    xtab=xtab[xtab['exptime']>=899]
    mjd,counts=np.unique(xtab['mjd'],return_counts=True)

    plt.close(1)
    plt.figure(1,(12,6))
    plt.clf()
    plt.subplot(1,4,1)
    plt.hist(counts)
    delta_b=[]
    delta_r=[]
    delta_z=[]
    i=0
    while i<len(mjd):
        if counts[i]>=nexp:
            xfoo=xtab[xtab['mjd']==mjd[[i]]]
            delta_b.append(np.std(xfoo['mod_mean_senb'])/np.average(xfoo['mod_mean_senb']))
            delta_r.append(np.std(xfoo['mod_mean_senr'])/np.average(xfoo['mod_mean_senr']))
            delta_z.append(np.std(xfoo['mod_mean_senz'])/np.average(xfoo['mod_mean_senz']))
        i+=1
    plt.subplot(1,4,2)
    plt.hist(delta_b,1000,range=(0,1),cumulative=-1,density=True,log=True,label='B')
    plt.ylabel('Frac < STD/Mdan(sens)')
    plt.xlabel('STD/Mean (sens)')
    plt.xlim(0,1)
    plt.legend()

    plt.subplot(1,4,3)
    plt.hist(delta_r,1000,range=(0,1),cumulative=-1,density=True,log=True,label='R')
    plt.ylabel('Frac < STD/Mean(sens)')
    plt.xlabel('STD/Mean (sens)')
    plt.xlim(0,1)
    plt.legend()
    plt.subplot(1,4,4)
    plt.hist(delta_z,1000,range=(0,1),cumulative=-1,density=True,log=True,label='Z')
    plt.ylabel('Frac < STD/Mean (sens)')
    plt.xlabel('STD/Mean(sens)')
    plt.xlim(0,1)
    plt.legend()

    plt.suptitle(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('Plot_drpall/%s_mjd_stability_ratio.png' % outroot)




def steer(argv):
    """Parse command-line arguments and run all analysis routines.

    Reads a drpall FITS table and produces the full set of summary plots
    and statistics by calling :func:`drp_stats`, :func:`calibration_used`,
    :func:`plot_sensitivities`, :func:`compare_sensitivity_methods`,
    :func:`compare_sensitivity_ratio`, :func:`mjd_stability`, and
    :func:`mjd_stability_ratio`.

    All figures are written to the ``Plot_drpall/`` directory, which is
    created if it does not already exist.

    :param argv: Command-line argument list (typically ``sys.argv``).
        Recognised options:

        ``filename``
            Path to the drpall FITS table to analyse (required).

        ``-outroot name``
            Prefix prepended to every output figure filename.  If
            omitted, the prefix is derived from *filename* by stripping
            the directory path and the ``.fits`` extension
            (e.g. ``path/to/drpall-1.2.1.fits`` → ``drpall-1.2.1``).

        ``-h``
            Print the module docstring and exit.

    :type argv: list
    """

    drp_file=''
    outroot=''

    i=1
    while i<len(argv):
        if argv[i][:2]=='-h':
            print(__doc__)
            return
        elif argv[i]=='-outroot':
            i+=1
            outroot=argv[i]
        elif argv[i][0]=='-':
            print('Error: Could not intepret commands: ',argv)
            return
        else:
            drp_file=argv[i]
        i+=1

    if outroot=='':
        outroot=drp_file.split('/')[-1].replace('.fits','')

    print('Examining: %s' % drp_file)
    print('Outroot  : %s' % outroot)

    os.makedirs('Plot_drpall',exist_ok=True)
    drp_stats(filename=drp_file)
    calibration_used(filename=drp_file,outroot=outroot)
    plot_sensitivities(filename=drp_file,outroot=outroot)
    compare_sensitivity_methods(filename=drp_file,outroot=outroot)
    compare_sensitivity_ratio(filename=drp_file,outroot=outroot)
    mjd_stability(filename=drp_file,outroot=outroot)
    mjd_stability_ratio(filename=drp_file,outroot=outroot)




# Next lines permit one to run the routine from the command line
if __name__ == "__main__":
    import sys
    if len(sys.argv)>1:
        steer(sys.argv)
    else:
        print (__doc__)
