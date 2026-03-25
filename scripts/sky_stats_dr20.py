#code to go through all LVM data and create a table of info+stats for sky quality checks

#copied imports from sky_qa and SummarizeSframe

import sys
from glob import glob
import os
from astropy.io import ascii, fits
from astropy.io.fits import Header
import numpy as np
import subprocess
import matplotlib.pyplot as plt
#import xhtml
from matplotlib.gridspec import GridSpec
from astropy.wcs import WCS
from astropy.coordinates import get_body, solar_system_ephemeris, get_sun, AltAz, EarthLocation, get_moon, SkyCoord
from astropy.time import Time
import astropy.units as u
#import quick_map
#import eval_standard
from astropy.table import Table, vstack, hstack
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.backends.backend_pdf

from astropy.table import join, Table
import shutil
from datetime import datetime
from astropy.wcs import WCS

#Things that should not change and dir

XTOP='/uufs/chpc.utah.edu/common/home/sdss51/'

def read_rssfile(filename):
    try:
        x=fits.open(filename)
    except Exception:
        print('Error: eval_qual: Could not open %s' % filename)
        return   
    slitmap=Table(x['SLITMAP'].data)
    wav=x['WAVE'].data
    flux=x['FLUX'].data
    sky=x['SKY'].data
    ivar=x['IVAR'].data
    mask=x['MASK'].data
    header=x[0].header
    x.close()
    return wav, flux, sky, ivar, mask, slitmap, header

def get_med(flux, sky, ivar, mask, slitmap,):
    
    xgood=selfib_good(slitmap, 'sci')
    flux_sci=flux[xgood,:]
    sky_sci=sky[xgood,:]
    ivar_sci=ivar[xgood,:]
    mask_sci=mask[xgood,:]

    xgood_skye=selfib_good(slitmap, 'skye')
    skye_flux=flux[xgood_skye,:]
    skye_sky=sky[xgood_skye,:]
    skye_mask=mask[xgood_skye,:]
    xgood_skyw=selfib_good(slitmap, 'skyw')
    skyw_flux=flux[xgood_skyw,:]
    skyw_sky=sky[xgood_skyw,:]
    skyw_mask=mask[xgood_skyw,:]

    flux_sci=np.ma.masked_array(flux_sci, mask_sci)
    sky_sci=np.ma.masked_array(sky_sci, mask_sci)
    ivar_sci=np.ma.masked_array(ivar_sci, mask_sci)
    skye_flux=np.ma.masked_array(skye_flux, skye_mask)
    skyw_flux=np.ma.masked_array(skyw_flux, skyw_mask) 
    skye_sky=np.ma.masked_array(skye_sky, skye_mask)
    skyw_sky=np.ma.masked_array(skyw_sky, skyw_mask)

    med_sky=np.ma.median(sky_sci,axis=0)
    med_flux=np.ma.median(flux_sci,axis=0)
    med_ivar=np.ma.median(ivar_sci,axis=0)*len(xgood)/(1.253**2)
    med_skye=np.ma.median(skye_flux+skye_sky,axis=0)
    med_skyw=np.ma.median(skyw_flux+skyw_sky,axis=0)
     
    return med_flux, med_sky, med_ivar, med_skye, med_skyw

def selfib_good(slitmap,type):
    selfib = slitmap['fibstatus']==0
    if type == 'sci':
        seltype = slitmap['targettype']=='science'
    elif type == 'skye':
        seltype = (slitmap['targettype'] =='SKY') & (slitmap['telescope'] == 'SkyE')
    elif type == 'skyw':
        seltype = (slitmap['targettype'] =='SKY') & (slitmap['telescope'] == 'SkyW')
    else:
        print('wrong type, must be sci, skye, or skyw')
    sel = seltype*selfib
    return sel

# CHANGE so it reads the DR20 file, same structure as drpall
def read_drpall():
    DRPFILE = 'dr20_pointings_final_v1.2.0.csv'
    # First try to locate the DRP file locally, otherwise
    if os.path.isfile(DRPFILE):
        xfile=DRPFILE
    else:
        print('Error: Could not locate : ', xfile)
        return []

    try:
        drpall=ascii.read(xfile)
    except Exception:
        print('Error: Located but could not read  : ', xfile)
        return []

    drp_tab=Table(drpall)
    return  drp_tab


def find_top():
    if os.path.isdir(XTOP):
        loc='Utah'
        topdir=XTOP
    else:
        print('Error: I donot know where I am:', os.pwd())
        return ''

    print('We are on : ', loc)
    return topdir

def make_med_spec(xtab,data_dir,outfile=''):
    i=0
    xfiles=[]
    while i < len(xtab):
        xfile='%s/%s' % (data_dir,xtab['location'][i])
        if os.path.isfile(xfile):
            xfiles.append(xfile)
        i+=1
    print('There are %d files to process' % (len(xfiles)))

    # print(xfiles)
    i=0
    xsci_flux=[]
    xsci_sky=[]
    xsci_ivar=[]
    xskye_flux=[]
    xskyw_flux=[]
    

    while i<len(xfiles):
        #wav,sci_flux,sci_sky,sci_var=get_med_spec(xfiles[i]) #change to my codes
        wave, flux, sky, ivar, mask, slitmap, header = read_rssfile(xfiles[i])
        med_flux, med_sky, med_ivar, med_skye, med_skyw = get_med(flux, sky, ivar, mask, slitmap)
        xsci_flux.append(med_flux)
        xsci_sky.append(med_sky)
        xsci_ivar.append(med_ivar)
        xskye_flux.append(med_skye)
        xskyw_flux.append(med_skyw)
       
        #add the stats
        if i%10==0:
            print('Finished %d of %d' % (i,len(xfiles)))
                                          
        i+=1
    
   
    wav=np.array(wave)
    xsci_flux=np.array(xsci_flux)
    xsci_sky=np.array(xsci_sky)
    xsci_ivar=np.array(xsci_ivar)
    xskye_flux=np.array(xskye_flux)
    xskyw_flux=np.array(xskyw_flux)
    hdu1=fits.PrimaryHDU(data=None)
    hdu1.header['Title'] = 'SFrame_Summary for D20 tiles'
    hdu2=fits.ImageHDU(data=wav,name='WAVE')
    hdu3=fits.ImageHDU(data=xsci_flux,name='FLUX')
    hdu4=fits.ImageHDU(data=xsci_sky,name='SKY')
    hdu5=fits.ImageHDU(data=xsci_ivar,name='IVAR')
    hdu6=fits.ImageHDU(data=xskye_flux,name='SKYE_FLUX')
    hdu7=fits.ImageHDU(data=xskyw_flux, name='SKYW_FLUX')
    hdu8=fits.BinTableHDU(xtab, name='drp_all')

    wmin=wav[0]
    dwave=0.5

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1, 1]  # Reference pixel (1-based index)
    wcs.wcs.crval = [wmin, 0]  # Coordinates at the reference pixel: minimum wavelength and line number 0
    wcs.wcs.cdelt = [dwave, 1]  # Pixel scale: 0.5 Angstroms per pixel in wavelength, 1 per pixel in line number
    wcs.wcs.ctype = ['WAVE', 'LINE']  # Coordinate types: wavelength and line number

    hdu3.header.update(wcs.to_header())
    hdu4.header.update(wcs.to_header())    
    hdu5.header.update(wcs.to_header())
    hdu6.header.update(wcs.to_header())
    hdu7.header.update(wcs.to_header())
       
    hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8])

    if outfile=='':
        outfile='test.fits'
    else:
        if outfile.count('.fits')==0:
            outfile=outfile+'.fits'
    hdul.writeto(outfile,overwrite=True)
    print('Wrote results to %s' % outfile)
    return


def doit(out_name=''):
    xtop=find_top() #not sure I need this
    drptab=read_drpall() #reads in the DRP fits file, not h5, need UTAH sas path
        
    if out_name=='':
        out_name='XSframe_DR20.fits'
    make_med_spec(xtab=drptab,data_dir=xtop,outfile=out_name)



# Next lines permit one to run the routine from the command line
if __name__ == "__main__":
    doit()
