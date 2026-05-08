""" 
Measure the equivalent width for telluric absorption lines

This script addresses point 11 of the LVM DRP QA. The reuslt is saved
to a csv table. It contains the following columns:

expnum                    - exposure number of the pointing.
fibers                    - list of fiber IDs used in the measurement
n_fibers                  - number of fibers used
ew_X                      - equivalent width in angstrom
ew_X_abs                  - equivalent width of the absolute flux 
ew_X_std                  - uncertainty estimated from continuum scatter
  
We measure the EW for the following three lines in the specified 
range and compare it to the flux from the surrounding continuum:

 - 6868.5 between (6866.5, 6877.5) and continuum in (6818,6918)
 - 7604.0 between (7594.5, 7613.5) and continuum in (7504,7554)
 - 8228.0 between (8225.5, 8235.0) and continuum in (8178,8278)

See also this list of telluric lines:
https://www2.keck.hawaii.edu/inst/common/makeewww/Atmosphere/atmabs.txt

Because most fibers lack enough continuum flux to detect the absorption, 
we restrict the measurement to fibers that overlap with a star. The 
combined stellar catalogue used for fiber selection was assembled from
Hipparcos 

```
from astroquery.vizier import Vizier
vizier = Vizier()
vizier.ROW_LIMIT = 3e6
tycho2 = vizier.get_catalogs("I/280B")[0]
tycho2.meta.clear()
tycho2.write("tycho2.fits")
```

and the Gaia archive with ADQL (https://gea.esac.esa.int/archive/)

```
SELECT
    source_id,
    ra,
    dec,
    phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < 11
``` 
"""

from pathlib import Path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import tqdm 

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits 
from astropy.table import Table 
import astropy.units as u


# define the position of the line and the continuum
line_ranges = [(6866.5, 6877.5),(7594.5, 7613.5),(8225.5, 8235.0)]
continuum_ranges = [(6818,6918),(7504,7554),(8178,8278)]

# catalogue with coordinates of stars
stars = Table.read("../data/stars.csv")
stars["SkyCoord"] = SkyCoord(ra=stars["ra"],dec=stars["dec"],unit=u.degree)


def select_fibers_with_stars(header,sci_index,stars):
    """ 
    Parameters
    ----------
    header : fits.header
        Header of the fits file with coordinates.
    sci_index : astropy.table.Table
        Table of the fibers with coordinates.
    stars : list of astropy.coordinates.SkyCoord
        List of star coordiantes.
        
    Returns
    -------
    slitmap : np.ndarray
        Mask for sci_index to select fibers that contain a star.
    """
    
    # we first create a subsample of nearby stars to the tile 
    separation = SkyCoord(ra=header['POSCIRA'], dec=header['POSCIDE'], unit=u.deg).separation(stars)
    nearby_stars = stars[(separation<0.3*u.deg)]

    # select science fibers with status good
    science_fibers = sci_index[(sci_index["targettype"]=="science") & (sci_index["fibstatus"]==0)]

    # match stars to fibers. we have to remove some stars that fall 
    # outside the tile and do not overlap
    idx, dist, _ = match_coordinates_sky( 
        matchcoord = nearby_stars,
        catalogcoord = SkyCoord(ra=science_fibers["ra"],dec=science_fibers["dec"],unit=u.degree)
        )
    idx = idx[dist<0.5*u.arcmin]

    # create the slitmap to select the valid fibers
    slitmap = np.zeros(len(sci_index),dtype=bool)
    slitmap[science_fibers[idx]["fiberid"]-1] = True

    return slitmap


def measure_ew(cube,wlam,slitmap,continuum_range,line_range):
    """
    For each fiber selected with slitmap, we normalize the spectrum with
    the medium inside continuum_range. Then the median of all spectra is
    computed and the equivalent width is measured from this spectrum.
    
    
    Parameters
    ----------
    cube : np.ndarray of shape (m,n)
        Datacube with m fibers, each containing n spectral pixels.
    wlam : np.ndarray of shape (n,)
        Array with the wavelengths.
    slitmap : np.ndarray
        Mask to select fibers that contain a star.
    continuum_range : tuple
        Lower and upper limit between which the continuum is measured.
    line_range : tuple
        Lower and upper limit between which the absorption line is measured.
    
    Returns
    -------
    ew : float
        Equivalent width of the absorption line
    ew_abs : float
        Equivalent width of the absolute of the absorption line
    ew_std : float
        Uncertainty of the equivalent width based on std of the continuum.
    """
    
    # define masks to select line and continuum
    line_mask = (line_range[0] < wlam) & (wlam<line_range[1])
    continuum_mask = (continuum_range[0]<wlam) & (wlam<continuum_range[1])
    
    spectra = []
    for spectrum in cube[slitmap,:]:
        if np.all(np.isnan(spectrum[continuum_mask & ~line_mask])):
            continue
        median = np.nanmedian(spectrum[continuum_mask & ~line_mask])
        if median < 0:
            continue
        spectra.append(spectrum / median)
    
    if len(spectra) == 0:
        return np.nan, np.nan, np.nan
    
    spectra = np.array(spectra)                       

    line_spectrum = np.nanmedian(spectra[:, line_mask], axis=0)
    continuum_spectrum = np.nanmedian(spectra[:, continuum_mask & ~line_mask], axis=0) 
    std_spectrum = np.full_like(line_spectrum, np.nanstd(continuum_spectrum) )

    ew     = np.trapezoid(1-line_spectrum, wlam[line_mask])
    ew_abs = np.trapezoid(np.abs(1-line_spectrum), wlam[line_mask])
    ew_std = np.trapezoid(std_spectrum, wlam[line_mask])

    return ew, ew_abs, ew_std


if __name__ == "__main__":
    dr20 = pd.read_csv(Path("../data/dr20_pointings_final_v4.csv"))
    dr20.set_index("expnum",inplace=True,drop=False)
    sdsswork_folder = Path("/","data","LVM","sdsswork")

    dr20["drp_file"] = dr20.apply(lambda row: 
        sdsswork_folder / "lvm" / "spectro" / "redux" / "1.2.0" /
        f"{row['tilegrp']}" / f"{row['tileid']}" / f"{row['mjd']}" /  
        f"lvmSFrame-{row['expnum']:08}.fits",
        axis=1
        )
    
    rows = []
    for i, row in tqdm.tqdm(dr20.iterrows()):
        if not row["drp_file"].is_file():
            print(f'missing file for expnum={row["expnum"]}')
            continue
        
        with fits.open(row["drp_file"]) as hdul:    
            header = hdul["PRIMARY"].header
            cube = hdul["FLUX"].data
            ivar = hdul["ivar"].data
            wlam = hdul["WAVE"].data
            sci_index = Table(hdul["SLITMAP"].data)
            
        slitmap = select_fibers_with_stars(header,sci_index,stars["SkyCoord"])
        if np.sum(slitmap) == 0:
            print(f'no stars in expnum={row["expnum"]}')
            continue
        
        result = {
            "expnum":row["expnum"],
            "fibers":(np.where(slitmap)[0]+1).tolist(),
            "n_fibers":np.sum(slitmap)
            }
        for j in range(3):
            ew, ew_abs, ew_std = measure_ew(cube,wlam,slitmap,continuum_ranges[j],line_ranges[j])
            result[f"ew_{j}"] = ew 
            result[f"ew_{j}_abs"] = ew_abs
            result[f"ew_{j}_std"] = ew_std

        rows.append(result)

    df = pd.DataFrame(rows)
    df.set_index("expnum",inplace=True)
    df.to_csv("ew_telluric.csv")
    