"""
Find the weakest continuum for each channel

This script addresses point 7 of the LVM DRP QA. The result is 
saved a csv table with the following columns:

expnum                - exposure number of the pointing.

The following columns are present for each filter (B, R, I):

weakest_continuum_X   - Weakest detected continuum level.
continuum_noise_X     - Noise at the weakest detected continuum.
continuum_std_X       - Std at the weakest detected continuum.
"""


from astropy.io import fits 
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from pathlib import Path
import re
from tqdm import tqdm


# all emission lines from the DAP
emission_lines = np.array([
    3686.83, 3691.56, 3697.15, 3703.85, 3711.97, 3726.03, 3728.82,
    3734.37, 3750.15, 3758.9 , 3770.63, 3797.9 , 3819.61, 3835.38,
    3868.75, 3888.65, 3889.05, 3933.66, 3964.73, 3967.46, 3968.47,
    3970.07, 4026.19, 4068.6 , 4076.35, 4101.77, 4120.81, 4177.21,
    4227.2 , 4243.98, 4267.  , 4287.4 , 4340.49, 4358.1 , 4358.37,
    4359.34, 4363.21, 4413.78, 4414.45, 4416.27, 4452.11, 4457.95,
    4470.29, 4471.48, 4474.91, 4485.21, 4562.48, 4571.1 , 4632.27,
    4658.1 , 4685.68, 4701.62, 4711.33, 4713.14, 4724.17, 4733.93,
    4740.2 , 4754.83, 4769.6 , 4774.74, 4777.88, 4813.9 , 4814.55,
    4861.36, 4881.11, 4889.63, 4893.4 , 4905.35, 4921.93, 4924.5 ,
    4930.5 , 4942.5 , 4958.91, 4972.5 , 4973.39, 4985.9 , 5006.84,
    5015.68, 5039.1 , 5072.4 , 5107.95, 5111.63, 5145.8 , 5158.  ,
    5158.9 , 5176.  , 5184.8 , 5191.82, 5197.9 , 5200.26, 5220.06,
    5261.61, 5268.88, 5270.3 , 5273.38, 5277.8 , 5296.84, 5302.86,
    5309.18, 5333.65, 5335.2 , 5376.47, 5411.52, 5412.64, 5424.2 ,
    5426.6 , 5484.8 , 5517.71, 5527.33, 5577.34, 5631.1 , 5677.  ,
    5720.7 , 5754.59, 5876.  , 5889.95, 5895.92, 6087.  , 6300.3 ,
    6312.06, 6363.78, 6374.51, 6435.1 , 6548.05, 6562.85, 6583.45,
    6678.15, 6716.44, 6730.82, 6855.18, 7005.67, 7065.19, 7135.8 ,
    7155.14, 7171.98, 7236.  , 7281.35, 7290.42, 7291.46, 7318.92,
    7323.88, 7329.66, 7377.83, 7388.16, 7411.61, 7452.5 , 7637.52,
    7686.19, 7686.9 , 7751.06, 7774.  , 7891.8 , 7999.85, 8125.22,
    8229.55, 8236.77, 8239.24, 8240.82, 8242.51, 8244.32, 8246.27,
    8248.35, 8250.6 , 8253.03, 8255.65, 8258.49, 8261.57, 8264.92,
    8268.57, 8272.56, 8276.94, 8281.75, 8287.06, 8292.94, 8299.47,
    8300.99, 8306.75, 8308.39, 8314.89, 8324.06, 8334.42, 8345.55,
    8357.51, 8359.  , 8374.48, 8392.4 , 8446.  , 8467.25, 8498.02,
    8502.48, 8542.09, 8545.38, 8578.7 , 8598.39, 8616.96, 8662.14,
    8665.02, 8727.13, 8750.47, 8862.78, 8891.88, 9014.91, 9033.45,
    9051.92, 9069.  , 9123.6 , 9226.6 , 9229.02, 9266.  , 9267.54,
    9399.02, 9470.93, 9531.1 , 9545.97, 9682.13
    ])

# strong emission lines from Sanchez+2025 Table 6
emission_lines_strong = np.array([
    3726.03,3728.82,4101.77,4340.49,4861.36,4958.91,5006.84,6300.30,
    6548.05,6562.85,6583.45,6716.44,6730.82,9069.00,9531.10
    ])

telluric_lines = np.array([
    (5881, 5905),(5893, 5913),(5913, 5937),(5933, 5961),(5961, 5981),
    (6270, 6331),(6468, 6488),(6477, 6498),(6511, 6531),(6860, 6978),(6979, 7045),
    (7034, 7056),(7046, 7067),(7143, 7164),(7154, 7348),(7339, 7385),
    (7375, 7397),(7388, 7410),(7584, 7713),(7889, 7912),(7953, 7976),
    (8004, 8027),(8092, 8392),(8400, 8424),(8926, 9868),
    (9863, 9890),
    (10738, 10767),(10764, 10793),(10790, 10832),(10830, 10926),(10923, 11000)
    ])



def find_weakest_continuum(cube,ivar,slitmap,mask,snr=3):
    """Weakest detected continuum level in one tile.
    
    Parameters
    ----------
    cube : np.ndarray of shape (m,n)
        Datacube with m fiber, each containing n spectral pixels.
    ivar : np.ndarray of shape (m,n)
        Inverse variance of the datacube.
    slitmap : np.ndarray of shape (m,) of bool
        Mask to select only science fibers.
    mask : np.ndarray of shape (n,) of bool
        Spectral mask to select part of the spectrum.
    snr : float, optional
        Signal to noise when we deem a continuum detected.
        
    Returns
    -------
    weakest : float
        Weakest detected continuum level.
    weakest_noise : float
        Noise at the weakest detected continuum.
    weakest_std : float
        Std at the weakest detected continuum.
    """
    
    # select science fibers and the wavelengths we use for the continuum
    continuum_spectra = cube[np.ix_(slitmap, mask)]
    continuum_ivar    = ivar[np.ix_(slitmap, mask)]
    
    # we remove a few fibers that only contain nan
    continuum_ivar    = continuum_ivar[~np.all(np.isnan(continuum_spectra), axis=1)]
    continuum_spectra = continuum_spectra[~np.all(np.isnan(continuum_spectra),axis=1)]
    
    # continuum level is the ivar-weighted mean
    continuum_level = np.nansum(continuum_spectra * continuum_ivar, axis=1) / np.nansum(continuum_ivar, axis=1)
    continuum_noise = 1.0 / np.sqrt(np.nansum(continuum_ivar, axis=1))
    continuum_std   = np.nanstd(continuum_spectra,axis=1) / np.sqrt(continuum_spectra.shape[1])

    # for each fiber compute signal to noise and sel ect those larger than snr 
    detected = (continuum_level / continuum_noise) >= snr
    weakest_idx = np.nanargmin(continuum_level[detected])
    
    weakest       = continuum_level[detected][weakest_idx]
    weakest_noise = continuum_noise[detected][weakest_idx]
    weakest_std   = continuum_std[detected][weakest_idx]

    return weakest, weakest_noise, weakest_std


if __name__ == "__main__":
    sdsswork_folder = Path("..","..","data","work")

    # the catalogue with the tiles for DR20
    dr20 = pd.read_csv(Path("dr20_pointings_final_v4.csv"))
    dr20["drp_file"] = dr20.apply(lambda row: 
        sdsswork_folder / "lvm" / "spectro" / "redux" / "1.2.0" /
        f"{row['tilegrp']}" / f"{row['tileid']}" / f"{row['mjd']}" /  
        f"lvmSFrame-{row['expnum']:08}.fits",
        axis=1
        )
    
    rows = []
    for drp_file in tqdm(dr20["drp_file"]):
        with fits.open(drp_file) as hdul:
            cube = hdul["FLUX"].data
            ivar = hdul["IVAR"].data
            wlam = hdul["WAVE"].data
            sci_index = Table(hdul["SLITMAP"].data)
            slitmap = np.zeros(len(sci_index),dtype=bool)
            slitmap[sci_index[sci_index["targettype"]=="science"]["fiberid"]-1] = True

        # build masks to exclude contaminated wavelengths before measuring continuum.
        # strong lines get a wider window (30 AA) since they can appear in redshifted 
        # extragalactic observations.
        strong_emission_line_mask = np.min(np.abs(wlam[:, None] - emission_lines_strong[None, :]), axis=1) < 30
        emission_line_mask = np.min(np.abs(wlam[:, None] - emission_lines[None, :]), axis=1) < 10
        telluric_mask = np.any((wlam[:, None] >= telluric_lines[:, 0]) & (wlam[:, None] <= telluric_lines[:, 1]), axis=1)
        
        row = {"expnum": int(drp_file.stem.rsplit("-",1)[1])}

        # B: blue channel, R: red channel, I: near-infrared channel
        for channel, name in zip([(3750,5500),(5900,7250),(7800,9500)],["B","R","I"]):
            # define a mask within each channel and exclude all lines that fall inside
            mask = ((channel[0]<wlam) & (wlam<channel[1])) & ~emission_line_mask & ~strong_emission_line_mask & ~telluric_mask
            weakest_continuum, continuum_noise, continuum_std = find_weakest_continuum(cube,ivar,slitmap,mask)
            row[f"weakest_continuum_{name}"] = weakest_continuum
            row[f"continuum_noise_{name}"] = continuum_noise
            row[f"continuum_std_{name}"] = continuum_std

        rows.append(row)
        
    continuum = pd.DataFrame(rows)
    continuum = continuum.set_index("expnum")
    continuum.to_csv("weakest_continuum.csv")