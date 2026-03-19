"""
Find the weakest detected emission line in the DAP

This script addresses point 6 of the LVM DRP QA. The result is 
saved to three csv tables. The contain the following columns:

* expnum : exposure number of the pointing.
* Line : line that has at least min_detected fibers above threshold.
* n_detected : number of fibers where this line is above the threshold.
* snr : the largest flux/error of this line.
* flux : the flux of the fiber with the heighest snr.
* error : the error of the fiber with the heighest snr.
* detected_lines : all other lines that have at least min_detected fibers above threshold.
"""

from astropy.io import fits
from astropy.table import Table 
import numpy as np 
import pandas as pd 
from pathlib import Path
import re
from tqdm import tqdm


def find_weakest_eline(elines, threshold=3, min_detected=5):
    """Find the weakest eimssion-line that is detected with 3 sigma. 
    
    Parameters
    ----------
    elines : astropy.Table 
        Table with non-parametric emission-line properties.
    threshold : float
        Detection threshold in sigma.
    min_detected : int
        Number of fibers in which a line must be detected.
    """

    # find all emission lines in this table
    pattern = re.compile(r"^flux_(.+)$")
    lines = sorted(
        m.group(1)
        for col in list(elines.columns)
        if (m := pattern.match(col))
    )
    
    rows = []
    for line in lines:
        # compute for all fibers
        snr = np.divide(elines[f"flux_{line}"],
                        elines[f"e_flux_{line}"],
                        out=np.zeros_like(elines[f"flux_{line}"]),
                        where=elines[f"e_flux_{line}"]>0)    
        
        # how many fibers are detected
        detected_mask = snr > threshold
        # search for the fiber with the strongest S/N and record the flux/error
        max_idx = np.argmax(snr)
        max_flux = elines[f"flux_{line}"][max_idx]
        max_error = elines[f"e_flux_{line}"][max_idx]
        
        rows.append({
            "line": line,
            "n_detected": detected_mask.sum(),
            "max_snr": np.nanmax(snr),
            "max_snr_flux" : max_flux,
            "max_snr_error" : max_error
        })
        
    line_properties = pd.DataFrame(rows) 
    # search for weakest detected line
    idx = line_properties[line_properties["n_detected"]>=min_detected]["max_snr"].idxmin()
    weakest_line = line_properties.loc[idx].to_list()
    # add list of lines that are detected
    return weakest_line + [",".join(line_properties[line_properties["n_detected"]>min_detected]["line"])]

if __name__ == "__main__":
    # the catalogue with the tiles for DR20
    dr20 = pd.read_csv(Path("dr20_pointings_final_v2.csv"))
    # adjust the location. This is for the flatiron server
    sdsswork_folder = Path("..","data","work")
    # the folder structure for the dap files is different
    dap_files = dr20.apply(lambda row: 
        sdsswork_folder / "lvm" / "spectro" / "analysis" / "1.2.0" /
        f"{row['tilegrp']}" / 
        f"{row['tileid']}" / 
        f"{row['mjd']}" / 
        f"{row['expnum']:08}" / 
        f"dap-rsp108-sn20-{row['expnum']:08}.dap.fits.gz",
        axis=1
        )

    rowsB = []
    rowsR = []
    rowsI = []
    for filename in tqdm(dap_files):
        expnum = int(re.search(r"-(\d+)\.",filename.stem).group(1))
        with fits.open(filename) as hdul:
            NP_ELINES_B = Table(hdul["NP_ELINES_B"].data)
            NP_ELINES_R = Table(hdul["NP_ELINES_R"].data)
            NP_ELINES_I = Table(hdul["NP_ELINES_I"].data)
        rowsB.append([expnum]+find_weakest_eline(NP_ELINES_B, threshold=3, min_detected=5))
        rowsR.append([expnum]+find_weakest_eline(NP_ELINES_R, threshold=3, min_detected=5))
        rowsI.append([expnum]+find_weakest_eline(NP_ELINES_I, threshold=3, min_detected=5))
    
    weakest_linesB = pd.DataFrame(rowsB,columns = ["expnum","Line","n_detected","snr","flux","error","detected_lines"])
    weakest_linesR = pd.DataFrame(rowsR,columns = ["expnum","Line","n_detected","snr","flux","error","detected_lines"])
    weakest_linesI = pd.DataFrame(rowsI,columns = ["expnum","Line","n_detected","snr","flux","error","detected_lines"])
    
    weakest_linesB.to_csv("elinesB.csv",index=False)
    weakest_linesR.to_csv("elinesR.csv",index=False)
    weakest_linesI.to_csv("elinesI.csv",index=False)

    print(f'{len(weakest_linesB["detected_lines"].str.split(",").explode().unique())} unique lines in B')
    print(f'{len(weakest_linesR["detected_lines"].str.split(",").explode().unique())} unique lines in R')
    print(f'{len(weakest_linesI["detected_lines"].str.split(",").explode().unique())} unique lines in I')