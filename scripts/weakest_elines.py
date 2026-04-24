"""
Find the weakest detected emission line in the DAP

This script addresses point 6 of the LVM DRP QA. The result is 
saved a csv table with the following columns:

expnum           - exposure number of the pointing
detected_lines   - all other lines with at least min_detected fibers above threshold

The following columns are present for each filter (B, R, I):

weakest_line_X   - line with at least min_detected fibers above threshold
flux_X           - flux of the fiber with the highest SNR
error_X          - error of the fiber with the highest SNR
snr_X            - largest flux/error ratio for this line
n_detected_X     - number of fibers above threshold
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
    return weakest_line + [";".join(line_properties[line_properties["n_detected"]>min_detected]["line"])]

if __name__ == "__main__":
    sdsswork_folder = Path("..","..","data","work")

    dr20 = pd.read_csv(Path("dr20_pointings_final_v4.csv"))
    dr20["dap_file"] = dr20.apply(lambda row: 
        sdsswork_folder / "lvm" / "spectro" / "analysis" / "1.2.0" /
        f"{row['tilegrp']}" / f"{row['tileid']}" / f"{row['mjd']}" / f"{row['expnum']:08}" / 
        f"dap-rsp108-sn20-{row['expnum']:08}.dap.fits.gz",
        axis=1
        )
    
    rows = []
    for filename in tqdm(dr20["dap_file"]):
        expnum = int(re.search(r"-(\d+)\.",filename.stem).group(1))

        if filename.is_file():
            with fits.open(filename) as hdul:
                NP_ELINES_B = Table(hdul["NP_ELINES_B"].data)
                NP_ELINES_R = Table(hdul["NP_ELINES_R"].data)
                NP_ELINES_I = Table(hdul["NP_ELINES_I"].data)
        else:
            print(f"missing file: {filename}")
            continue

        row = {"expnum" : expnum}
        detected_lines = []
        for table, channel in zip ([NP_ELINES_B,NP_ELINES_R,NP_ELINES_I],["B","R","I"]):
            results = find_weakest_eline(table, threshold=3, min_detected=5)
            row = row | {
                f"weakest_line_{channel}" : results[0],
                f"flux_{channel}" : results[3], 
                f"error_{channel}" : results[4],     
                f"snr_{channel}" : results[2],
                f"n_detected_{channel}" : results[1]
            }
            detected_lines.append(results[5])
        row = row | {"detected_lines": ";".join(detected_lines)}
        rows.append(row)
    
    weakest_lines = pd.DataFrame(rows)
    weakest_lines = weakest_lines.set_index("expnum")
    weakest_lines.to_csv("weakest_elines.csv")
    