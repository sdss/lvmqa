"""
Comparison with existing datasets for coloros and absolute photometry

This script addresses point 2 and 3 of the LVM DRP QA. The reuslt is
saved to a csv table. It contains the following columns:

expnum                    - exposure number of the pointing.
color_comparison_BR       - change in color B and R between SCI and MOD
color_comparison_ZR       - change in color Z and R between SCI and MOD
photometry_comparison_R   - change in color R between SCI and MOD
"""

from astropy.io import fits 
from astropy.table import Table 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from pathlib import Path
import re
from tqdm import tqdm


def get_sensitivity_from_header(drp_files):
    """ 
    Extract sensitivity information from lvmSFrame FITS headers.
    
    Parameters
    ----------
    drp_files : list of Path
        List of FITS files to read. 
        
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by `expnum` containing the sensitivity measurments.
    """ 
    
    # create a list of extentions to extract from the header
    modes = ["MOD","SCI","STD"]
    stats = ["M","R"]
    channels = ["B","R","Z"]
    columns = [f"{mode}SEN{stat}{channel}" 
               for mode in modes 
               for channel in channels 
               for stat in stats]
    
    rows = []
    for filename in tqdm(drp_files, desc="Reading FITS headers"):
        if filename.is_file():
            with fits.open(filename) as hdul:
                header = hdul["PRIMARY"].header
            
            row = {"expnum": int(filename.stem.rsplit("-",1)[1])}
            for column in columns:
                row[column] = header.get(column,np.nan)
            rows.append(row)
        else:
            print(f"missing file: {filename}")

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["expnum"]+columns)

    df = df.set_index("expnum")
    # invalid values in the header are -999.9
    df.replace(-999.9, np.nan, inplace=True)
    return df


def color_comparison(df,channel1,channel2="R"):
    """
    Compare the change in color between SCI and MOD:
        (SCIchannel1 / SCIchannel2) - (MODchannel1 / MODchannel2)
    The result is expressed in units of standard deviations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SCISENM*, SCISENR*, MODSENM*, MODSENR* columns.
    channel1 : str
        The first channel to compare (e.g., 'B' or 'Z').
    channel2 : str
        The reference channel (default is 'R').
        
    Returns
    -------
    pd.Series
        The absolute difference in units of standard deviations.
    """ 
    
    ratio_sci = df[f"SCISENM{channel1}"]/df[f"SCISENM{channel2}"] 
    ratio_mod = df[f"MODSENM{channel1}"]/df[f"MODSENM{channel2}"] 
    err_sci = np.sqrt((df[f"SCISENR{channel1}"] / df[f"SCISENM{channel2}"])**2 + 
                      (df[f"SCISENM{channel1}"] * df[f"SCISENR{channel2}"] / df[f"SCISENM{channel2}"]**2)**2)
    err_mod = np.sqrt((df[f"MODSENR{channel1}"] / df[f"MODSENM{channel2}"])**2 + 
                      (df[f"MODSENM{channel1}"] * df[f"MODSENR{channel2}"] / df[f"MODSENM{channel2}"]**2)**2)
    
    difference = np.abs(ratio_sci - ratio_mod)
    error = np.sqrt(err_sci**2 + err_mod**2)

    return difference / error


def photometry_comparison(df,channel="R"):
    """ 
    Compute the difference between SCI and MOD measurements for a single channel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SCISENM* and SCISENR*, MODSENM* and MODSENR* columns.
    channel : {'B', 'R', 'Z'}, optional
        Channel to use. Default is 'R'.

    Returns
    -------
    pd.Series
        The absolute difference in units of standard deviations.
    """

    difference = np.abs(df[f"SCISENM{channel}"]-df[f"MODSENM{channel}"])
    error = np.sqrt(df[f"SCISENR{channel}"]**2 + df[f"MODSENR{channel}"]**2)

    return difference / error


def plot_channels(df,expnum):
    """ 
    Plot MOD and SCI measurements for 'B', 'R', and 'Z' channels.
    
    Parameters
    ----------
    df : pd.DataFrame
    expnum : int
    """ 
    
    row = df.loc[expnum]
    mod  = row[["MODSENMB","MODSENMR","MODSENMZ"]]
    err_mod = row[["MODSENRB","MODSENRR","MODSENRZ"]]
    sci  = row[["SCISENMB","SCISENMR","SCISENMZ"]]
    err_sci = row[["SCISENRB","SCISENRR","SCISENRZ"]]
    
    fig,ax=plt.subplots(figsize=(3,2))
    ax.errorbar(np.arange(len(mod))+0.9,mod,yerr=err_mod,fmt="o",label="mod")
    ax.errorbar(np.arange(len(sci))+1.1,sci,yerr=err_sci,fmt="o",label="sci")
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(["B","R","Z"])
    ax.set(ylabel="sensitivity")
    ax.legend()
    plt.show()


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

    sensitivity = get_sensitivity_from_header(dr20["drp_file"])
    sensitivity["color_comparison_BR"] = color_comparison(sensitivity,channel1="B",channel2="R")
    sensitivity["color_comparison_ZR"] = color_comparison(sensitivity,channel1="Z",channel2="R")
    sensitivity["photometry_comparison_R"] = photometry_comparison(sensitivity,channel="R")
    
    print(f"{sensitivity.isna().any(axis=1).sum()} entries with invalid values")
    print(f"SCIB/SCIR - MODB/MODR > 3σ: {np.sum(sensitivity['color_comparison_BR']>3)}")
    print(f"SCIZ/SCIR - MODZ/MODR > 3σ: {np.sum(sensitivity['color_comparison_ZR']>3)}")
    print(f"SCIR-MODR > 3σ: {np.sum(sensitivity['photometry_comparison_R']>3)}")

    sensitivity[list(sensitivity.columns)[-3:] + list(sensitivity.columns)[:-3]].to_csv("photometry.csv")