'''
get_DAP_file_list.py

Author: Maren Cosens
Date: 2024-03-26

Description:
    This script generates a list of DAP file locations from the table of DR20 tiles and downloads them from the sas by running the following command in the terminal (this preserves the file structure so the scripts can easily be run on UTAH later):
    "rsync -avz --no-motd --files-from=dap_file_list.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork"
    Script will prompt for the password to download files (twice if some files fail and require checking for the zipped version)
'''
##import necessary libraries
import pandas as pd
import numpy as np
import os

##set download directory
extragalactic_targets=['NGC598', 'NGC4945']#list of extragalactic targets which will have secondary DAP analysis performed

def get_file_list(exposure_list):
    ##read in the file list
    file_list=pd.read_csv(exposure_list)
    #get DAP filenames from the file list
    drp_loc=file_list['location'].to_list()
    drp_files=file_list['filename'].to_list()

    #replace redux with analysis and remove lvmSFrame- prefix and .fits suffix to get DAP file locations
    dap_loc= [entry.replace('redux', 'analysis').replace('lvmSFrame-', '').replace('.fits', '/') for entry in drp_loc]

    # concatenate dap_loc and drp_files entries
    dap_files = [f"{loc}{file}" for loc, file in zip(dap_loc, drp_files)]
    dap_files = [entry.replace('lvmSFrame', 'dap-rsp108-sn20').replace('.fits', '.dap.fits').replace('sdsswork', '') for entry in dap_files]
    ##add cl_prefix to extragalactic targets that are run through secondary analysis
    for i in range(len(file_list)):
        if file_list['NGCname'][i] in extragalactic_targets:
            dap_files[i] = dap_files[i].replace('dap-', 'cl_dap-')

    ##save file list to a text file
    with open('dap_file_list.txt', 'w') as f:
        for item in dap_files:
            f.write("%s\n" % item) 

##if some fail
def list_of_failed_files(failed_files):
    f= open(failed_files, 'r')
    fail_list=list(f)
    new_list=[]
    for i in range(len(fail_list)):
        try:
            new_list.append(fail_list[i].split('"')[1].replace('.fits', '.fits.gz')) #replace with just DAP filename (zipped version)
        except:
            #do not add row to new list to avoid errors in subsequent steps 
            pass
    with open('dap_file_list_gz.txt', 'w') as f:
        for item in new_list:
            f.write("%s\n" % item)

def unzip_files(dap_file_list):
    files=open(dap_file_list, 'r')
    for f in files:
        os.system(f"gunzip sdsswork/{f}")

#to run in command line
if __name__ == '__main__':
    #generate list of DAP files
    get_file_list('dr20_pointings_final_v1.2.0.csv')
    #retrieve files from SAS
    os.system("rsync -avz --no-motd --files-from=dap_file_list.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork 2>rsync-errors.txt") #-delete ensures an exact mirror (tiles that have been removed from list are removed from directory)
    #if errors written to 'rsync_errors.txt'; modify that list for zipped dap files
    try:
        list_of_failed_files('rsync-errors.txt')
        #download zipped versions
        os.system("rsync -avz --no-motd --files-from=dap_file_list_gz.txt rsync://sdss5@dtn.sdss.org/sdsswork sdsswork") #removed -delete command to avoid removing files from previous step
        #extract for use in subsequent scripts
        unzip_files('dap_file_list_gz.txt')
    except: 
        pass