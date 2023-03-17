import json
import glob
import os 
import pandas as pd
import shutil


# Define renaming dictionary for output naming
output_dict = {
    'extract_roi': '_roi',
    'fill_nan': '_nan',
    'func_resample': '_resamp',
}


# rename output for file renaming
def rename_output(fp, step, ext='.nii.gz'):
    fp_strip = strip_suffix(fp)
    fp_step = f'{fp_strip}{step}{ext}'
    return os.path.abspath(fp_step)


# Remove file extension for .nii or .nii.gz
def strip_suffix(fp):
    return os.path.splitext(os.path.splitext(fp)[0])[0]


# Utility to pull task/ses/subj from functional file path 
def ibc_get_task(fp):
    fp_base = os.path.basename(fp)
    if '_run-' in fp_base:
        return fp_base.split('_')[:5]
    else:
        return fp_base.split('_')[:4]



