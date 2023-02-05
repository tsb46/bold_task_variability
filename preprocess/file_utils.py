import json
import glob
import os 
import pandas as pd
import shutil


# Define renaming dictionary for output naming
output_dict = {
    'fill_nan': '_nan',
    'func_resample': '_resamp',
    'smooth': '_sm',
    'filtz': '_filtz',
    'applymask': '_mask'
}


# rename output for file renaming
def rename_output(fp, step, ext='.nii.gz'):
    fp_strip = strip_suffix(fp)
    fp_step = f'{fp_strip}{step}{ext}'
    return os.path.abspath(fp_step)


# Remove file extension for .nii or .nii.gz
def strip_suffix(fp):
    return os.path.splitext(os.path.splitext(fp)[0])[0]

