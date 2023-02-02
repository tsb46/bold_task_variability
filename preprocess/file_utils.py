import json
import glob
import os 
import pandas as pd
import shutil


# Define renaming dictionary for output naming
output_dict = {
    'select_vol': '_firstvol', 
    'transform': '_transform',
    'func_resample': '_resamp',
    'smooth': '_sm',
    'filtz': '_filtz',
    'applymask': '_mask'
}


def create_protocol_cache(subject_list, main_dir, anat_list, ignore_cache=False):
    # Create cache .json for communicating b/w anat and func pipelines
    json_cache = {}
    for subj, anat in zip(subject_list, anat_list):
        if os.path.isfile(f'{os.path.abspath(main_dir)}/{subj}_cache.json') & ~ignore_cache:
            json_cache_subj = json.load(open(f'{os.path.abspath(main_dir)}/{subj}_cache.json', 'rb'))
            json_cache[subj] = json_cache_subj 
        else:
            json_cache_subj = {}
            json_cache_subj['anat'] = {}
            json_cache_subj['anat']['orig'] = anat
            json_cache[subj] = json_cache_subj

    return json_cache


# rename output for file renaming
def rename_output(fp, step, ext='.nii.gz'):
    fp_strip = strip_suffix(fp)
    fp_step = f'{fp_strip}{step}{ext}'
    return os.path.abspath(fp_step)


# Remove file extension for .nii or .nii.gz
def strip_suffix(fp):
    return os.path.splitext(os.path.splitext(fp)[0])[0]

