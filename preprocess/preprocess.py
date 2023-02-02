import argparse
import json
import numpy as np
import os

from configobj import ConfigObj
from itertools import repeat
from multiprocessing import Pool
from nipype.interfaces.io import DataGrabber
from file_utils import create_protocol_cache
from utils_anat import anat_preproc
from utils_func import func_preproc


def preprocess(main_dir, rm_interm_func, ignore_cache, n_cores):
    # Ignore tasks
    task_ignore = ['Bang', 'ClipsTrn', 'ClipsVal', 'ContRing', 'WedgeAnti',
                   'WedgeClock' 'Retinotopy-Wedge', 'Raiders', 'RestingState']
    # Hard code subject list
    subj_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
                 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
                 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']
    # Set templates for finding functional and anatomical (T1) files
    func_file = os.path.abspath('data/%s/ses-*/func/*bold.nii.gz')
    anat_file = os.path.abspath('data/%s/ses-00/anat/*T1w.nii.gz')

    # Use DataGrabber to collect functional and anatomical scans
    dg = DataGrabber(infields=['sub'], outfields=['anat', 'func'])
    dg.inputs.base_directory = os.path.abspath(main_dir)
    dg.inputs.field_template = {'anat': anat_file,
                                'func': func_file}
    dg.inputs.template_args = {'anat': [['sub']],
                               'func': [['sub']]}
    dg.inputs.template = '*'
    dg.inputs.sort_filelist = False
    dg.inputs.sub = subj_list
    iter_list = dg.run().outputs
    anat_list = iter_list.anat
    func_list = [(subj, f) for subj, func_ses in zip(subj_list, iter_list.func) for f in func_ses]

    # Ignore naturalistic viewing and resting state scans
    func_list = [func for func in func_list if all([t not in func[1] for t in task_ignore])]

    # Read in (or create) protocol cache for keeping up with output
    json_cache_anat = create_protocol_cache(subj_list, main_dir, 
                                            anat_list, ignore_cache=False)

    # Loop through T1w images and preprocess
    print('T1w preprocessing')
    pool = Pool(processes=n_cores)
    pool.starmap(run_anat_preproc, zip(anat_list, subj_list, 
                                       repeat(json_cache_anat), 
                                       repeat(main_dir)))
    run_anat_preproc(anat_list[0], subj_list[0], json_cache_anat, main_dir)
    
    # Loop through functional sessions
    print('func preprocessing')
    pool = Pool(processes=n_cores)
    pool.starmap(run_func_preproc, zip(func_list, repeat(json_cache_anat), 
                                       repeat(main_dir), repeat(rm_interm_func)))

def run_anat_preproc(anat, subj, json_cache, main_dir):
        print(f'subject: {subj}')
        json_cache_subj = anat_preproc(anat, json_cache[subj])
        json_output = f'{os.path.abspath(main_dir)}/{subj}_cache.json'
        json.dump(json_cache_subj, open(json_output, 'w'), ensure_ascii=False, indent=4)


def run_func_preproc(func, json_cache, main_dir, rm_interm_func):
        print(f'subject: {func[0]}')
        func_output = func_preproc(func[1], json_cache[func[0]])
        if rm_interm_func:
            rm_files = [('resample_out'), ('smooth'), ('temporal_filt_z')]
            for file in rm_files:
                os.remove(func_output[file])


if __name__ == '__main__':
    """Preprocess functional & anatomical scans from IBC dataset"""
    parser = argparse.ArgumentParser(description='Preprocess functional & anatomical scans from IBC dataset')
    parser.add_argument('-d', '--main_dir',
                        help='directory where IBC data is stored',
                        required=False,
                        default='data',
                        type=str)
    parser.add_argument('-r', '--remove_intermediate_func',
                        help='remove output of all functional preprocessing, except final (save a lot of space)',
                        required=False,
                        default=0,
                        type=int)
    parser.add_argument('-c', '--ignore_cache',
                        help='whether to ignore cache (i.e. ignore intermediate outputs,'
                        'start from scratch (default: 0)',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('-n', '--n_cores',
                        help='number of cores to use for parallel processing',
                        default = 14, # nsubjects (for most protocols)
                        required=False,
                        type=int)


    args_dict = vars(parser.parse_args())
    preprocess(args_dict['main_dir'], args_dict['remove_intermediate_func'],
               args_dict['ignore_cache'], args_dict['n_cores'])




