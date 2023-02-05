import argparse
import json
import numpy as np
import os

from configobj import ConfigObj
from itertools import repeat
from multiprocessing import Pool
from nipype.interfaces.io import DataGrabber
from file_utils import strip_suffix, output_dict
from utils_func import func_preproc


def preprocess(main_dir, rm_interm_func, n_cores):
    # Ignore tasks
    task_ignore = ['Bang', 'ClipsTrn', 'ClipsVal', 'ContRing', 'WedgeAnti',
                   'WedgeClock' 'Retinotopy-Wedge', 'Raiders', 'RestingState']
    # Hard code subject list
    subj_list = ['sub-01', 'sub-02', 'sub-04', 'sub-05',
                 'sub-06', 'sub-07', 'sub-08', 'sub-09',
                 'sub-11', 'sub-12', 'sub-13', 'sub-14', 
                 'sub-15']
    subj_list = ['sub-01', 'sub-02']
    # Set templates for finding functional and anatomical (T1) files
    func_file = os.path.abspath('data/%s/ses-*/func/*bold.nii.gz')

    # Use DataGrabber to collect functional and anatomical scans
    dg = DataGrabber(infields=['sub'], outfields=['func'])
    dg.inputs.base_directory = os.path.abspath(main_dir)
    dg.inputs.field_template = {'func': func_file}
    dg.inputs.template_args = {'func': [['sub']]}
    dg.inputs.template = '*'
    dg.inputs.sort_filelist = False
    dg.inputs.sub = subj_list
    iter_list = dg.run().outputs
    func_list = [(subj, f) for subj, func_ses in zip(subj_list, iter_list.func) for f in func_ses]

    # Ignore naturalistic viewing and resting state scans
    func_list = [func for func in func_list if all([t not in func[1] for t in task_ignore])]
    # Ignore scans that that are already preprocessed
    prep_steps = ['fill_nan', 'func_resample', 'smooth', 'filtz', 'applymask']
    prep_ext = ''.join([output_dict[p] for p in prep_steps])
    func_list = [f for f in func_list 
                 if not os.path.isfile(strip_suffix(f[1]) + prep_ext + '.nii.gz')]
    
    # Loop through functional sessions
    print('func preprocessing')
    pool = Pool(processes=n_cores)
    pool.starmap(run_func_preproc, zip(func_list,repeat(main_dir), repeat(rm_interm_func)))


def run_func_preproc(func, main_dir, rm_interm_func):
        print(f'subject: {func[0]}')
        func_output = func_preproc(func[1])
        if rm_interm_func:
            rm_files = [('fill_nan'), ('resample_out'), ('smooth'), ('temporal_filt_z')]
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
    parser.add_argument('-n', '--n_cores',
                        help='number of cores to use for parallel processing',
                        default = 14, # nsubjects (for most protocols)
                        required=False,
                        type=int)


    args_dict = vars(parser.parse_args())
    preprocess(args_dict['main_dir'], args_dict['remove_intermediate_func'],
               args_dict['n_cores'])




