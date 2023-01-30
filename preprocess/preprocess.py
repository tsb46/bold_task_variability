import argparse
import json
import numpy as np
import os

from configobj import ConfigObj
from itertools import repeat
from multiprocessing import Pool
from nipype.interfaces.io import DataGrabber
from fsl_topup import apply_topup
from file_utils import create_protocol_cache, get_subject_session, prepare_derivatives
from utils_anat import anat_preproc
from utils_func import func_preproc


def preprocess(protocol, main_dir, ignore_cache, n_cores):
    # Script parameters
    cache_dir = f'{main_dir}/tmp'

    # Set output directory based on protocol name
    output_dir = f'{main_dir}/derivatives/{protocol}'

    # Create protocol derivatives folder if not exist
    os.makedirs(output_dir, exist_ok=True)

    # prepare derivatives folder (where results of pipeline stored)
    # only needs to be run once, but OK if runs again
    prepare_derivatives(main_dir)

    # Get subjects-sessions that belong to protocol
    subject_session_func = get_subject_session(protocol)

    # Pull config obj
    cobj = ConfigObj(f'preprocess/pipeline_config/IBC_preproc_{protocol}.ini')

    # Pull tasks for session 
    ses_tasks = [key.split('session_task-')[1].split('_func')[0] for key in cobj['config'].keys() 
                 if key.startswith('session_task') & key.endswith('func')]
    # Pull first task json (for slice timing parameters)
    task_json = json.load(open(f'data/task-{ses_tasks[0]}_bold.json', 'rb'))
    t_custom = task_json['SliceTiming']
    np.savetxt(f'{cache_dir}/{protocol}_timing.txt', t_custom)

    # Pull metadata
    tr = float(cobj['config']['TR'])
    metadata_dict = dict(tr=tr, 
                         slicetime=f'{cache_dir}/{protocol}_timing.txt')


    # Apply FSL TOPUP Distortion Correction
    acq = None
    if protocol in ['rs']:
        acq = 'mb6'
    elif protocol in ['mtt1', 'mtt2']:
        acq = 'mb3'

    apply_topup(main_dir, cache_dir, n_cores, subject_session_func, acq)

    # Set templates for finding functional and anatomical (T1) files
    func_file = os.path.abspath('data/derivatives/%s/%s/func/dcsub*bold.nii.gz')
    anat_file = os.path.abspath(
        os.path.join('data/derivatives/', cobj['config']['anat'].replace('..', '%s'))
    )

    # Use DataGrabber to collect functional and anatomical scans
    dg = DataGrabber(infields=['sub', 'ses'], outfields=['anat', 'func'])
    dg.inputs.base_directory = os.path.abspath(output_dir)
    dg.inputs.field_template = {'anat': anat_file,
                                'func': func_file}
    dg.inputs.template_args = {'anat': [['sub']],
                               'func': [['sub', 'ses']]}
    dg.inputs.template = '*'
    dg.inputs.sort_filelist = False
    dg.inputs.sub = [s[0] for s in subject_session_func]
    dg.inputs.ses = [s[1] for s in subject_session_func]
    iter_list = dg.run().outputs
    anat_list = iter_list.anat
    func_list = iter_list.func

    # Read in (or create) protocol cache for keeping up with output
    json_cache = create_protocol_cache(subject_session_func, output_dir, protocol, 
                                       func_list, anat_list, ignore_cache=False)

    # Loop through T1w images and preprocess
    print('T1w preprocessing')
    pool = Pool(processes=n_cores)
    pool.starmap(run_anat_preproc, zip(anat_list, subject_session_func, 
                                       repeat(json_cache), repeat(output_dir), 
                                       repeat(protocol)))
    

    # Loop through functional sessions
    print('func preprocessing')
    pool = Pool(processes=n_cores)
    pool.starmap(run_func_preproc, zip(func_list, subject_session_func, 
                                       repeat(json_cache), repeat(metadata_dict),
                                       repeat(output_dir), repeat(protocol)))
    

def run_anat_preproc(anat, sub_ses, json_cache, 
                     output_dir, protocol):
        subj = sub_ses[0]
        print(f'subject: {subj}')
        json_cache_subj = anat_preproc(anat, json_cache[subj])
        json_output = f'{os.path.abspath(output_dir)}/{protocol}_{subj}_cache.json'
        json.dump(json_cache_subj, open(json_output, 'w'), ensure_ascii=False, indent=4)


def run_func_preproc(func_ses, sub_ses, json_cache, 
                     metadata_dict, output_dir, protocol):
        subj = sub_ses[0]
        print(f'subject: {subj}')
        json_cache_subj = func_preproc(func_ses, json_cache[subj], metadata_dict)
        json_output = f'{os.path.abspath(output_dir)}/{protocol}_{subj}_cache.json'
        json.dump(json_cache_subj, open(json_output, 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    """Preprocess functional & anatomical scans from protocol IBC dataset"""
    parser = argparse.ArgumentParser(description='Preprocess functional & anatomical scans from protocol IBC dataset')
    parser.add_argument('-p', '--protocol',
                        help='<Required> protocol string',
                        required=True,
                        type=str)
    parser.add_argument('-d', '--main_dir',
                        help='directory where IBC data is stored',
                        required=False,
                        default='data',
                        type=str)
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
    preprocess(args_dict['protocol'], args_dict['main_dir'], 
               args_dict['ignore_cache'], args_dict['n_cores'])




