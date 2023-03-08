import argparse
import nibabel as nb
import numpy as np
import pandas as pd
import pickle
import os

from nipype.interfaces.io import DataGrabber
from patsy import dmatrix
from sklearn.linear_model import LinearRegression, Ridge
from utils.load_write import group_concat, subj_ses_dict, write_nifti
from utils.paradigm import create_event_file
from utils.glm_utils import create_regressor, spline_model_predict


def find_scans(subject, main_dir):
    # Use DataGrabber to collect preprocessed functional data (assume masked data is final step)
    func_template = os.path.abspath('data/%s/%s/func/*mask.nii.gz')
    event_template = os.path.abspath('data/%s/%s/func/*events.tsv')
    dg = DataGrabber(infields=['sub', 'ses', 'ev'], outfields=['func', 'event'])
    dg.inputs.base_directory = os.path.abspath(main_dir)
    dg.inputs.field_template = {'func': func_template, 
                                'event': event_template}
    dg.inputs.template_args = {'func': [['sub', 'ses']], 
                               'event': [['ev', 'ses']]}
    dg.inputs.template = '*'
    dg.inputs.sort_filelist = True # very important to align ev and func
    dg.inputs.sub = subject
    dg.inputs.ses = subj_ses_dict[subject]
    dg.inputs.ev = subject
    dg_res = dg.run()
    func_list = dg_res.outputs.func
    ev_list = dg_res.outputs.event
    # flatten lists
    func_list = [f for ses in func_list for f in ses]
    ev_list = [e for ses in ev_list for e in ses]

    return func_list, ev_list


def concat_events(ev_list, tr, slicetime_ref, basis_type, 
                  task_all, func_metadata, func_n):
    ev_concat = []
    for ev, task_meta, n_scan in zip(ev_list, func_metadata, func_n):
        print(task_meta[2])
        # Get event onsets and duration dataframe
        ev_df = create_event_file(ev, task_meta[2])
        # Create regressor from event dataframe
        ev_reg, basis = create_regressor(ev_df, basis_type, tr, n_scan,
                                             slicetime_ref, task_all)
        ev_concat.append(ev_reg)

    ev_concat = pd.concat(ev_concat, axis=0, ignore_index=True)

    return ev_concat, basis


def evaluate_model(model, basis, mask):
    pred_bold = spline_model_predict(model, basis)
    for dur_pred in pred_bold:
        write_nifti(dur_pred[1], mask, f'pred_duration_{dur_pred[0]}s')


def regression(basis_type, func_concat, ev_concat, intercept=True):
    if basis_type == 'spline':
        intercept=False
    else:
        intercept=True
    lin_reg = LinearRegression(fit_intercept=intercept)
    lin_reg.fit(ev_concat.values, func_concat)
    return lin_reg


def run_main(subject, tr, task_all, basis_type, main_dir, mask_fp, n_cores, slicetime_ref):
    # Find subject functional scans
    func_list, ev_list = find_scans(subject, main_dir)
    # Load brain mask
    mask = nb.load(mask_fp)
    mask_bin = mask.get_fdata()
    # Temporal concatenation of functional scans
    func_concat, func_meta, func_n = group_concat(func_list, mask_bin, 
                                                  return_length=True)
    # Temporal concanation of task regressors
    ev_concat, basis = concat_events(ev_list, tr, slicetime_ref, 
                                     basis_type, task_all, func_meta, 
                                     func_n)
    reg_model = regression(basis_type, func_concat, ev_concat)
    # reg_model, _ = pickle.load(open('lin_reg.pkl', 'rb'))
    # Write out coefficients
    write_nifti(reg_model.coef_.T, mask, f'{subject}_coef.nii')
    pickle.dump([reg_model, basis], open('lin_reg.pkl', 'wb'))
    if basis_type == 'spline':
        evaluate_model(reg_model, basis, mask)

    



if __name__ == '__main__':
    """Preprocess functional & anatomical scans from IBC dataset"""
    parser = argparse.ArgumentParser(description='Preprocess functional & anatomical scans from IBC dataset')
    parser.add_argument('-s', '--subject',
                        help='subject string',
                        required=True,
                        type=str)
    parser.add_argument('-t', '--repetition_time',
                        help='repetition time (TR) for functional scan',
                        default = 2, 
                        required=False,
                        type=int)
    parser.add_argument('-u', '--universal_task',
                        help='whether to model all task blocks as a single type',
                        default = 1, 
                        required=False,
                        type=int)
    parser.add_argument('-b', '--basis_type',
                        help='type of basis for event regressor',
                        required=False,
                        choices=['spline', 'hrf', 'hrf3'],
                        default='spline',
                        type=str)
    parser.add_argument('-d', '--main_dir',
                        help='directory where IBC data is stored',
                        required=False,
                        default='data',
                        type=str)
    parser.add_argument('-m', '--mask',
                        help='filepath to brain mask',
                        default = 'preprocess/MNI152_T1_2mm_brain_mask.nii.gz',
                        required=False,
                        type=str)
    parser.add_argument('-n', '--n_cores',
                        help='number of cores to use for parallel processing',
                        default = 14, 
                        required=False,
                        type=int)
    parser.add_argument('-r', '--slicetime_ref',
                        help='what time point (in seconds) between consecutive volumes' 
                        'are slices interpolated to in the slicetiming correction.' 
                        'If no slicetiming correction is applied, could set to 0',
                        default = 0.5, # default slicetime reference in current pipeline
                        required=False,
                        type=float)


    args_dict = vars(parser.parse_args())
    run_main(args_dict['subject'], args_dict['repetition_time'], 
             args_dict['universal_task'], args_dict['basis_type'], 
             args_dict['main_dir'], args_dict['mask'], 
             args_dict['n_cores'], args_dict['slicetime_ref'])
