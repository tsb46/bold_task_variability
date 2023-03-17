import argparse
import numpy as np
import pandas as pd
import pickle
import os

from itertools import repeat
from multiprocessing import Pool
from nilearn.glm.first_level import make_first_level_design_matrix as make_design_mat
from nilearn.signal import clean as niclean
from nipype.interfaces.io import DataGrabber
from scipy.stats import zscore
from sklearn.metrics import r2_score
from utils.file_utils import ibc_get_task
from utils.load_write import group_concat, subj_ses_dict, write_nifti
from utils.paradigm import create_event_file
from yaglm.Glm import Glm
from yaglm.config.loss import LinReg
from yaglm.config.penalty import Lasso, FusedLasso, GeneralizedLasso


# Some issues found in these tasks
task_ignore = ['task-Discount', 'task-ColumbiaCards']


def find_scans(subject, main_dir):
    # Use DataGrabber to collect preprocessed functional data (assume masked data is final step)
    func_template = os.path.abspath('data/%s/%s/func/*roi.txt')
    event_template = os.path.abspath('data/%s/%s/func/*events.tsv')
    dg = DataGrabber(infields=['sub', 'ses', 'ev'], outfields=['func', 'event'])
    dg.inputs.base_directory = os.path.abspath(main_dir)
    dg.inputs.field_template = {'func': func_template, 
                                'event': event_template}
    dg.inputs.template_args = {'func': [['sub', 'ses']], 
                               'event': [['ev', 'ses']]}
    dg.inputs.template = '*'
    dg.inputs.sort_filelist = True 
    dg.inputs.sub = subject
    dg.inputs.ses = subj_ses_dict[subject]
    dg.inputs.ev = subject
    dg_res = dg.run()
    func_list = dg_res.outputs.func
    ev_list = dg_res.outputs.event
    # flatten lists
    func_list = [f for ses in func_list for f in ses]
    ev_list = [e for ses in ev_list for e in ses]

    # Filter out sessions
    func_list = [f for f in func_list 
                 if all([t not in f for t in task_ignore])]
    ev_list = [e for e in ev_list 
               if all([t not in e for t in task_ignore])]            
    ev_list = match_events(func_list, ev_list)

    return func_list, ev_list


def concat_events(ev_list, tr, slicetime_ref, basis_type, 
                  func_metadata, func_len):
    ev_concat = []
    for ev, task_meta, n_scan in zip(ev_list, func_metadata, func_len):
        # Get event onsets and duration dataframe
        ev_df = create_event_file(ev, task_meta[2])
        # Create regressor from event dataframe
        ev_reg = create_regressor(ev_df, basis_type, tr, n_scan, slicetime_ref)
        ev_concat.append(ev_reg)

    # Create block diagonal matrix of regressors
    ev_concat = pd.concat(ev_concat, axis=0, ignore_index=True).fillna(0)
    return ev_concat


def concat_func(func_list, tr, hp_freq=0.01):
    func_concat = []
    func_len = [] # keep track of scan length
    func_str = [] # keep track of what scan is stacked on what
    indx = 0 # initialize index
    # Loop through functional scans and stack
    for func_fp in func_list:
        func = np.loadtxt(func_fp)
        func_clean = niclean(func, standardize=True, detrend=True, 
                             t_r=tr, high_pass=hp_freq)
        func_concat.append(func_clean)
        func_n = func_clean.shape[0]
        func_len.append(func_n)
        func_str.append(ibc_get_task(func_fp) + [range(indx, indx+func_n)])
        indx += func_n

    func_concat = np.vstack(func_concat)

    return func_concat, func_str, func_len


def create_regressor(ev_df, basis_type, tr, n_scan, slicetime_ref):
    if basis_type == 'hrf':
        basis_str = 'glover'
    elif basis_type == 'hrf3':
        basis_str = 'glover + derivative + dispersion'
    frametimes = np.linspace(slicetime_ref, (n_scan - 1 + slicetime_ref) * tr, n_scan)
    design_mat = make_design_mat(frametimes, ev_df, hrf_model=basis_str, 
                                 drift_model=None)
    return design_mat


def fused_lasso(y_i, X, edge_list, penalty):
    # Extract index and ROI ts
    i = y_i[0]
    y = y_i[1]
    # Print progress
    print(i)
    # Specify penalty and fit
    penalty_fl = FusedLasso(pen_val=penalty, edgelist=edge_list)
    fl = Glm(loss=LinReg(), penalty=penalty_fl, solver='cvxpy')
    fl.fit(X, y)
    # get coefficients and in-sample r2
    y_pred = fl.predict(X)
    fl_r2 = r2_score(y, y_pred)
    fl_coef = fl.coef_
    return (i, (fl_coef, fl_r2))


def match_events(func_list, ev_list):
    ev_list_reorder = []
    for func_str in func_list:
        base_fp = os.path.basename(func_str)
        search_str = '_'.join(ibc_get_task(base_fp))
        ev_match = [ev for ev in ev_list if search_str in ev]
        if len(ev_match) > 1:
            raise Exception("""
                multiple matches between functional file and event file, 
                check the directory belonging to {}
            """.format(search_str))
        elif len(ev_match) == 0:
            raise Exception("""
                no matches of functional file with an event file, 
                check the directory belonging to {}
            """.format(search_str))
        else:
            ev_list_reorder.append(ev_match[0])

    return ev_list_reorder


def run_main(subject, tr, basis_type, main_dir, n_cores, slicetime_ref):
    # Find subject functional scans
    func_list, ev_list = find_scans(subject, main_dir)
    # Temporal concatenation of functional scans
    func_concat, func_str, func_len = concat_func(func_list, tr)
    # Temporal concanation of task regressors
    ev_concat = concat_events(ev_list, tr, slicetime_ref,
                              basis_type, func_str, func_len)
    coef, r2 = fit_model(func_concat, ev_concat, n_cores)
    # Write out coefficients
    pickle.dump([coef, r2], open(f'fl_{subject}.pkl', 'wb'))


def fit_model(func_concat, ev_concat, n_cores, penalty=0.0001):
    # Drop intercept (don't need) and z-score
    ev_concat_z = zscore(ev_concat.drop(columns='constant'))
    n_reg = ev_concat_z.shape[1]
    # Create full graph (all pair-wise edges)
    tril_ind = np.tril_indices_from(np.zeros((n_reg, n_reg)), k=-1)
    edge_list = [(x,y) for x, y in zip(tril_ind[0], tril_ind[1])]
    # split columns of functional concat into nested list for looping
    func_concat = func_concat.T.tolist()
    # enumerate list to keep index
    func_concat = [(i, f) for i, f in enumerate(func_concat)]

    # REMOVE
    func_concat = [func_concat[i] for i in range(12)]
    ########

    print('fit fused lasso models')
    pool = Pool(processes=n_cores)
    fl_res = pool.starmap(fused_lasso, 
                          zip(func_concat, 
                              repeat(ev_concat_z.values), 
                              repeat(edge_list),
                              repeat(penalty))
                          )
    coef_mat = np.array([f[1][0] for f in fl_res])
    r2_vec = [f[1][1] for f in fl_res]
    return coef_mat, r2_vec


    

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
    parser.add_argument('-b', '--basis_type',
                        help='type of basis for event regressor',
                        required=False,
                        choices=['hrf', 'hrf3'],
                        default='hrf',
                        type=str)
    parser.add_argument('-d', '--main_dir',
                        help='directory where IBC data is stored',
                        required=False,
                        default='data',
                        type=str)
    parser.add_argument('-n', '--n_cores',
                        help='number of cores to use for parallel processing',
                        default = 10, 
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
             args_dict['basis_type'], args_dict['main_dir'], 
             args_dict['n_cores'], args_dict['slicetime_ref'])
