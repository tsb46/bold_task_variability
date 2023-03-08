import numpy as np
import os
import sys

# Ugly hack for sibling import
sys.path.insert(0, os.path.abspath('.'))

from scipy.io import savemat
from utils.load_write import load_func, initialize_group_array, group_concat, write_nifti
from sklearn.decomposition import PCA


def pca(fps, output_dir, mask, n_comps, n_iter=10):
    mask_bin = mask.get_data()
    print('concatenate scans')
    group_func, task_list = group_concat(fps, mask_bin)
    n_samples = group_func.shape[0]
    print('run PCA')
    # See: https://github.com/scikit-learn/scikit-learn/issues/20589
    n_oversample = 500 # increase precision
    pca = PCA(n_components = n_comps, svd_solver='randomized', n_oversamples=n_oversample)
    pca_scores = pca.fit_transform(group_func)
    output_dict = {'singular_vectors': pca.components_,
                   'singular_values': pca.singular_values_,
                   'eigenvalues': pca.explained_variance_ ,
                   'exp_var': pca.explained_variance_ratio_,
                   'pc_scores': pca_scores
                   }
    write_output(task_list, mask, output_dict, output_dir)


def write_output(task_list, mask, output_dict, output_dir):
    task_output = {}
    subj_label = task_list[0][0]
    pca_fp = f'{output_dir}/pca_{subj_label}_comp_weights.nii'
    write_nifti(output_dict['singular_vectors'], mask, pca_fp)

    ses_vals = list(set([t[1] for t in task_list]))
    for ses in ses_vals:
        task_output[ses] = {}

    for task in task_list:
        task_label = '_'.join(task[2:-1])
        task_output[task[1]][task_label] = output_dict['pc_scores'][task[-1], :]

    final_output_dict = {
        'pca': output_dict,
        'task': task_output
     }
    savemat(f'{output_dir}/pca_{subj_label}.mat', final_output_dict)







