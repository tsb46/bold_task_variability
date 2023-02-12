import numpy as np

from scipy.io import savemat
from load_write import load_func, initialize_group_array, write_nifti
from file_utils import ibc_get_task
from sklearn.decomposition import PCA


def pca(fps, output_dir, mask, n_comps, n_iter=10):
    mask_bin = mask.get_data()
    print('concatenate scans')
    group_func, task_list = group_concat(fps, mask_bin)
    n_samples = group_func.shape[0]
    print('run PCA')
    pca = PCA(n_components = n_comps)
    pca_scores = pca.fit_transform(group_func)
    output_dict = {'singular_vectors': pca.components_,
                   'singular_values': pca.singular_values_,
                   'eigenvalues': pca.explained_variance_ ,
                   'exp_var': pca.explained_variance_ratio_,
                   'pc_scores': pca_scores
                   }
    write_output(task_list, mask, output_dict, output_dir)


def group_concat(fps, mask):
    # get # of voxels from mask
    mask_n = len(np.nonzero(mask)[0])
    # Initialize a zero array for faster stacking
    group_func = initialize_group_array(fps, mask_n)
    # Loop through functional scans per subject and stack
    indx = 0 # initialize index
    group_str = [] # keep track of what scan is stacked on what
    for scan_fp in fps:
        func = load_func(scan_fp, mask)
        func_n = func.shape[0]
        group_func[indx:(indx+func_n), :] = func
        group_str.append(ibc_get_task(scan_fp) + [range(indx, indx+func_n)])
        indx += func_n

    return group_func, group_str


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







