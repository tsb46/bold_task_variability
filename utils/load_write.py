import nibabel as nb
import numpy as np
import os
import sys

# Ugly hack for sibling import
sys.path.insert(0, os.path.abspath('.'))

from utils.file_utils import ibc_get_task

# global variables
subj_ses_dict = {
    'sub-01': ['ses-03', 'ses-04', 'ses-05', 'ses-07', 
              'ses-14', 'ses-15', 'ses-18', 'ses-19', 
              'ses-20', 'ses-22', 'ses-23', 'ses-24'],
    'sub-02': ['ses-01', 'ses-04', 'ses-05', 'ses-06'],
    'sub-04': ['ses-01', 'ses-02', 'ses-03', 'ses-04',
               'ses-11', 'ses-12', 'ses-15', 'ses-16', 
               'ses-17', 'ses-18', 'ses-19', 'ses-21', 
               'ses-22', 'ses-23', 'ses-24', 'ses-25', 
               'ses-26'],
    'sub-05': ['ses-01', 'ses-02', 'ses-03', 'ses-04',
               'ses-11', 'ses-12', 'ses-15', 'ses-16', 
               'ses-17', 'ses-18', 'ses-19', 'ses-21',
               'ses-23', 'ses-24', 'ses-25', 'ses-26']

}

def convert_2d(mask, nifti_data):
    nonzero_indx = np.nonzero(mask)
    nifti_2d = nifti_data[nonzero_indx]
    return nifti_2d.T


def initialize_group_array(fps, mask_n):
    n_t = 0
    for fp in fps:
        nifti = nb.load(fp)
        n_t += nifti.header['dim'][4]
    group_array = np.zeros((n_t, mask_n))
    return group_array


def group_concat(fps, mask, return_length=False):
    # get # of voxels from mask
    mask_n = len(np.nonzero(mask)[0])
    # Initialize a zero array for faster stacking
    group_func = initialize_group_array(fps, mask_n)
    # Loop through functional scans per subject and stack
    indx = 0 # initialize index
    group_str = [] # keep track of what scan is stacked on what
    func_len = [] # keep track of length of each scan
    for scan_fp in fps:
        func = load_func(scan_fp, mask)
        func_n = func.shape[0]
        func_len.append(func_n)
        group_func[indx:(indx+func_n), :] = func
        group_str.append(ibc_get_task(scan_fp) + [range(indx, indx+func_n)])
        indx += func_n

    if return_length:
        return group_func, group_str, func_len
    else:
        return group_func, group_str


def load_func(fp, mask):
    # Load scan
    nifti = nb.load(fp, keep_file_open = True)
    nifti_data = nifti.get_data()
    nifti.uncache()
    nifti_data = convert_2d(mask, nifti_data)
    return nifti_data


def write_nifti(data, mask, output_fp):
    mask_bin = mask.get_fdata() > 0
    nifti_4d = np.zeros(mask.shape + (data.shape[0],), 
                        dtype=data.dtype)
    nifti_4d[mask_bin, :] = data.T

    nifti_out = nb.Nifti2Image(nifti_4d, mask.affine)
    nb.save(nifti_out, output_fp)


