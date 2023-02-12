import nibabel as nb
import numpy as np


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


def load_func(fp, mask):
    # Load scan
    nifti = nb.load(fp, keep_file_open = True)
    nifti_data = nifti.get_data()
    nifti.uncache()
    nifti_data = convert_2d(mask, nifti_data)
    return nifti_data


def write_nifti(data, mask, output_fp):
    mask_bin = mask.get_data() > 0
    nifti_4d = np.zeros(mask.shape + (data.shape[0],), 
                        dtype=data.dtype)
    nifti_4d[mask_bin, :] = data.T

    nifti_out = nb.Nifti2Image(nifti_4d, mask.affine)
    nb.save(nifti_out, output_fp)


