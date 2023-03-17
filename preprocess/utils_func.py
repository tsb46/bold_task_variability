import json
import nibabel as nb
import numpy as np
import os
import pickle
import subprocess 
import shutil
import sys

from nilearn.maskers import NiftiLabelsMasker
from nipype.interfaces import fsl

# Ugly hack for sibling import
sys.path.insert(0, os.path.abspath('.'))
from utils.file_utils import output_dict, rename_output

# Ensure output is .nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


def func_preproc(func_file, fwhm=4.0, highpass=128, tr=2):
    func_output = {}

    # Fill NaNs with zero (FSL doesn't like NaN)
    fill_nan = fsl.UnaryMaths(operation='nan')
    fill_nan.inputs.in_file = func_file
    fill_nan.inputs.out_file = rename_output(func_file, output_dict['fill_nan'])
    fill_nan_res = fill_nan.run()
    func_output['fill_nan'] = fill_nan_res.outputs.out_file

    # Transform functional to resampled (2mm) MNI space using ApplyXFM (in FSL)
    flirt_resamp = fsl.FLIRT(apply_xfm=True, uses_qform=True)
    flirt_resamp.inputs.in_file = func_output['fill_nan']
    flirt_resamp.inputs.reference = os.path.abspath('preprocess/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    flirt_resamp.inputs.out_file = rename_output(func_output['fill_nan'], output_dict['func_resample'])
    flirt_resamp_res = flirt_resamp.run()
    os.remove(flirt_resamp_res.outputs.out_matrix_file)
    func_output['resample_out'] = flirt_resamp_res.outputs.out_file

    # Extract ROI time courses
    parc_fp = os.path.abspath('preprocess/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
    parc = NiftiLabelsMasker(labels_img=parc_fp)
    parc_ts = parc.fit_transform(func_output['resample_out'])
    out_file = rename_output(func_output['resample_out'], output_dict['extract_roi'], ext='.txt')
    func_output['extract_roi'] = out_file
    np.savetxt(out_file, parc_ts)



    return func_output


