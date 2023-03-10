import json
import nibabel as nb
import os
import pickle
import subprocess 
import shutil

from file_utils import output_dict, rename_output
from nipype.interfaces import fsl
from nipype.interfaces.utility import Function



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
    flirt_resamp.inputs.reference = os.path.abspath('preprocess/MNI152_T1_2mm_brain_mask.nii.gz')
    flirt_resamp.inputs.out_file = rename_output(func_output['fill_nan'], output_dict['func_resample'])
    flirt_resamp_res = flirt_resamp.run()
    os.remove(flirt_resamp_res.outputs.out_matrix_file)
    func_output['resample_out'] = flirt_resamp_res.outputs.out_file


    # 4mm FWHM isotropic smoothing
    smooth = fsl.Smooth(fwhm=fwhm)
    smooth.inputs.in_file = func_output['resample_out']
    smooth.inputs.smoothed_file=rename_output(func_output['resample_out'], output_dict['smooth'])
    smooth_res = smooth.run()
    func_output['smooth'] = smooth_res.outputs.smoothed_file

    # High-pass temporal filtering to remove trends - gaussian filter (in sigma)
    # In addition, z-score normalize along the time dimension
    # formula for hertz to sigma 
    hpf_sigma = (highpass / 2.0) / tr
    # High-pass filter
    filt = fsl.TemporalFilter(highpass_sigma=hpf_sigma)
    filt.inputs.in_file = func_output['smooth']
    filt.inputs.out_file=rename_output(func_output['smooth'], output_dict['filtz'])
    filt_res = filt.run()
    # z-score normalize 
    zscore_norm = Function(input_names=['in_file', 'out_file'], 
                           output_names=['out_file'],
                           function=zscore4d)
    zscore_norm.inputs.in_file = filt.inputs.out_file
    zscore_norm.inputs.out_file = filt.inputs.out_file
    zscore_norm.run()
    func_output['temporal_filt_z'] = filt.inputs.out_file

    # apply mask to functional image and fill NaNs (from BET mask in MNI space)
    applymask = fsl.ApplyMask()
    applymask.inputs.in_file = func_output['temporal_filt_z']
    applymask.inputs.mask_file = os.path.abspath('preprocess/MNI152_T1_2mm_brain_mask.nii.gz')
    applymask.inputs.out_file=rename_output(func_output['temporal_filt_z'], 
                                            output_dict['applymask'])
    applymask_res = applymask.run()
    func_output['func_mask'] = applymask_res.outputs.out_file

    return func_output


# Function for z-score normalizing along the time dimension
def zscore4d(in_file, out_file):

    import nibabel as nb
    import numpy as np
    from scipy.stats import zscore

    # Load the data
    nii = nb.load(in_file)
    data = nii.get_data()

    # Zscore along the time dimension
    newdata = zscore(data, axis=3)
    np.nan_to_num(newdata, copy=False)

    # Save the new data in a new NIfTI image
    nb.Nifti1Image(newdata, nii.affine, nii.header).to_filename(out_file)




    # flirt -in data/dataset_nki/anat/proc1_bet/$filename -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
        # -out data/dataset_nki/anat/proc2_affine/$filename -omat data/dataset_nki/anat/proc2_affine/$filename.mat

# # FLIRT - pre-alignment of functional images to anatomical images
# coreg_pre = MapNode(fsl.FLIRT(dof=6),
#                  name="coreg_pre", iterfield=['in_file', 'reference'])

# # FLIRT - coregistration of functional images to anatomical images with BBR
# coreg_bbr = MapNode(fsl.FLIRT(dof=6,
#                        cost='bbr',
#                        schedule=os.path.join(os.getenv('FSLDIR'),
#                                     'etc/flirtsch/bbr.sch')),
#                  name="coreg_bbr", iterfield=['in_file', 'reference'])

# # Apply coregistration warp to functional images
# applywarp = MapNode(fsl.FLIRT(interp='spline',
#                        apply_isoxfm=1.5),
#                        name="applywarp", iterfield=['in_file', 'reference'])

# # Apply coregistration warp to mean file
# applywarp_mean = MapNode(fsl.FLIRT(interp='spline',
#                             apply_isoxfm=1.5),
#                  name="applywarp_mean", iterfield=['in_file', 'reference'])


