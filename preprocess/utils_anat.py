import os
import pickle 

from file_utils import output_dict, rename_output
from nipype import Node, Workflow, Function, MapNode
from nipype.interfaces import fsl
from pathlib import Path


# Ensure output is .nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


def anat_preproc(anat_file, json_cache):
    # Use json_cache to ensure anatomical preprocessing is not re-run if output exists:
    # BET - Skullstrip anatomical Image
    if json_cache['anat'].get('bet') is None:
        bet_anat = fsl.BET(frac=0.5, mask=True)
        bet_anat.inputs.in_file = anat_file
        bet_anat.inputs.out_file = rename_output(anat_file, output_dict['bet'])
        bet_anat_res = bet_anat.run()
        json_cache['anat']['bet'] = {}
        json_cache['anat']['bet']['map'] = bet_anat_res.outputs.out_file
        json_cache['anat']['bet']['mask'] = bet_anat_res.outputs.mask_file

    # Resample to 2mm
    if json_cache['anat'].get('anat_resample') is None:
        flirt_resamp = fsl.FLIRT(apply_isoxfm=2)
        flirt_resamp.inputs.in_file = anat_file
        flirt_resamp.inputs.reference = anat_file
        flirt_resamp.inputs.out_file = rename_output(anat_file, output_dict['anat_resample'])
        out_matrix_fp = rename_output(anat_file, output_dict['anat_resample'], ext='.mat')
        flirt_resamp.out_matrix_file = out_matrix_fp
        flirt_resamp_res = flirt_resamp.run()
        json_cache['anat']['anat_resample'] = {}
        json_cache['anat']['anat_resample']['map'] = flirt_resamp_res.outputs.out_file
        # Flirt saves output matrix in base directory (seems to be an issue related to the FAST issue above), 
        # move to results directory
        os.rename(flirt_resamp_res.outputs.out_matrix_file, out_matrix_fp)
        json_cache['anat']['anat_resample']['mat'] = out_matrix_fp

    # Send BET mask to new resampled space
    if json_cache['anat'].get('bet_mask') is None:
        flirt_mask = fsl.ApplyXFM(apply_xfm=True)
        flirt_mask.inputs.in_file = json_cache['anat']['bet']['mask']
        flirt_mask.inputs.reference = json_cache['anat']['anat_resample']['map']
        flirt_mask.inputs.in_matrix_file = json_cache['anat']['anat_resample']['mat'] 
        flirt_mask.inputs.out_file = rename_output(json_cache['anat']['bet']['mask'], output_dict['bet_mask'])
        flirt_mask_res = flirt_mask.run()
        # Binarize after resampling ()
        flirt_mask_thres = fsl.Threshold(thresh=0.9, args='-bin')
        flirt_mask_thres.inputs.in_file = flirt_mask.inputs.out_file
        flirt_mask_thres.inputs.out_file = flirt_mask.inputs.out_file
        flirt_mask_thres_res = flirt_mask_thres.run()
        json_cache['anat']['bet_mask'] = flirt_mask_thres_res.outputs.out_file
        # Remove .mat output
        bet_mask_out = rename_output(json_cache['anat']['bet']['mask'], '_flirt', ext='.mat')
        bet_mask_fp = Path(bet_mask_out)
        tmp = os.path.join(*(bet_mask_fp.parts[:-5] + (bet_mask_fp.parts[-1],)))
        os.remove(tmp)
        
    return json_cache


