import os
import pickle 

from file_utils import output_dict, rename_output
from nipype import Node, Workflow, Function, MapNode
from nipype.interfaces import fsl


# Ensure output is .nii.gz
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')


def anat_preproc(anat_file, json_cache):
    # Use json_cache to ensure anatomical preprocessing is not re-run if output exists:
    # Reorient 2 standard
    if json_cache['anat'].get('reorient') is None:
        reorient = fsl.utils.Reorient2Std()
        reorient.inputs.in_file = anat_file
        reorient.inputs.out_file = rename_output(anat_file, output_dict['reorient'])
        reorient_res = reorient.run()
        json_cache['anat']['reorient'] = reorient_res.outputs.out_file

    # BET - Skullstrip anatomical Image
    if json_cache['anat'].get('bet') is None:
        bet_anat = fsl.BET(frac=0.5,robust=True, mask=True)
        bet_anat.inputs.in_file = json_cache['anat']['reorient']
        bet_anat.inputs.out_file = rename_output(json_cache['anat']['reorient'], 
                                                 output_dict['bet'])
        bet_anat_res = bet_anat.run()
        json_cache['anat']['bet'] = {}
        json_cache['anat']['bet']['map'] = bet_anat_res.outputs.mask_file
        json_cache['anat']['bet']['mask'] = bet_anat_res.outputs.out_file


    # FAST - Image Segmentation
    if json_cache['anat'].get('fast') is None:
        fast = fsl.FAST()
        fast.inputs.in_files = json_cache['anat']['bet']['map']
        fast.inputs.out_basename = rename_output(json_cache['anat']['bet']['map'], 
                                                 output_dict['fast'], ext='')
        # Nipype FAST issue with writing out tissue_class_map - Ignore
        # https://github.com/nipy/nipype/issues/3311
        try: 
            fast_res = fast.run()
            fast_out = fast.outputs.tissue_class_map
        except FileNotFoundError: 
            fast_out = f'{fast.inputs.out_basename}_seg.nii.gz'

        json_cache['anat']['fast'] = {}
        json_cache['anat']['fast']['seg'] = fast_out
        json_cache['anat']['fast']['wm_pve'] = f'{fast.inputs.out_basename}_pve_2.nii.gz'

    # Threshold white matter partial volume
    if json_cache['anat'].get('wm_seg') is None:
        wm_thres = fsl.Threshold(thresh=0.5, args='-bin')
        wm_thres.inputs.in_file = json_cache['anat']['fast']['wm_pve']
        wm_thres.inputs.out_file = rename_output(json_cache['anat']['fast']['wm_pve'], 
                                                 output_dict['wm_thres'])
        wm_thres_res = wm_thres.run()
        json_cache['anat']['wm_thres'] = wm_thres_res.outputs.out_file
    
    # FLIRT affine registration to MNI template
    if json_cache['anat'].get('flirt') is None:
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = json_cache['anat']['bet']['map']
        flirt.inputs.reference = f'{os.environ["FSLDIR"]}/data/standard/MNI152_T1_1mm_brain.nii.gz'
        flirt.inputs.out_file = rename_output(json_cache['anat']['bet']['map'], output_dict['flirt_anat'])
        out_matrix_fp = rename_output(json_cache['anat']['bet']['map'], output_dict['flirt_anat'], ext='.mat')
        flirt.out_matrix_file = out_matrix_fp
        flirt_res = flirt.run()
        # Flirt saves output matrix in base directory (seems to be an issue related to the FAST issue above), 
        # move to results directory
        os.rename(flirt_res.outputs.out_matrix_file, out_matrix_fp)
        json_cache['anat']['flirt'] = {}
        json_cache['anat']['flirt']['map'] = flirt_res.outputs.out_file
        json_cache['anat']['flirt']['mat'] = out_matrix_fp

    # FLIRT affine registration to MNI template
    if json_cache['anat'].get('flirt_resamp') is None:
        flirt_resamp = fsl.FLIRT(apply_isoxfm=1.5)
        flirt_resamp.inputs.in_file = json_cache['anat']['flirt']['map']
        flirt_resamp.inputs.reference = json_cache['anat']['flirt']['map']
        flirt_resamp.inputs.out_file = rename_output(json_cache['anat']['flirt']['map'], output_dict['flirt_resamp'])
        flirt_resamp_res = flirt_resamp.run()
        json_cache['anat']['flirt_resamp'] = flirt_resamp_res.outputs.out_file

    # Send BET mask to standard space
    if json_cache['anat'].get('bet_mask') is None:
        flirt_mask = fsl.ApplyXFM(apply_xfm=True)
        flirt_mask.inputs.in_file = json_cache['anat']['bet']['mask']
        flirt_mask.inputs.reference = json_cache['anat']['flirt_resamp']
        flirt_mask.inputs.in_matrix_file = json_cache['anat']['flirt']['mat'] 
        flirt_mask.inputs.out_file = rename_output(json_cache['anat']['flirt_resamp'], output_dict['bet_mask'])
        flirt_mask_res = flirt_mask.run()
        # Binarize after resampling ()
        flirt_mask_thres = fsl.Threshold(thresh=0.9, args='-bin')
        flirt_mask_thres.inputs.in_file = flirt_mask.inputs.out_file
        flirt_mask_thres.inputs.out_file = flirt_mask.inputs.out_file
        flirt_mask_thres_res = flirt_mask_thres.run()
        json_cache['anat']['bet_mask'] = flirt_mask_thres_res.outputs.out_file
        

    return json_cache





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


