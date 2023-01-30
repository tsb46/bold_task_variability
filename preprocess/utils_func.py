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


def func_preproc(func_list, json_cache, json_output, metadata, fwhm=3.0, highpass=128):
    # Use json_cache to ensure functional preprocessing is not re-run if output exists

    # Define slice timing node that takes as input the 
    # saved .txt slice time file (see above)
    if json_cache['func'].get('slicetime') is None:
        slicetime_list = []
        for func_file in func_list:
            slicetimer = fsl.SliceTimer(
                custom_timings=metadata['slicetime'], 
                time_repetition=metadata['tr']
            ) 
            slicetimer.inputs.in_file = func_file
            slicetimer.inputs.out_file = rename_output(func_file, output_dict['slice'])
            slicetimer_res = slicetimer.run()
            slicetime_list.append(slicetimer_res.outputs.slice_time_corrected_file)
        json_cache['func']['slicetime'] = slicetime_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)


    # Mcflirt motion correction
    if json_cache['func'].get('mcflirt') is None:
        mcflirt_list = {'map': [], 'mean_vol': []}
        for func_file in json_cache['func']['slicetime']:
            mcflirt = fsl.MCFLIRT(mean_vol=True, save_plots=True)
            mcflirt.inputs.in_file = func_file
            mcflirt.inputs.out_file = rename_output(func_file, output_dict['mcflirt'])
            mcflirt_res = mcflirt.run()
            # weird renaming of mean vol, rename
            mean_vol_rename = f'{rename_output(func_file, output_dict["mcflirt"], ext="")}_mean_reg.nii.gz'
            os.rename(mcflirt_res.outputs.mean_img, mean_vol_rename)
            mcflirt_list['map'].append(mcflirt_res.outputs.out_file)
            mcflirt_list['mean_vol'].append(mean_vol_rename)
        json_cache['func']['mcflirt'] = mcflirt_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # Coregister functional with T1w
    if json_cache['func'].get('coregister') is None:
        coregister_list = {'map': [], 'omat': []}
        for func_file in json_cache['func']['mcflirt']['mean_vol']:
            epireg = fsl.EpiReg()
            epireg.inputs.epi = func_file
            epireg.inputs.t1_head=json_cache['anat']['reorient']
            epireg.inputs.t1_brain=json_cache['anat']['bet']
            # epireg expects the wmseg output as a suffix to the epi image (weird)
            # rename for now
            wmseg = f'{rename_output(func_file, output_dict["coregister"], ext="")}_fast_wmseg.nii.gz'
            shutil.copyfile(json_cache['anat']['wm_thres'], wmseg)
            epireg.inputs.wmseg = wmseg
            epireg.inputs.out_base = rename_output(func_file, output_dict['coregister'], ext='')
            epireg_res = epireg.run()
            coregister_list['map'].append(epireg_res.outputs.out_file)
            coregister_list['omat'].append(epireg_res.outputs.epi2str_mat)
        json_cache['func']['coregister'] = coregister_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # Concatenate affine transform matrices (func2struct & struct2MNI)
    if json_cache['func'].get('convertxfm') is None:
        convertxfm_list = []
        for func_omat in json_cache['func']['coregister']['omat']:
            convertxfm = fsl.ConvertXFM(concat_xfm=True)
            convertxfm.inputs.in_file = func_omat
            convertxfm.inputs.in_file2=json_cache['anat']['flirt']['mat']
            convertxfm.inputs.out_file=rename_output(func_omat, output_dict['convertxfm'], ext='.mat')
            convertxfm_res = convertxfm.run()
            convertxfm_list.append(convertxfm_res.outputs.out_file)
        json_cache['func']['convertxfm'] = convertxfm_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # Transform functional to MNI space using ApplyXFM4D (in FSL)
    if json_cache['func'].get('standard') is None:
        standard_list = []
        for func_file, func2mni_mat in zip(json_cache['func']['mcflirt']['map'], 
                                           json_cache['func']['convertxfm']):
            fsldir = os.environ['FSLDIR']
            reference_file = json_cache['anat']['flirt_resamp']
            standard_out = rename_output(func_file, output_dict['standard'])
            applyxfm_fsl = f'{fsldir}/bin/applyxfm4D'
            applyxfm_cmd = f'{applyxfm_fsl} {func_file} {reference_file} {standard_out} {func2mni_mat} -singlematrix'
            subprocess.run([applyxfm_fsl, func_file, reference_file, standard_out, func2mni_mat, '-singlematrix'])
            standard_list.append(standard_out)
        json_cache['func']['standard'] = standard_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # 3mm FWHM isotropic smoothing
    if json_cache['func'].get('smooth') is None:
        smooth_list = []
        for func_file in json_cache['func']['standard']:
            smooth = fsl.Smooth(fwhm=fwhm)
            smooth.inputs.in_file = func_file
            smooth.inputs.smoothed_file=rename_output(func_file, output_dict['smooth'])
            smooth_res = smooth.run()
            smooth_list.append(smooth_res.outputs.smoothed_file)
        json_cache['func']['smooth'] = smooth_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # High-pass temporal filtering to remove trends - gaussian filter (in sigma)
    # In addition, z-score normalize along the time dimension
    if json_cache['func'].get('temporal_filt_z') is None:
        filt_list = []
        # formula for hertz to sigma 
        hpf_sigma = (highpass / 2.0) / metadata['tr']
        for func_file in json_cache['func']['smooth']:
            # High-pass filter
            filt = fsl.TemporalFilter(highpass_sigma=hpf_sigma)
            filt.inputs.in_file = func_file
            filt.inputs.out_file=rename_output(func_file, output_dict['filtz'])
            filt_res = filt.run()
            # z-score normalize 
            zscore_norm = Function(input_names=['in_file', 'out_file'], 
                                   output_names=['out_file'],
                                   function=zscore4d)
            zscore_norm.inputs.in_file = filt.inputs.out_file
            zscore_norm.inputs.out_file = filt.inputs.out_file
            zscore_norm.run()
            filt_list.append(rename_output(func_file, output_dict['filtz']))
        json_cache['func']['temporal_filt_z'] = filt_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    # apply mask to functional image (from BET mask in MNI space)
    if json_cache['func'].get('func_mask') is None:
        mask_list = []
        for func_file in json_cache['func']['temporal_filt_z']:
            applymask = fsl.ApplyMask()
            applymask.inputs.in_file = func_file
            applymask.inputs.mask_file = json_cache['anat']['bet_mask']
            applymask.inputs.out_file=rename_output(func_file, output_dict['applymask'])
            applymask_res = applymask.run()
            mask_list.append(applymask_res.outputs.out_file)
        json_cache['func']['func_mask'] = mask_list
        json.dump(json_cache, open(json_output, 'w'), ensure_ascii=False, indent=4)

    return json_cache


# Function for z-score normalizing along the time dimension
def zscore4d(in_file, out_file):

    import nibabel as nb
    from scipy.stats import zscore

    # Load the data
    nii = nb.load(in_file)
    data = nii.get_data()

    # Zscore along the time dimension
    newdata = zscore(data, axis=3)

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


