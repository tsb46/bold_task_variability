import glob
import json
import nibabel as nb
import numpy as np
import os


from joblib import Memory, Parallel, delayed

"""
All code is copied and adapted from:
https://github.com/hbp-brain-charting/public_analysis_code/blob/f0aca3f6c5062a8f966e881ef0ad0475405c244e/ibc_public/utils_pipeline.py#L56

"""

def _bids_filename_to_dic(filename):
	"""Make a dictionary of properties from a bids filename"""
	parts = os.path.basename(filename).split('_')
	dic = {}
	for part in parts:
		if '-' in part:
			key, value = part.split('-')
			dic[key] = value
	return dic


def _make_topup_param_file(field_maps, acq_params_file):
	"""Create the param file based on the json info attached to fieldmaps"""
	jsons = [os.path.join(os.path.dirname(fm), '../../..',
						  os.path.basename(fm).split('_')[-2] + '_epi.json')
			 for fm in field_maps]
	fw = open(acq_params_file, 'w')
	for json_ in jsons:
		info = json.load(open(json_, 'r'))
		if info['PhaseEncodingDirection'] == 'j':
			vals = '0.0 1.0 0.0 %f\n' % (info['TotalReadoutTime'] * 1000)
		elif info['PhaseEncodingDirection'] == 'j-':
			vals = '0.0 -1.0 0.0 %f\n' % (info['TotalReadoutTime'] * 1000)
		fw.write(vals)
	fw.close()



def _make_merged_filename(fmap_dir, basenames):
	"""Create filename for merged field_maps"""
	dic0 = _bids_filename_to_dic(basenames[0])
	dic1 = _bids_filename_to_dic(basenames[1])
	if 'sub' not in dic0.keys():
		dic0['sub'] = dic0['pilot']
	if 'sub' not in dic1.keys():
		dic0['sub'] = dic1['pilot']

	if 'acq' in dic0.keys():
		merged_basename = (
			'sub-' + dic0['sub'] + '_ses-' + dic0['ses'] + '_acq-' +
			dic0['acq'] + '_dir-' + dic0['dir'] + dic1['dir'] + '_epi.nii.gz')
	else:
		merged_basename = (
			'sub-' + dic0['sub'] + '_ses-' + dic0['ses'] + '_dir-' +
			dic0['dir'] + dic1['dir'] + '_epi.nii.gz')
	# merged_basename = basenames[0][:19] + basenames[1][18:]
	return(os.path.join(fmap_dir, merged_basename))


def apply_topup(main_dir, cache_dir, n_core, subject_sess=None, acq=None):
	""" Call topup on the datasets """
	mem = Memory(cache_dir)
	if subject_sess is None:
		subject_sess = [('sub-%02d, ses-%02d' % (i, j)) for i in range(0, 50)
						for j in range(0, 15)]
	Parallel(n_jobs=n_core)(
		delayed(run_topup)(mem, main_dir, subject_ses[0], subject_ses[1],
						   acq=acq)
		for subject_ses in subject_sess)


def fsl_topup(field_maps, fmri_files, mem, write_dir, modality='func'):
	""" This function calls topup to estimate distortions from field maps
	then apply the ensuing correction to fmri_files"""
	# merge the 0th volume of both fieldmaps
	fmap_dir = os.path.join(write_dir, 'fmap')
	basenames = [os.path.basename(fm) for fm in field_maps]
	merged_zeroth_fieldmap_file = _make_merged_filename(fmap_dir, basenames)
	zeroth_fieldmap_files = field_maps  # FIXME
	fslmerge_cmd = "fslmerge -t %s %s %s" % (
		merged_zeroth_fieldmap_file, zeroth_fieldmap_files[0],
		zeroth_fieldmap_files[1])
	print("\r\nExecuting '%s' ..." % fslmerge_cmd)
	print(os.system(fslmerge_cmd))
	# add one slide if the number is odd
	odd = (np.mod(nb.load(merged_zeroth_fieldmap_file).shape[2], 2) == 1)
	if odd:
		cmd = "fslroi %s /tmp/pe 0 -1 0 -1 0 1 0 -1" %\
			  merged_zeroth_fieldmap_file
		print(cmd)
		os.system(cmd)
		cmd = "fslmerge -z %s /tmp/pe %s" % (
			merged_zeroth_fieldmap_file, merged_zeroth_fieldmap_file)
		print(cmd)
		os.system(cmd)

	# TOPUP
	acq_params_file = os.path.join(fmap_dir, 'b0_acquisition_params_AP.txt')
	_make_topup_param_file(field_maps, acq_params_file)
	# import shutil
	# shutil.copy('b0_acquisition_params_AP.txt', acq_params_file)
	topup_results_basename = os.path.join(fmap_dir, 'topup_result')
	if os.path.exists(topup_results_basename):
		os.system('rm -f %s' % topup_results_basename)
	topup_cmd = (
		"topup --imain=%s --datain=%s --config=b02b0.cnf "
		"--out=%s" % (merged_zeroth_fieldmap_file, acq_params_file,
					  topup_results_basename))
	print("\r\nExecuting '%s' ..." % topup_cmd)
	print(os.system(topup_cmd))
	# apply topup to images
	func_dir = os.path.join(write_dir, modality)
	for i, f in enumerate(fmri_files):
		dcf = os.path.join(func_dir, "dc" + os.path.basename(f))
		if '-ap' in os.path.basename(f):
			inindex = 1
		elif '-pa' in os.path.basename(f):
			inindex = 2
		else:
			inindex = 2

		applytopup_cmd = (
			"applytopup --imain=%s --verbose --inindex=%s "
			"--topup=%s --out=%s --datain=%s --method=jac" % (
				f, inindex, topup_results_basename, dcf, acq_params_file))
		print("\r\nExecuting '%s' ..." % applytopup_cmd)
		print(os.system(applytopup_cmd))


def run_topup(mem, data_dir, subject, ses, acq=None):
	write_dir = os.path.join(data_dir, 'derivatives', subject, ses)
	# gather the BOLD data to be corrected
	functional_data = glob.glob(
		os.path.join(data_dir, subject, ses, 'func/*.nii.gz'))

	if functional_data == []:
		return
	if acq == 'mb6':
		functional_data = [
			fd for fd in functional_data if 'RestingState' in fd]
	functional_data.sort()

	# gather the field maps
	if acq == 'mb3':
		field_maps = [
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*acq-mb3_dir-ap_epi.nii.gz'))[-1],
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*acq-mb3_dir-pa_epi.nii.gz'))[-1]]
	elif acq == 'mb6':
		field_maps = [
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*acq-mb6_dir-ap_epi.nii.gz'))[-1],
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*acq-mb6_dir-pa_epi.nii.gz'))[-1]]
	elif acq is None:
		field_maps = [
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*dir-ap_epi.nii.gz'))[-1],
			glob.glob(
				os.path.join(data_dir, subject, ses,
							 'fmap/*dir-pa_epi.nii.gz'))[-1]]
	else:
		raise ValueError('Unknown acq %s' % acq)
	return fsl_topup(field_maps, functional_data, mem, write_dir)