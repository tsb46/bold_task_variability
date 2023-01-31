import json
import glob
import os 
import pandas as pd
import shutil


# Define renaming dictionary for output naming
output_dict = {
    'reorient': '_reorient',
    'bet': '_bet',
    'fast': '_fast',
    'wm_thres': '_wmthres',
    'flirt_anat': '_reg',
    'flirt_resamp': '_resamp',
    'slice': '_st',
    'mcflirt': '_mc',
    'coregister': '_coreg',
    'convertxfm': '_func2mni',
    'standard': '_reg',
    'bet_mask': '_mask_reg',
    'smooth': '_sm',
    'filtz': '_filtz',
    'applymask': '_mask'
}


def create_protocol_cache(subject_session, output_dir, protocol, func_list, 
                          anat_list, anatomical, ignore_cache=False):
    # Create cache .json for communicating b/w anat and func pipelines
    json_cache = {}
    for sub_ses, func, anat in zip(subject_session, func_list, anat_list):
        subj = sub_ses[0]
        if os.path.isfile(f'{os.path.abspath(output_dir)}/{protocol}_{subj}_cache.json') & ~ignore_cache:
            json_cache_subj = json.load(open(f'{os.path.abspath(output_dir)}/{protocol}_{subj}_cache.json', 'rb'))
            json_cache[subj] = json_cache_subj 
        else:
            json_cache_subj = {'func': {}, 'anat': {}}
            json_cache_subj['func']['orig'] = func
            json_cache_subj['anat']['orig'] = anat
            json_cache[subj] = json_cache_subj

        if ~anatomical:
            output_dir_anat = output_dir.replace(protocol, 'anat')
            json_cache_subj_anat = json.load(open(f'{os.path.abspath(output_dir_anat)}/anat_{subj}_cache.json', 'rb'))
            json_cache[subj]['anat'] = json_cache_subj_anat['anat'] 
    return json_cache

"""
Copied from:
https://github.com/hbp-brain-charting/public_analysis_code/blob/f0aca3f6c5062a8f966e881ef0ad0475405c244e/ibc_public/utils_data.py
"""
def get_subject_session(protocols):
    """
    Utility to get all (subject, session) for a given protocol or set
    of protocols

    Parameters
    ----------
    protocols: string or list,
               name(s) of the protocols the user wants to retrieve

    Returns
    -------
    subject_session: list of tuples
                     Each element correspondes to a (subject, session) pair
                     for the requested protocols
    """
    df = pd.read_csv('preprocess/ibc_metadata/sessions.csv', index_col=0)
    subject_session = []

    # corerce to list
    if isinstance(protocols, str):
        protocols_ = [protocols]
    else:
        protocols_ = protocols

    for protocol in protocols_:
        for session in df.columns:
            if (df[session] == protocol).any():
                subjects = df[session][df[session] == protocol].keys()
                for subject in subjects:
                    subject_session.append((subject,  session))
    return subject_session


"""
Copied from:
https://github.com/hbp-brain-charting/public_analysis_code/blob/f0aca3f6c5062a8f966e881ef0ad0475405c244e/scripts/pipeline.py
"""
def prepare_derivatives(main_dir):
    source_dir = main_dir
    output_dir = os.path.join(main_dir, 'derivatives')
    subjects = ['sub-%02d' % i for i in range(0, 16)]
    sess = ['ses-%02d' % j for j in range(0, 50)]
    modalities = ['anat', 'fmap', 'func', 'dwi']
    dirs = ([output_dir] +
            [os.path.join(output_dir, subject) for subject in subjects
             if os.path.exists(os.path.join(source_dir, subject))] +
            [os.path.join(output_dir, subject, ses) for subject in subjects
             for ses in sess
             if os.path.exists(os.path.join(source_dir, subject, ses))] +
            [os.path.join(output_dir, subject, ses, modality)
             for subject in subjects
             for ses in sess for modality in modalities
             if os.path.exists(
                os.path.join(source_dir, subject, ses, modality))])

    for dir_ in dirs:
        if not os.path.exists(dir_):
            print(dir_)
            os.mkdir(dir_)

    for subject in subjects:
        for ses in sess:
            tsv_files = glob.glob(
                os.path.join(source_dir, subject, ses, 'func', '*.tsv'))
            dst = os.path.join(output_dir, subject, ses, 'func')
            for tsv_file in tsv_files:
                shutil.copyfile(tsv_file,
                                os.path.join(dst, os.path.basename(tsv_file)))
        highres = glob.glob(
            os.path.join(source_dir, subject, 'ses-*', 'anat', '*'))

        for hr in highres:
            parts = hr.split('/')
            dst = os.path.join(
                output_dir, subject, parts[-3], 'anat', parts[-1])
            if not os.path.isfile(dst[:-3]):
                shutil.copyfile(hr, dst)
                if dst[-3:] == '.gz':
                    os.system('gunzip %s' % dst)

# rename output for file renaming
def rename_output(fp, step, ext='.nii.gz'):
    fp_strip = strip_suffix(fp)
    fp_step = f'{fp_strip}{step}{ext}'
    return os.path.abspath(fp_step)


# Remove file extension for .nii or .nii.gz
def strip_suffix(fp):
    return os.path.splitext(os.path.splitext(fp)[0])[0]

