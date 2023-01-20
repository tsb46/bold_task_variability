import os

from configobj import ConfigObj
from glob import glob

# Set config variables
config_mod_dir = 'preprocess/pipeline_config/modified'
data_base_dir = 'data'
output_dir = 'data/derivatives'
fwhm = ['[3', '3', '3]']
scratch_dir='data/tmp'

ini_fps = glob('preprocess/pipeline_config/original/*.ini')

for ini in ini_fps:
    cobj = ConfigObj(ini)
    # modified file path
    filename = os.path.basename(cobj.filename)
    cobj.filename = f'{config_mod_dir}/{filename}'
    # modify data directory
    cobj['config']['dataset_dir'] = data_base_dir
    # modify output directory
    cobj['config']['output_dir'] = output_dir
    # modify scratch directory
    cobj['config']['scratch'] = scratch_dir
    # modify fwhm smoothing param
    cobj['config']['fwhm'] = fwhm
    # write to new directory
    cobj.write()




