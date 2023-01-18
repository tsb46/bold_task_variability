import config
import os
import openneuro as on


ds_num = 'ds002685'
tag_num = '1.3.1'

os.makedirs('data', exist_ok=True)
on.download(dataset=ds_num, tag=tag_num, target_dir='data')







