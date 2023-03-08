import os

from config import ebrains_user, ebrains_pw
from itertools import repeat
from ebrains_drive import BucketApiClient
from multiprocessing import Pool
from pathlib import Path


def ebrains_connect(ebrains_user, ebrains_pw):
	client = BucketApiClient(ebrains_user, ebrains_pw)
	# Preprocess IBC Database
	bucket = client.buckets.get_dataset("3ca4f5a1-647b-4829-8107-588a699763c1")
	return bucket


def download_file(fp, output_dir):
	fp_path = Path(fp.name)
	fp_out = Path(output_dir, *fp_path.parts[1:])
	if os.path.isfile(fp_out):
		return
	else:
		print(fp_out)
		os.makedirs(os.path.dirname(fp_out), exist_ok=True)
		open(fp_out, 'wb').write(fp.get_content())




if __name__ == '__main__':

	output_dir = 'data'

	bucket = ebrains_connect(ebrains_user, ebrains_pw)

	# Pull all files in database
	bucket_files = bucket.ls()

	# Parallel download with 3 workers
	pool = Pool(processes=3)
	pool.starmap(download_file, zip(bucket_files, repeat(output_dir)))


