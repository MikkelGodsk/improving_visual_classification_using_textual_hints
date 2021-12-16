import tensorflow_datasets as tfds
import os

"""
    Requires imagenet .tar-files to be stored in /work3/s184399/imagenet/downloads/manual
    
    NOTE: In order for it not to store the unpacked dataset in ~/tensorflow_datasets, you must specify data_dir for the
     tfds.builder
     
    NOTE: When submitted as a batch-job at HPC, it required at max 1832 MB RAM, and took 1890 sec. to run using 
        at max 8 threads and 4 processes.
"""

dir = '/work3/s184399/imagenet/'

os.chdir(dir)

download_config = tfds.download.DownloadConfig(
    extract_dir = os.path.join(dir, 'downloads/extract'),
    manual_dir = os.path.join(dir, 'downloads/manual')
)

s = tfds.builder('imagenet2012', data_dir=dir)
s.download_and_prepare(download_dir=os.path.join(dir, 'downloads'),  # download path?
                       download_config=download_config)