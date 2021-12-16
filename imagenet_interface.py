import tensorflow_datasets as tfds
import numpy as np
import os


def get_dataset(split, use_subset=True, prefetch=True, buffer_size=20):
    dsdir = '/work3/s184399/imagenet'
    dsname = 'imagenet2012' + ('_subset' if use_subset else '')
    if prefetch:
        return tfds.load(dsname, data_dir=dsdir, split=split).prefetch(buffer_size)
    else:
        return tfds.load(dsname, data_dir=dsdir, split=split)


def get_dataset_iterator(split, use_subset=True, buffer_size=20):
    return get_dataset(split, use_subset=use_subset, buffer_size=buffer_size).__iter__()


def get_dataset_labels():
    labeldir = os.getcwd()
    labelfile = 'labels.labels.txt'
    return np.loadtxt(os.path.join(labeldir, labelfile), dtype=np.str)
