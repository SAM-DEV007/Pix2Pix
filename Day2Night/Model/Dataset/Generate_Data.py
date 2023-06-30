from tqdm import tqdm

import tensorflow as tf

import tarfile
import os


if __name__ == '__main__':
    dataset_url = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz'
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    tf.keras.utils.get_file(os.path.join(dataset_path, 'night2day.tar.gz'), dataset_url)

    if not os.path.exists(os.path.join(dataset_path, 'night2day')):
        with tarfile.open(os.path.join(dataset_path, 'night2day.tar.gz')) as f:
            for member in tqdm(iterable=f.getmembers(), total=len(f.getmembers()), desc='Extracting: '):
                f.extract(member, dataset_path)
