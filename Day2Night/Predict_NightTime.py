from tqdm import tqdm
from zipfile import ZipFile
from pathlib import Path

import tensorflow as tf

import os
import sys

sys.path.append(os.path.join(Path(os.path.abspath(os.path.join(os.path.dirname(__file__)))).parent.absolute(), 'Base_Model'))

import Model


def initiate_ckpt(ckpt_path: str):
    ckpt_url = 'https://drive.google.com/uc?export=download&id=10rDY7HvgYR6y5GGYi6eWMUl1O4PYxgzk&confirm=t'
    
    tf.keras.utils.get_file(os.path.join(ckpt_path, 'ckpt_d2n.zip'), ckpt_url)

    if not os.path.exists(os.path.join(ckpt_path, 'ckpt')):
        with ZipFile(os.path.join(ckpt_path, 'ckpt_d2n.zip')) as f:
            for member in tqdm(f.infolist(), desc='Extracting: '):
                f.extract(member, ckpt_path)


def resize_img(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image


def normalize_img(input_image):
    input_image = (input_image / 127.5) - 1

    return input_image


if __name__ == '__main__':
    IMPORT_CKPT = True

    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__)) + '\\Model\\Checkpoints\\')

    if IMPORT_CKPT:
        initiate_ckpt(ckpt_path)

    checkpoint_prefix = os.path.join(ckpt_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=Model.generator_optimizer,
                                    discriminator_optimizer=Model.discriminator_optimizer,
                                    generator=Model.generator,
                                    discriminator=Model.discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=1)

    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    for f in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__)) + '\\Input_Image\\')):
        try:
            img = tf.io.read_file(os.path.abspath(os.path.join(os.path.dirname(__file__)) + f'\\Input_Image\\{f}'))
            img = tf.io.decode_image(img)
            img = tf.cast(img, tf.float32)
            img = resize_img(img, 256, 256)
            img = tf.expand_dims(img, 0)
            img = normalize_img(img)
            Model.generate_images(Model.generator, img)
        except Exception: pass
