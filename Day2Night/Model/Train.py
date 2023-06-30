from pathlib import Path

import sys
import os
import time

import tensorflow as tf

sys.path.append(os.path.join(Path(os.path.abspath(os.path.join(os.path.dirname(__file__)))).parent.parent.absolute(), 'Base_Model'))

import Model


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = Model.load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = Model.load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = Model.generator(input_image, training=True)

        disc_real_output = Model.discriminator([input_image, target], training=True)
        disc_generated_output = Model.discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = Model.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = Model.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, Model.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, Model.discriminator.trainable_variables)

    Model.generator_optimizer.apply_gradients(zip(generator_gradients, Model.generator.trainable_variables))
    Model.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, Model.discriminator.trainable_variables))


def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            Model.generate_images_train(Model.generator, example_input, example_target)
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

        if (step + 1) % 5000 == 0:
            manager.save()


if __name__ == '__main__':
    BUFFER_SIZE = 17500
    BATCH_SIZE = 1
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    path_ = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\night2day\\'
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Checkpoints\\'

    train_dataset = tf.data.Dataset.list_files(os.path.join(path_, 'train\\*jpg'))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    try:
        test_dataset = tf.data.Dataset.list_files(os.path.join(path_, 'test\\*jpg'))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(os.path.join(path_, 'val\\*jpg'))
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=Model.generator_optimizer,
                                    discriminator_optimizer=Model.discriminator_optimizer,
                                    generator=Model.generator,
                                    discriminator=Model.discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=1)

    # Restore checkpoint to continue training
    # checkpoint.restore(manager.latest_checkpoint)

    fit(train_dataset, test_dataset, steps=200000)
