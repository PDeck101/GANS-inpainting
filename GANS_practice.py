import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os, time
from IPython import display
import imageio
import glob
import pathlib

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# https://scikit-image.org/docs/dev/user_guide/data_types.html

'''
Instructs TensorFlow to use the systems GPU memory as needed. This is due to the development system using a GTX 970, 
which run slow when over 3.5 gigs of the 4gigs of memory available is used.
'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def get_img(img_path):
    return tf.io.decode_image(tf.io.read_file(img_path), channels=3, dtype=tf.dtypes.uint8)


def decode_img(img):
    decode = tf.io.decode_image(tf.io.read_file(img), channels=3, dtype=tf.dtypes.uint8)
    decode = tf.image.convert_image_dtype(decode, tf.float32)
    return tf.image.resize(decode, [256, 256])


masked_img_path = './image_screenshot_03.05.2020.png'
dataset_img_path = 'G:/places/data_256/f/field/wild/00000001.jpg'
tf_img_masked = decode_img(masked_img_path)
tf_img_dataset = decode_img(dataset_img_path)
print("Tensor Flow image shape png:", tf_img_masked.shape)
print("Decoded png:\n", tf_img_masked)
print("Tensor Flow image shape jpg:", tf_img_dataset.shape)
#print("Decoded jpg:\n", tf_img_dataset)


# Create dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
# https://www.tensorflow.org/tutorials/load_data/images
data_dir = pathlib.Path('G:/places/data_256/f/field')
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
BATCH_SIZE = 15 #256
EPOCHS = 25 #50

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
# print out 5 image paths in the dataset
for f in list_ds.take(5):
    print(f.numpy())


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


# convert the image to a 3D unit8 tensor
def decode_dataset_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [256, 256])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_dataset_img(img)
    return img, label

# making parallel calls so multiple images are loaded and processed in parallel
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
'''
print("labeled ds", labeled_ds)
for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("First record", image)
    print("Label: ", label.numpy())
'''

# shuffles and batches the dataset
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_dataset = prepare_for_training(labeled_ds)
train_images, label_batch = next(iter(train_dataset))

'''
normalize the dataset and masked image RGB values in the range of [-1, 1]
RGB values range from 0 to 255, where 0 is black and 255 is white. 255/2 = 127.5 
'''
#tf_img_masked = tf_img_masked.reshape(tf_img_masked.shape[0], 256, 256, 3).astype('float32')
#tf_img_masked = tf.reshape(tf_img_masked.shape[0], 256, 256, 3).astype('float32')
tf_img_masked = (tf_img_masked - 127.5) / 127.5
train_images = (train_images - 127.5) / 127.5
print("tf masked image shape:", tf_img_masked.shape)


def make_generator_model():
    model = tf.keras.Sequential(name='Generator')
    model.add(layers.Dense(64*64*256, use_bias=False, input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((64, 64, 256)))
    assert model.output_shape == (None, 64, 64, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 256, 256, 3)

    return model


generator = make_generator_model()
print(generator.summary())
#plot_model(generator, to_file='generator.png', show_shapes=True, show_layer_names=True)


def make_discriminator_model():
    model = tf.keras.Sequential(name='Discriminator')
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(256, 256, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
print(discriminator.summary())

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator, discriminator=discriminator)


@tf.function
def train_step(dataset):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(tf_img_masked, training=True)

        real_output = discriminator(train_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # gen_train_var = generator.trainable_variables
    # dis_train_var = discriminator.trainable_variables

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(epochs, dataset):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_img(generator, epoch + 1, tf_img_masked)

        # Save the model ever 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_img(generator, epoch, tf_img_masked)


def generate_and_save_img(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(1, 1))
    plt.imshow(predictions[1, :, :, 0] * 127.5 + 127.5, cmap='rgb')
    plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

print('Train images shape:', train_images.shape)
train(EPOCHS, train_images)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

animate_file = 'inpainting.gif'

with imageio.get_writer(animate_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


