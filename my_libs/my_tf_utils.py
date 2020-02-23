import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import random
import shutil
import glob
import multiprocessing


AUTOTUNE = tf.data.experimental.AUTOTUNE
PIXELS = None
CLASS_NAMES = None



def get_label(file_path):
    global CLASS_NAMES
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    global PIXELS
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [PIXELS, PIXELS])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def prepare_for_training(ds, batch_size, cache=True, shuffle=True, shuffle_buffer_size=1000, repeat=True):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    # ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_tf_image_dataset(data_dir, batch_size, shuffle, cache, buffer_size, pixels, repeat):
    data_dir = pathlib.Path(data_dir)
    global CLASS_NAMES
    global PIXELS
    
    CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])
    PIXELS = pixels
    
    image_count = len(list(data_dir.glob("*/*.jpg")))
    STEPS_PER_EPOCH = int(np.ceil(image_count/batch_size))
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    train_ds = prepare_for_training(labeled_ds, batch_size, cache, shuffle, buffer_size, repeat)

    return train_ds, CLASS_NAMES, image_count, STEPS_PER_EPOCH


def prepare_submission(test_data, model, class_names):
    file_ids = [item.split("/")[1].split(".")[0] for item in test_data.filenames]
    preds = model.predict(test_data)
    
    submission = pd.DataFrame(data=preds, columns=list(class_names))
    submission["ID"] = file_ids
    
    return submission[["ID", "leaf_rust", "stem_rust", "healthy_wheat"]]

def show_batch(image_batch, label_batch, class_names=None):
    
    plt.figure(figsize=(10,10))
    for n in range(min(25, len(image_batch))):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        tmp_ind = label_batch[n]==1
        plt.title(class_names[tmp_ind][0].title())
        plt.axis('off')
    plt.show()

def split_class(in_path, out_path, test_split):
    random.seed(123)
    class_name = pathlib.Path(in_path).name
    
    list_files = [pathlib.Path(item).name for item in glob.glob(os.path.join(in_path, "*.jpg"))]
    
    random.shuffle(list_files)
    
    train_files = list_files[: int(len(list_files)*(1-test_split)) ]
    test_files = [item for item in list_files if item not in train_files]
    
    os.makedirs(os.path.join(out_path, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(out_path, "validation", class_name), exist_ok=True)
    
    
    in_train = [os.path.join(in_path, f) for f in train_files]
    in_test = [os.path.join(in_path, f) for f in test_files] 
    out_train = [os.path.join(out_path, "train", class_name, f) for f in train_files]
    out_test = [os.path.join(out_path, "validation", class_name, f) for f in test_files]
    
    list_train_args = list(zip(in_train, out_train))
    list_test_args = list(zip(in_test, out_test))
    
    with multiprocessing.Pool() as pool:
        _ = pool.starmap(shutil.copy, tqdm.tqdm(list_train_args))
        _ = pool.starmap(shutil.copy, tqdm.tqdm(list_test_args))


def flip_image(x, y):
    return tf.image.random_flip_left_right(x), y

def saturate_image(x, y):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

def rotate(x, y):
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), y



def augment_data(ds, list_transforms):
    assert len(list_transforms) > 0
    for transf in list_transforms:
        ds = ds.map(transf, num_parallel_calls=AUTOTUNE)
    return ds
# def zoom(x, y):

#     # Generate 20 crop settings, ranging from a 1% to 20% crop.
#     scales = list(np.arange(0.8, 1.0, 0.01))
#     boxes = np.zeros((len(scales), 4))

#     for i, scale in enumerate(scales):
#         x1 = y1 = 0.5 - (0.5 * scale)
#         x2 = y2 = 0.5 + (0.5 * scale)
#         boxes[i] = [x1, y1, x2, y2]

#     def random_crop(img):
#         # Create different crops for an image
#         crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
#         # Return a random crop
#         return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


#     choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

#     # Only apply cropping 50% of the time
#     return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x)), y