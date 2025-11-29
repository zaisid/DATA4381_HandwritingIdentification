
import os
import shutil
import zipfile

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from tensorflow import data as tf_data
import os
import random

w_ids = ['w0001',  'w0002',  'w0003',  'w0004',  'w0005',  'w0006',  'w0009',  'w0010',  'w0011',  'w0012',  'w0013',  'w0015',  'w0016',  'w0017',  'w0018',  'w0020',  'w0022',  'w0023',  'w0024',  'w0025',  'w0026',  'w0027',  'w0028',  'w0029',  'w0030',  'w0031',  'w0032',  'w0033',  'w0034',  'w0035',  'w0036',  'w0038',  'w0043',  'w0061',  'w0062',  'w0063',  'w0064',  'w0066',  'w0069',  'w0070',  'w0071',  'w0073',  'w0074',  'w0075',  'w0076',  'w0077',  'w0078',  'w0080',  'w0082',  'w0083',  'w0085',  'w0086',  'w0087',  'w0088',  'w0089',  'w0091',  'w0092',  'w0093',  'w0094',  'w0095',  'w0121',  'w0122',  'w0123',  'w0124',  'w0125',  'w0126',  'w0128',  'w0129',  'w0130',  'w0131',  'w0133',  'w0134',  'w0135',  'w0136',  'w0137',  'w0138',  'w0139',  'w0142',  'w0143',  'w0144',  'w0145',  'w0147',  'w0148',  'w0149',  'w0151',  'w0152',  'w0153',  'w0154',  'w0155',  'w0156']


def stratify_split(source_dir,split=[15,6,6]):
    '''Re-organizes images in writer folders into train/val/test folders with writer sub-directories'''
    for writer in w_ids:
        writer_path = f"{source_dir}/{writer}"

        writer_images = os.listdir(writer_path) #list of all images for writer

        #shuffle images
        random.shuffle(writer_images)

        tr,va,te = split
        #15/6/6 split
        train_set = writer_images[:tr] 
        val_set = writer_images[tr:tr+va]
        test_set = writer_images[tr+va:]

        #create sub directories in writer folders
        train_folder = os.path.join(source_dir, "train") #creates path for train folder
        val_folder = os.path.join(source_dir, "val")
        test_folder = os.path.join(source_dir, "test")
        os.makedirs(train_folder, exist_ok=True) #creates train folder
        os.makedirs(val_folder, exist_ok=True) # creates source/Data/val
        os.makedirs(test_folder, exist_ok=True)


        for image in train_set: #loop over every image for training in og place
            path = os.path.join(writer_path,image) #image og path
            destination = os.path.join(train_folder,writer) 
            os.makedirs(destination, exist_ok=True) #creates source/Data/train/writer_id

            shutil.move(path, destination) #move to writer directory in train folder

        for image in val_set:
            path = os.path.join(writer_path,image)
            destination = os.path.join(val_folder,writer)
            os.makedirs(destination, exist_ok=True)

            shutil.move(path, destination)

        for image in test_set:
            path = os.path.join(writer_path,image)
            destination = os.path.join(test_folder,writer)
            os.makedirs(destination, exist_ok=True)

            shutil.move(path, destination)


def stratify_split_alt(source_dir, split=(15, 6, 6)):
    """Re-organizes images in writer folders into train/val/test folders with writer sub-directories."""
    train_folder = os.path.join(source_dir, "train")
    val_folder = os.path.join(source_dir, "val")
    test_folder = os.path.join(source_dir, "test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for writer in w_ids:
        writer_path = os.path.join(source_dir, writer)
        if not os.path.exists(writer_path):
            print(f"⚠️ Writer directory not found: {writer_path}")
            continue

        writer_images = [f for f in os.listdir(writer_path) if f.endswith(".png")]
        if not writer_images:
            continue

        random.shuffle(writer_images)
        tr, va, te = split
        train_set = writer_images[:tr]
        val_set = writer_images[tr:tr+va]
        test_set = writer_images[tr+va:]

        for subset, folder, image_set in zip(
            ["train", "val", "test"],
            [train_folder, val_folder, test_folder],
            [train_set, val_set, test_set],
        ):
            destination = os.path.join(folder, writer)
            os.makedirs(destination, exist_ok=True)

            for image in image_set:
                src = os.path.join(writer_path, image)
                if os.path.exists(src):
                    shutil.move(src, destination)
                else:
                    print(f"⚠️ File not found, skipping: {src}")


def del_old_writer_directories(source_dir):
    '''Deletes original writer-organized directories (for use after creating train/val/test directories with stratify_split())'''
    for writer in w_ids: #delete old writer folders
        try:
            os.rmdir(f"{source_dir}/{writer}")
        except:
            pass


def organize_unzipped_files(source_dir):
    '''Organizes raw unzipped files into writer folders'''
    dest_dir = source_dir

    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            # Extract writer ID from filename, assumes format "wxxxx_syy_pzzz_rqq"
            writer_id = filename.split("_")[0]  # 'wxxxx'

            writer_folder = os.path.join(dest_dir, writer_id)
            os.makedirs(writer_folder, exist_ok=True)

            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(writer_folder, filename)

            shutil.move(src_path, dest_path)


def unzip(path_of_zipfile,source_dir):
    zip_file_path = path_of_zipfile
    extract_to_path = source_dir

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def train_val_test_split(source_dir,IMAGE_SIZE=(384,384),BATCH_SIZE=18):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{source_dir}/train",
        labels='inferred',
        color_mode='rgb',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{source_dir}/val",
        labels='inferred',
        color_mode='rgb',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{source_dir}/test",
        labels='inferred',
        color_mode='rgb',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE)

    return train_ds,val_ds,test_ds
    
###DATA AUGMENTATION
data_augmentation_layers = [
    layers.RandomRotation(0.02),
    layers.RandomCrop(120,120),
    layers.RandomContrast((0.2,0.5)),
    ]

def data_augmentation(images):
    '''Function for testing augmentations with test_augmentations(), not meant for use
    
    For applying augmentations, use: apply_augmentations()'''
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def test_augmentation(train_ds):
    '''Visualizing augmentations with matplotlib'''
    plt.figure(figsize=(12, 12))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"),cmap="gray",interpolation="nearest")
            plt.axis("off")

def apply_augmentations():
    '''Apply `data_augmentation` to the training images.'''
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
        )
    return train_ds


#####################

def full_unzip_and_stratify(path_of_zipfile,source_dir,IMAGE_SIZE=(384,384),BATCH_SIZE=18):
    '''Starts with zipped file, divides unzipped images by author, splits sorted images into train/val/test images,
    creates & returns train/val/test datatsets'''

    unzip(path_of_zipfile,source_dir)
    organize_unzipped_files(source_dir)

    #w_ids = os.listdir(source_dir)
    #w_ids.sort()
    #w_ids.pop(0)

    stratify_split(source_dir)
    del_old_writer_directories(source_dir)

    train_ds,val_ds,test_ds = train_val_test_split(source_dir,IMAGE_SIZE,BATCH_SIZE)
    return train_ds,val_ds,test_ds
    
