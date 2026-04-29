
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

w_ids90 = ['w0001',  'w0002',  'w0003',  'w0004',  'w0005',  'w0006',  'w0009',  'w0010',  'w0011',  'w0012',  'w0013',  'w0015',  'w0016',  'w0017',  'w0018',  'w0020',  'w0022',  'w0023',  'w0024',  'w0025',  'w0026',  'w0027',  'w0028',  'w0029',  'w0030',  'w0031',  'w0032',  'w0033',  'w0034',  'w0035',  'w0036',  'w0038',  'w0043',  'w0061',  'w0062',  'w0063',  'w0064',  'w0066',  'w0069',  'w0070',  'w0071',  'w0073',  'w0074',  'w0075',  'w0076',  'w0077',  'w0078',  'w0080',  'w0082',  'w0083',  'w0085',  'w0086',  'w0087',  'w0088',  'w0089',  'w0091',  'w0092',  'w0093',  'w0094',  'w0095',  'w0121',  'w0122',  'w0123',  'w0124',  'w0125',  'w0126',  'w0128',  'w0129',  'w0130',  'w0131',  'w0133',  'w0134',  'w0135',  'w0136',  'w0137',  'w0138',  'w0139',  'w0142',  'w0143',  'w0144',  'w0145',  'w0147',  'w0148',  'w0149',  'w0151',  'w0152',  'w0153',  'w0154',  'w0155',  'w0156']
w_ids475 = ['w0001', 'w0002', 'w0003', 'w0004', 'w0005', 'w0006', 'w0009', 'w0010', 'w0011', 'w0012', 'w0013', 'w0015', 'w0016', 'w0017', 'w0018', 'w0020', 'w0022', 'w0023', 'w0024', 'w0025', 'w0026', 'w0027', 'w0028', 'w0029', 'w0030', 'w0031', 'w0032', 'w0033', 'w0034', 'w0035', 'w0036', 'w0038', 'w0040', 'w0041', 'w0042', 'w0043', 'w0049', 'w0058', 'w0061', 'w0062', 'w0063', 'w0064', 'w0066', 'w0069', 'w0070', 'w0071', 'w0073', 'w0074', 'w0075', 'w0076', 'w0077', 'w0078', 'w0080', 'w0082', 'w0083', 'w0085', 'w0086', 'w0087', 'w0088', 'w0089', 'w0090', 'w0091', 'w0092', 'w0093', 'w0094', 'w0095', 'w0099', 'w0102', 'w0118', 'w0119', 'w0121', 'w0122', 'w0123', 'w0124', 'w0125', 'w0126', 'w0128', 'w0129', 'w0130', 'w0131', 'w0132', 'w0133', 'w0134', 'w0135', 'w0136', 'w0137', 'w0138', 'w0139', 'w0142', 'w0143', 'w0144', 'w0145', 'w0146', 'w0147', 'w0148', 'w0149', 'w0150', 'w0151', 'w0152', 'w0153', 'w0154', 'w0155', 'w0156', 'w0157', 'w0160', 'w0162', 'w0175', 'w0177', 'w0180', 'w0182', 'w0184', 'w0186', 'w0189', 'w0191', 'w0193', 'w0198', 'w0199', 'w0200', 'w0201', 'w0202', 'w0203', 'w0204', 'w0205', 'w0206', 'w0212', 'w0218', 'w0220', 'w0223', 'w0224', 'w0226', 'w0227', 'w0229', 'w0232', 'w0233', 'w0234', 'w0238', 'w0239', 'w0240', 'w0242', 'w0244', 'w0245', 'w0246', 'w0249', 'w0254', 'w0255', 'w0260', 'w0261', 'w0262', 'w0263', 'w0264', 'w0270', 'w0271', 'w0276', 'w0277', 'w0279', 'w0280', 'w0281', 'w0282', 'w0284', 'w0285', 'w0286', 'w0287', 'w0288', 'w0291', 'w0293', 'w0297', 'w0299', 'w0301', 'w0302', 'w0304', 'w0305', 'w0306', 'w0308', 'w0312', 'w0313', 'w0314', 'w0315', 'w0317', 'w0319', 'w0320', 'w0322', 'w0330', 'w0333', 'w0334', 'w0335', 'w0337', 'w0338', 'w0339', 'w0340', 'w0341', 'w0342', 'w0344', 'w0348', 'w0350', 'w0351', 'w0352', 'w0353', 'w0354', 'w0355', 'w0356', 'w0357', 'w0359', 'w0362', 'w0364', 'w0365', 'w0367', 'w0368', 'w0370', 'w0371', 'w0372', 'w0375', 'w0379', 'w0380', 'w0381', 'w0382', 'w0383', 'w0384', 'w0387', 'w0388', 'w0391', 'w0392', 'w0393', 'w0396', 'w0397', 'w0398', 'w0399', 'w0400', 'w0401', 'w0402', 'w0403', 'w0405', 'w0406', 'w0407', 'w0408', 'w0409', 'w0410', 'w0411', 'w0412', 'w0413', 'w0414', 'w0415', 'w0416', 'w0417', 'w0419', 'w0420', 'w0422', 'w0424', 'w0425', 'w0426', 'w0428', 'w0429', 'w0431', 'w0433', 'w0435', 'w0436', 'w0439', 'w0440', 'w0441', 'w0443', 'w0444', 'w0445', 'w0446', 'w0448', 'w0450', 'w0451', 'w0452', 'w0454', 'w0456', 'w0458', 'w0460', 'w0462', 'w0463', 'w0464', 'w0465', 'w0466', 'w0467', 'w0468', 'w0469', 'w0470', 'w0471', 'w0472', 'w0473', 'w0474', 'w0475', 'w0476', 'w0477', 'w0479', 'w0480', 'w0481', 'w0483', 'w0484', 'w0485', 'w0486', 'w0487', 'w0489', 'w0492', 'w0493', 'w0495', 'w0497', 'w0498', 'w0500', 'w0501', 'w0502', 'w0508', 'w0510', 'w0513', 'w0514', 'w0515', 'w0516', 'w0517', 'w0518', 'w0519', 'w0520', 'w0521', 'w0522', 'w0523', 'w0524', 'w0525', 'w0526', 'w0527', 'w0528', 'w0529', 'w0530', 'w0531', 'w0532', 'w0533', 'w0534', 'w0535', 'w0536', 'w0537', 'w0538', 'w0541', 'w0542', 'w0543', 'w0546', 'w0547', 'w0548', 'w0549', 'w0550', 'w0551', 'w0552', 'w0553', 'w0554', 'w0555', 'w0557', 'w0559', 'w0560', 'w0561', 'w0562', 'w0564', 'w0565', 'w0566', 'w0569', 'w0570', 'w0571', 'w0572', 'w0573', 'w0575', 'w0576', 'w0577', 'w0579', 'w0580', 'w0581', 'w0586', 'w0587', 'w0588', 'w0589', 'w0590', 'w0591', 'w0592', 'w0593', 'w0594', 'w0595', 'w0596', 'w0597', 'w0598', 'w0599', 'w0600', 'w0601', 'w0602', 'w0604', 'w0605', 'w0606', 'w0611', 'w0612', 'w0613', 'w0615', 'w0617', 'w0618', 'w0619', 'w0620', 'w0621', 'w0622', 'w0623', 'w0624', 'w0626', 'w0627', 'w0628', 'w0629', 'w0630', 'w0632', 'w0634', 'w0636', 'w0637', 'w0638', 'w0639', 'w0640', 'w0641', 'w0642', 'w0644', 'w0645', 'w0646', 'w0647', 'w0648', 'w0650', 'w0653', 'w0656', 'w0657', 'w0658', 'w0660', 'w0661', 'w0662', 'w0664', 'w0665', 'w0666', 'w0667', 'w0668', 'w0669', 'w0671', 'w0673', 'w0674', 'w0675', 'w0677', 'w0678', 'w0679', 'w0680', 'w0682', 'w0683', 'w0685', 'w0688', 'w0691', 'w0692', 'w0693', 'w0694', 'w0695', 'w0698', 'w0699', 'w0700', 'w0701', 'w0702', 'w0703', 'w0704', 'w0705', 'w0706', 'w0707', 'w0709', 'w0710', 'w0711', 'w0712', 'w0713', 'w0714', 'w0715', 'w0717', 'w0719', 'w0720', 'w0418', 'w0453', 'w0394', 'w0345', 'w0478', 'w0670', 'w0289', 'w0459', 'w0216', 'w0378']



def stratify_split(source_dir,w_ids,split=[15,6,6]):
    '''Re-organizes images in writer folders into train/val/test folders with writer sub-directories'''
    if w_ids == 90:
        print("Using 90 class labels")
        w_ids = w_ids90
    elif w_ids == 475:
        print("Using 475 class labels")
        w_ids = w_ids475

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


def stratify_split_alt(source_dir, w_ids=90,split=(15, 6, 6)):
    """Re-organizes images in writer folders into train/val/test folders with writer sub-directories."""
    train_folder = os.path.join(source_dir, "train")
    val_folder = os.path.join(source_dir, "val")
    test_folder = os.path.join(source_dir, "test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    if w_ids == 90:
        print("Using 90 class labels")
        w_ids = w_ids90
    elif w_ids == 475:
        print("Using 475 class labels")
        w_ids = w_ids475

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


def del_old_writer_directories(source_dir,w_ids=90):
    '''Deletes original writer-organized directories (for use after creating train/val/test directories with stratify_split())'''
    if w_ids == 90:
        #print("Using 90 class labels")
        w_ids = w_ids90
    elif w_ids == 475:
        #print("Using 475 class labels")
        w_ids = w_ids475

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


def train_val_test_split(source_dir,IMAGE_SIZE=(384,384),BATCH_SIZE=64):
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


def show_image(dataset):
    "Shows 3x3 grid of images from given TensorFlow dataset (e.g., test_ds)"
    plt.figure(figsize=(12, 12))
    for images, _ in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"),cmap="gray",interpolation="nearest")
            plt.axis("off")


def apply_augmentations():
    '''Apply `data_augmentation` to the training images.'''
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
        )
    return train_ds


def make_transfer_model(base_model, input_shape, num_classes, name):
    backbone = base_model

    backbone.trainable = False

    inputs = layers.Input(input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.25)(x) #changed to 30% dropout
    outputs = layers.Dense(num_classes, activation=None)(x)

    return keras.Model(inputs, outputs, name=name)

def show_testpreds(test_ds,model,classes=475):
    """Shows 3x3 grid of images with True vs Pred labels"""
    for images, labels in test_ds.take(1):
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            if classes == 475:
                w_ids = w_ids475
            elif classes == 90:
                w_ids = w_ids90
                
            true_label = w_ids[labels[i]]
            pred_label = w_ids[pred_labels[i]]
            if true_label == pred_label:
                c = "green"
            else:
                c="red"
            plt.title(f"T: {true_label}\nP: {pred_label}",color=c)
            plt.axis("off")
        plt.show()


#####################

def full_unzip_and_stratify(path_of_zipfile,source_dir,w_ids=90,IMAGE_SIZE=(384,384),BATCH_SIZE=64):
    '''Starts with zipped file, divides unzipped images by author, splits sorted images into train/val/test images,
    creates & returns train/val/test datatsets'''

    unzip(path_of_zipfile,source_dir)
    organize_unzipped_files(source_dir)

    #w_ids = os.listdir(source_dir)
    #w_ids.sort()
    #w_ids.pop(0)

    stratify_split(source_dir,w_ids)
    del_old_writer_directories(source_dir,w_ids)

    train_ds,val_ds,test_ds = train_val_test_split(source_dir,IMAGE_SIZE,BATCH_SIZE)
    return train_ds,val_ds,test_ds


######################

def process_image(args):
    img_path, output_path, target_size, bw_threshold = args

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        if h > w:
            # Tall image → crop from bottom
            img = img.crop((0, 0, w, w))   # keep top square
            squared_img = img

        else:
            # Wide image → pad normally
            max_dim = max(w, h)

            squared_img = Image.new("RGB", (max_dim, max_dim), (255,255,255))
            squared_img.paste(img, (0,0))

        if target_size:
            squared_img = squared_img.resize(target_size, Image.LANCZOS)

        # Convert to pure black and white (no greyscale) via thresholding
        greyscale = squared_img.convert("L")
        bw_img = greyscale.point(lambda p: 255 if p >= bw_threshold else 0, "1")

        bw_img.save(output_path)


def process_dataset(input_root, output_root, target_size=(384,384), workers=8, bw_threshold=235):
    """Applies squaring & BW coloring to images in train/val/test folders"""
    tasks = []

    for split in ["train", "val", "test"]:
        split_path = os.path.join(input_root, split)

        for class_name in os.listdir(split_path):

            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            output_class_path = os.path.join(output_root, split, class_name)
            os.makedirs(output_class_path, exist_ok=True)

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png','.jpg','.jpeg')):

                    img_path = os.path.join(class_path, img_name)
                    output_path = os.path.join(output_class_path, img_name)

                    tasks.append((img_path, output_path, target_size, bw_threshold))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks)))
