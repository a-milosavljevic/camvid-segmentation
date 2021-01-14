"""
INCLUDE ONLY, DO NOT EXECUTE

Be sure to download dataset from https://www.kaggle.com/carlolepelaars/camvid/download
and unpack it to "data" subfolder
"""
from settings import *
import csv
import numpy as np
import cv2 as cv
import tensorflow as tf
from random import *


"""
Read class_dict.csv and create structures for translating colors to class index and vice versa
"""
classes = []
colors = []
shades = []
color2index = dict()
with open(os.path.join(data_folder, 'class_dict.csv'), 'rt', newline='') as f:
    reader = csv.reader(f)
    first_row = True
    for row in reader:
        if not first_row:
            idx = len(classes)
            color = (int(row[3]), int(row[2]), int(row[1]))
            classes.append(row[0])
            colors.append(color)
            color2index[color] = idx
        first_row = False
num_classes = len(classes)
#print(classes)
#print(colors)
#print(num_classes)
#print(color2index)

"""
Finds grayscale shades that correspond to color palette (must not have duplicates!)
"""
tmp = np.zeros((num_classes, 1, 3), dtype=np.uint8)
for i in range(num_classes):
    tmp[i, 0] = colors[i]
tmp = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
for i in range(num_classes):
    shades.append(tmp[i, 0])

# Create lookup table for converting grayscale shades to indexes
shade2index = np.full((256,), 255, dtype=np.uint8)
for i in range(num_classes):
    shade2index[shades[i]] = i

# Create lookup table for converting indexes to BGR values
index2color = np.full((256, 1, 3), (255, 255, 255), dtype=np.uint8)
for i in range(num_classes):
    index2color[i, 0] = np.array(colors[i], dtype=np.uint8)
#print(index2color)


def bgr2cat(img):
    """
    Convert BGR to indexed image
    """
    if False:
        # TOO SLOW
        return np.apply_along_axis(lambda bgr: color2index[tuple(bgr)], 2, img)
    else:
        # Only applicable when grayscale values do not overlap
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return cv.LUT(gray, shade2index)


def cat2bgr(img):
    """
    Convert index to BGR image
    """
    bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return cv.LUT(bgr, index2color)


class DataProvider(tf.keras.utils.Sequence):
    """
    Custom data provider class that loads and augments images
    """
    def __init__(self, batch_size, is_validation, process_input):
        self.batch_size = batch_size
        self.is_validation = is_validation
        self.process_input = process_input

        if self.is_validation:
            self.images = np.random.RandomState(0).permutation(val_images)
            print("validation_elements = " + str(len(self.images)))
        else:
            self.images = np.random.permutation(train_images)
            print("training_elements = " + str(len(self.images)))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min(len(self.images), (idx + 1) * self.batch_size)
        batch_images = self.images[batch_start:batch_end]

        batch_x = np.zeros((len(batch_images), image_height, image_width, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_images), image_height, image_width, num_classes), dtype=np.float32)

        for i in range(len(batch_images)):
            if self.is_validation:
                image_path = os.path.join(val_folder, batch_images[i] + '.png')
                label_path = os.path.join(val_labels_folder, batch_images[i] + '_L.png')
            else:
                image_path = os.path.join(train_folder, batch_images[i] + '.png')
                label_path = os.path.join(train_labels_folder, batch_images[i] + '_L.png')

            # Load and resize images
            image = cv.imread(image_path)
            label = cv.imread(label_path)
            image = cv.resize(image, (image_width, image_height), interpolation=cv.INTER_LINEAR)
            label = cv.resize(label, (image_width, image_height), interpolation=cv.INTER_NEAREST)

            # Data augmentation (LR flip)
            if not self.is_validation and random() > 0.5:
                image = cv.flip(image, 1)
                label = cv.flip(label, 1)

            label_cat = bgr2cat(label)
            label_cat = tf.keras.utils.to_categorical(label_cat, num_classes=num_classes, dtype='float32')

            batch_x[i] = self.process_input(image)
            batch_y[i] = label_cat

        return batch_x, batch_y
