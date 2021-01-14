"""
INCLUDE ONLY, DO NOT EXECUTE

MY RES-U-NET MODEL, inspired by U-Net model proposed in the paper:
Ronneberger et al. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/pdf/1505.04597.pdf
"""
from data import *
import tensorflow as tf


# Use ResNet preprocessing
preprocessing = tf.keras.applications.resnet.preprocess_input


# Residual block
def conv_block(filters, x):
    # 1x1 convolution layer that is added to equalize number of channels in the feature map for future addition
    t = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding='same')(x)

    # The 1st convolutional block
    y = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(t)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    # The 2nd convolutional block
    y = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)

    # Residual connection and ReLU activation
    y = tf.keras.layers.add([t, y])
    y = tf.keras.layers.Activation('relu')(y)
    return y


# Create model
def create_model():
    input = tf.keras.layers.Input((image_height, image_width, 3))

    conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input)
    pool0 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv0)

    conv1 = conv_block(64, pool0)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(128, pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(256, pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(512, pool3)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(1024, pool4)

    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)
    merge6 = tf.keras.layers.concatenate([conv4, up6])
    conv6 = conv_block(512, merge6)

    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7])
    conv7 = conv_block(256, merge7)

    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8])
    conv8 = conv_block(128, merge8)

    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9])
    conv9 = conv_block(64, merge9)

    up10 = tf.keras.layers.UpSampling2D(size=(4, 4))(conv9)
    output = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(7, 7), activation='softmax', padding='same')(up10)

    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model
