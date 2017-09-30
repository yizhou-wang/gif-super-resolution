from __future__ import absolute_import, division, print_function

import glob
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
# from skimage.io import imsave

FLAGS = tf.app.flags.FLAGS


def split_and_gate(tensor, channels):
    """
    Split tensor into two channels of size channels and put the tensors through a PixelCNN gate
    """
    t1 = tensor[:, :, :, :channels]
    t2 = tensor[:, :, :, channels:]
    t1 = tf.nn.tanh(t1)
    t2 = tf.nn.sigmoid(t2)
    return t1 * t2


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Not using tf.nn.softmax because it throws an axis out of bounds error
    """
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


def logits_to_pixel(logits, mu=1.0):
    """
    Helper function to convert output logits from network into pixels
    """
    rebalance_logits = logits * mu
    probs = softmax(rebalance_logits)
    pixel_dict = np.arange(0, 256, dtype=np.float32)
    pixels = np.sum(probs * pixel_dict, axis=1)
    return np.floor(pixels)


def normalize_color(image):
    """
    Helper to rescale pixel color intensity to [-1, 1]
    """
    return image / 255.0 - 0.5


def save_samples(fetched_images, image_path):
    """
    Save image sampled from network
    """
    fetched_images = fetched_images.astype(np.uint8)
    n, h, w, _ = fetched_images.shape
    num = int(n ** 0.5)
    merged_image = np.zeros((n * h, n * w, 3), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            merged_image[i * h:(i + 1) * h, j * w: (j + 1) * w, :] = fetched_images[i * num + j, :, :, :]
    tf.summary.image(image_path, merged_image)
    # imsave(image_path, merged_image)
