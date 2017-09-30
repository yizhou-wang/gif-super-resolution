#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from model import PixelResNet
from trainer import ModelTrainer

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_float("learning_rate", 0.0004, "Learning Rate")
tf.app.flags.DEFINE_integer("B", 6, "Number of ResNet layers in conditional network")
tf.app.flags.DEFINE_integer("batch_size", 32, "Number of samples per batch")
tf.app.flags.DEFINE_integer("image_size", 8, "Size in pixels of image")
tf.app.flags.DEFINE_integer("iterations", 2000, "Number of training iterations")
tf.app.flags.DEFINE_string("dataset_dir", "dataset", "Path to dataset directory")
tf.app.flags.DEFINE_string("model_dir", "models", "Output folder where models are dumped.")
tf.app.flags.DEFINE_string("samples_dir", "samples", "Output folder where samples are dumped.")
tf.app.flags.DEFINE_boolean("use_gpu", True, "Use GPUs for training?")
tf.app.flags.DEFINE_integer("device_id", 0, "ID of GPU to use")
tf.app.flags.DEFINE_boolean("first_time", True, "first time running program")

def main(argv=None):
    trainer = ModelTrainer(PixelResNet)
    trainer.train()

if __name__ == "__main__":
    tf.app.run()
