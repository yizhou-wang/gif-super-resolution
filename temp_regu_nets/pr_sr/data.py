from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Dataset(object):
    """
    Class to create a image reader queue to batch dataset
    """

    def __init__(self, data_path, iterations, batch_size):

        if FLAGS.first_time:
            with open('fnames.txt', 'w') as f:
                records = lambda x: os.path.abspath(data_path + '/' + x)
                self.records = list(map(records, os.listdir(data_path)))
                for record in self.records:
                    f.write(record + '\n')

        else:
            with open('fnames.txt', 'r') as f:
                self.records = []
                for line in f:
                    self.records.append(line.strip())

        filename_queue = tf.train.string_input_producer(self.records)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file, 3)
        hr_image = tf.image.resize_images(image, [32, 32])  # downsample image
        lr_image = tf.image.resize_images(image, [8, 8])  # REALLY downsample image
        hr_image = tf.cast(hr_image, tf.float32)
        lr_image = tf.cast(lr_image, tf.float32)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size

        # batches images of shape [batch_size, 32, 32, 3],[batch_size, 8, 8, 3]
        self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size,
                                                                min_after_dequeue=min_after_dequeue, capacity=capacity)
