from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DataSet(object):
    def __init__(self, data_id, images_list_path, num_epoch, batch_size):

        input_file = open(images_list_path, 'r')
        self.record_list = []
        for line in input_file:
            line = line.strip()
            self.record_list.append(line)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 400 * batch_size

        if data_id == 'train':
            # filling the record_list
            filename_queue = tf.train.string_input_producer(self.record_list, num_epochs=num_epoch)
            image_reader = tf.WholeFileReader()
            _, image_file = image_reader.read(filename_queue)
            image = tf.image.decode_jpeg(image_file, channels=3)

            #preprocess
            hr_image = tf.image.resize_images(image, [32, 32])
            hr_image = tf.cast(hr_image, tf.float32)
            lr_image = tf.image.resize_images(image, [8, 8])
            lr_image = tf.cast(lr_image, tf.float32)
            # self.hr_images, self.lr_images = tf.train.shuffle_batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
            self.hr_images, self.lr_images = tf.train.batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity)

        if data_id == 'test':

            filename_queue = tf.train.string_input_producer(self.record_list, num_epochs=num_epoch)
            image_reader = tf.WholeFileReader()
            _, image_file = image_reader.read(filename_queue)
            image = tf.image.decode_jpeg(image_file, channels=3)

            hr_image = tf.image.resize_images(image, [32, 32])
            hr_image = tf.cast(hr_image, tf.float32)
            lr_image = tf.image.resize_images(image, [8, 8])
            lr_image = tf.cast(lr_image, tf.float32)
            # self.hr_images = hr_image
            # self.lr_images = lr_image
            # self.hr_images, self.lr_images = tf.train.batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity)
            self.hr_images, self.lr_images = tf.train.batch([hr_image, lr_image], batch_size=batch_size, capacity=capacity)
            
