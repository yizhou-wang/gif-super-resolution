from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf

from data import Dataset
from utils import *

FLAGS = tf.app.flags.FLAGS


class ModelTrainer(object):
    """
    Class to take care of setting up tensorflow configuration, model parameters, managing training, and producing samples
    """

    def __init__(self, model):
        '''
        Setup directories, dataset, model, and optimizer
        '''
        self.batch_size = FLAGS.batch_size
        self.iterations = FLAGS.iterations
        self.learning_rate = FLAGS.learning_rate

        self.model_dir = FLAGS.model_dir  # directory to write model summaries to
        self.dataset_dir = FLAGS.dataset_dir  # directory containing data
        self.samples_dir = FLAGS.samples_dir  # directory for sampled images
        self.device_id = FLAGS.device_id
        self.use_gpu = FLAGS.use_gpu

        # create directories if they don"t exist yert
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

        if self.use_gpu:
            device_str = '/gpu:' + str(self.device_id)
        else:
            device_str = '/cpu:0'
        with tf.device(device_str):
            self.global_step = tf.get_variable("global_step", [],
                                               initializer=tf.constant_initializer(0), trainable=False)

            # parse data and create model
            self.dataset = Dataset(self.dataset_dir, self.iterations, self.batch_size)
            self.model = model(self.dataset.hr_images, self.dataset.lr_images)
            learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                       500000, 0.5,  staircase=True)
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
            self.train_optimizer = optimizer.minimize(self.model.loss, global_step=self.global_step)

    def train(self):
        '''
        Initialize variables, setup summary writer, saver and Coordinator and start training
        '''
        init = tf.global_variables_initializer()
        summarize = tf.summary.merge_all()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            # write model summary
            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)
            # start input threads to enqueue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            iterations = 1
            try:
                while not coord.should_stop():
                    t1 = time.time()
                    _, loss = sess.run([self.train_optimizer, self.model.loss])
                    t2 = time.time()

                    print("Step {}, loss={:.2f}, ({:.1f} examples/sec; {:.3f} sec/batch)".format(iterations,
                                                                                                 loss, self.batch_size / (t2 - t1), (t2 - t1)))
                    # summarize model
                    if iterations % 10 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, iters)

                    # create sample
                    if iterations % 2 == 0:
                        print("Sampling from model")
                        self.sample_from_model(sess, mu=1.0, step=iterations)
                        summary_writer.add_summary
                        print("Done sampling model")

                    # save model
                    if iterations % 10000 == 0:
                        checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
                        saver.save(sess, checkpoint_path, global_step=iterations)

                    iterations += 1

            except tf.errors.OutOfRangeError:
                print("Done training")
            finally:
                coord.request_stop()
                coord.join(threads)

    def sample_from_model(self, sess, mu=1.1, step=None):
        conditioning_logits = self.model.conditioning_logits
        prior_logits = self.model.prior_logits

        hr_imgs = self.dataset.hr_images
        lr_imgs = self.dataset.lr_images
        # values are filled in
        fetched_hr_imgs, fetched_lr_imgs = sess.run([hr_imgs, lr_imgs])
        # new upscaled image
        generated_hr_imgs = np.zeros((self.batch_size, 32, 32, 3), dtype=np.float32)

        fetched_conditioning_logits = sess.run(conditioning_logits, feed_dict={lr_imgs: fetched_lr_imgs})

        for i in range(32):
            for j in range(32):
                for c in range(3):
                    fetched_prior_logits = sess.run(prior_logits, feed_dict={hr_imgs: generated_hr_imgs})
                    # get_value for new pixel
                    new_pixel = logits_to_pixel(
                        fetched_conditioning_logits[:, i, j, c * 256:(c + 1) * 256] + fetched_prior_logits[:, i, j, c * 256:(c + 1) * 256], mu=mu)
                    # add pixel to generated image
                    generated_hr_imgs[:, i, j, c] = new_pixel
                print("pixel", i, j)
        save_samples(fetched_lr_imgs, self.model_dir + '/lr_' + str(mu * 10) + '_' + str(step) + '.jpg')
        save_samples(fetched_hr_imgs, self.model_dir + '/hr_' + str(mu * 10) + '_' + str(step) + '.jpg')
        save_samples(generated_hr_imgs, self.model_dir + '/generate_' + str(mu * 10) + '_' + str(step) + '.jpg')
