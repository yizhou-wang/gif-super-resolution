import tensorflow as tf
import os
import sys
sys.path.insert(0, './')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import *

flags = tf.app.flags


#solver
flags.DEFINE_string("train_dir", "../result/prsr/models", "trained model save path")
flags.DEFINE_string("samples_dir", "../result/prsr/samples", "sampled images save path")
flags.DEFINE_string("train_imgs_path", "../data/train.txt", "images list file path")
flags.DEFINE_string("test_imgs_path", "../data/train.txt", "images list file path")

flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

flags.DEFINE_integer("num_epoch", 8, "train epoch num")
flags.DEFINE_integer("batch_size", 32, "batch_size")

# print("size of batch:",flags.FLAGS.batch_size)

# print("number of images for training: ", s)
# flags.DEFINE_integer("dataset_size", s, "size of dataset")

flags.DEFINE_integer("size_hr", 32, "size of high resolution images")
flags.DEFINE_integer("size_lr", 8, "size of low resolution image")

flags.DEFINE_float("learning_rate", 4e-4, "learning rate")

conf = flags.FLAGS


def main(_):
	solver = Solver()
	init_op = tf.initialize_all_variables()
	# sess = tf.Session()
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
	sess.run(init_op)  	
	solver.test()

if __name__ == '__main__':

  	tf.app.run()
