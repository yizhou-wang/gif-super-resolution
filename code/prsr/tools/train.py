import tensorflow as tf
import os
import sys
sys.path.insert(0, './')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import *

flags = tf.app.flags

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if not os.path.exists('../result'):
	os.mkdir('../result')
if not os.path.exists('../result/prsr'):
	os.mkdir('../result/prsr')
if not os.path.exists('../result/prsr/models'):
	os.mkdir('../result/prsr/models')
if not os.path.exists('../result/prsr/samples'):
	os.mkdir('../result/prsr/samples')

#solver
flags.DEFINE_string("train_dir", "../result/prsr/models", "trained model save path")
flags.DEFINE_string("samples_dir", "../result/prsr/samples", "sampled images save path")
flags.DEFINE_string("imgs_list_path", "../data/train.txt", "images list file path")

flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")
flags.DEFINE_integer("device_id", 0, "gpu device id")

flags.DEFINE_integer("num_epoch", 16, "train epoch num")
flags.DEFINE_integer("batch_size", 32, "batch_size")

print("size of batch:",flags.FLAGS.batch_size)

data_file = "../data/train.txt"
s = file_len(data_file)
flags.DEFINE_integer("file_length", s, "file_length")
print("number of images for training: ", s)
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
	solver.train()

if __name__ == '__main__':

  	tf.app.run()
