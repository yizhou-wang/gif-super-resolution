import glob
import numpy as np
import scipy.ndimage
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

hr = 32
lr = 8
channel = 3

def load_lr_gif(lr_dir='../data/lr_imgs/', tag='face', number='999'):
    print 'lr_path =', lr_dir + tag + '/' + number + '/*.png'
    lr_list = glob.glob(lr_dir + tag + '/' + number + '/*.png')
    lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print lr_list
    data_lr_gif = np.zeros((len(lr_list), lr, lr, channel))

    for idx, lr_img in enumerate(lr_list):
        im = scipy.ndimage.imread(lr_img)
        data_lr_gif[idx, :, :] = im

    return data_lr_gif

def load_bi_gif(bi_dir='../data/bi_imgs/', tag='face', number='999'):
    print 'bi_path =', bi_dir + tag + '/' + number + '/*.png'
    bi_list = glob.glob(bi_dir + tag + '/' + number + '/*.png')
    bi_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print bi_list
    data_bi_gif = np.zeros((len(bi_list), hr, hr, channel))

    for idx, bi_img in enumerate(bi_list):
        im = scipy.ndimage.imread(bi_img)
        data_bi_gif[idx, :, :] = im

    return data_bi_gif

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def temp_filter(data_gif):
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz
    shape = data_gif.shape
    T = shape[0]
    data_tf_gif = data_gif
    for i in range(shape[1]):
        for j in range(shape[2]):
            for k in range(shape[3]):
                data_tf_gif[:, i, j, k] = butter_lowpass_filter(data_gif[:, i, j, k], cutoff, fs, order)
    return data_tf_gif

def GD(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


if __name__ == '__main__':
    '''
    Step 1: Read images.
        'data_lr_gif':  read lr GIF in a array (frame X 8 X 8 X 3)
    '''
    # data_lr_gif = load_lr_gif()
    data_lr_gif = load_lr_gif(lr_dir='../../data/lr_imgs/')
    print 'data_lr_gif =', data_lr_gif.shape

    '''
    Step 2: BI on each frame.
        'data_bi_gif':  bicubic interpolation on each frame (frame X 32 X 32 X 3)
        'alpha':    coefficients of interpolation function (frame X 32 X 32)
                    (each alpha: 4 X 4, 8 X 8 alphas for each frame)
    '''
    # data_bi_gif = load_bi_gif()
    data_bi_gif = load_bi_gif(bi_dir='../../data/bi_imgs/')
    print 'data_bi_gif =', data_bi_gif.shape

    data_tf_gif = temp_filter(data_bi_gif)

    '''
    Step 3: Optimization.
        - Compute cost = BI cost + TR cost
        - Gradient descent
        - Next iteration
    '''



