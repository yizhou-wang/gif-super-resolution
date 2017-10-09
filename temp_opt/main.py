import glob
import numpy as np
import numpy.linalg
import scipy.ndimage

from utils import *

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
        data_lr_gif[idx, :, :, :] = im

    return data_lr_gif

def load_fl_frame(hr_dir='../data/hr_imgs/', tag='face', number='999'):
    print 'hr_path =', hr_dir + tag + '/' + number + '/*.png'
    hr_list = glob.glob(hr_dir + tag + '/' + number + '/*.png')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print lr_list
    data_fl_frame = np.zeros((2, hr, hr, channel))
    # Load first frame
    f_frame = scipy.ndimage.imread(hr_list[0])
    data_fl_frame[0, :, :, :] = f_frame
    # Load last frame
    l_frame = scipy.ndimage.imread(hr_list[-1])
    data_fl_frame[1, :, :, :] = l_frame
    return data_fl_frame

def load_bi_gif(bi_dir='../data/bi_imgs/', tag='face', number='999'):
    print 'bi_path =', bi_dir + tag + '/' + number + '/*.png'
    bi_list = glob.glob(bi_dir + tag + '/' + number + '/*.png')
    bi_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print bi_list
    data_bi_gif = np.zeros((len(bi_list), hr, hr, channel))

    for idx, bi_img in enumerate(bi_list):
        im = scipy.ndimage.imread(bi_img)
        data_bi_gif[idx, :, :, :] = im

    return data_bi_gif

def get_loss_gradiant(frame_num, params, data_fl_frame, data_bi_gif):
    n = frame_num - 1
    rho = params[0]
    gamma = params[1]
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += rho**(n-i) * gamma * data_bi_gif[i, :, :, :]
    Fn = rho**n + sum0
    loss = numpy.linalg.norm(Fn - Fn_gt)
    # Compute Fn_rho = partial_Fn / partial_rho
    sum1 = 0
    for i in range(1, n-1):
        sum1 += (n-i) * rho**(n-i-1) * gamma * data_bi_gif[i, :, :, :]
    Fn_rho = n * rho**(n-1) * F0_gt + sum1
    # Compute Fn_gamma = partial_Fn / partial_gamma
    Fn_gamma = 0
    for i in range(1, n):
        Fn_gamma += rho**(n-i) * data_bi_gif[i, :, :, :]
    # Compute partial_rho = partial_l / partial_rho
    partial_rho = 2 * Fn * Fn_rho - 2 * Fn_gt * Fn_rho
    # Compute partial_gamma = partial_l / partial_gamma
    partial_gamma = 2 * Fn * Fn_gamma - 2 * Fn_gt * Fn_gamma
    grad = np.array([partial_rho, partial_gamma])
    return loss, grad

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
    frame_num = data_lr_gif.shape[0]
    print 'data_lr_gif =', data_lr_gif.shape
    data_fl_frame = load_fl_frame(hr_dir='../../data/hr_imgs/')
    print 'data_fl_frame =', data_fl_frame.shape

    '''
    Step 2: BI on each frame.
        'data_bi_gif':  bicubic interpolation on each frame (frame X 32 X 32 X 3)
        'alpha':    coefficients of interpolation function (frame X 32 X 32)
                    (each alpha: 4 X 4, 8 X 8 alphas for each frame)
    '''
    # data_bi_gif = load_bi_gif()
    data_bi_gif = load_bi_gif(bi_dir='../../data/bi_imgs/')
    print 'data_bi_gif =', data_bi_gif.shape
    # optical_flow(data_bi_gif)
    # data_tf_gif = temp_filter(data_bi_gif)

    '''
    Step 3: Optimization.
        - Compute cost = BI cost + TR cost
        - Gradient descent
        - Next iteration
    '''
    params = np.array([0.5, 0.5])
    # print params
    loss, grad_l = get_loss_gradiant(frame_num, params, data_fl_frame, data_bi_gif)
    print loss
    print grad_l
    




