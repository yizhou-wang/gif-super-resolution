import glob
import numpy as np
import numpy.linalg
import scipy.ndimage, scipy.misc
import os, sys

from utils import *

hr = 32
lr = 8
channel = 3
# frame_num = 0
# TEST_COUNTER = 0

def wait():
    raw_input("Press Enter to continue...")

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

def load_hr_gif(hr_dir='../data/hr_imgs/', tag='face', number='999'):
    print 'hr_path =', hr_dir + tag + '/' + number + '/*.png'
    hr_list = glob.glob(hr_dir + tag + '/' + number + '/*.png')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print hr_list
    data_hr_gif = np.zeros((len(hr_list), hr, hr, channel))

    for idx, hr_img in enumerate(hr_list):
        im = scipy.ndimage.imread(hr_img)
        data_hr_gif[idx, :, :, :] = im

    return data_hr_gif

def load_fl_frame(hr_dir='../data/hr_imgs/', tag='face', number='999'):
    print 'hr_path =', hr_dir + tag + '/' + number + '/*.png'
    hr_list = glob.glob(hr_dir + tag + '/' + number + '/*.png')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print hr_list
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

def gif_norm(gif, mutli_frame=True):
    size = gif.shape
    res = 0
    if mutli_frame == True:
        frame_num = size[0]
        for f in range(frame_num):
            for c in range(channel):
                # print numpy.linalg.norm(gif[f, :, :, c])
                res += numpy.linalg.norm(gif[f, :, :, c])
    else:
        for c in range(channel):
            # print gif[:, :, c]
            # print numpy.linalg.norm(gif[:, :, c])
            res += numpy.linalg.norm(gif[:, :, c])
    return res

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
    Fn = rho**n * F0_gt + sum0
    # global TEST_COUNTER
    # if TEST_COUNTER % 10 == 0:
    #     save_frame(Fn, dir='../../data/test/', f=TEST_COUNTER)
    # TEST_COUNTER += 1
    loss = gif_norm(Fn - Fn_gt, False)
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
    # grad = np.array([np.mean(partial_rho), np.mean(partial_gamma)])
    return loss, grad

def GD(data_bi_gif, data_fl_frame, params, step_size, numIterations, data_hr_gif):
    frame_num = data_bi_gif.shape[0]
    for i in range(0, numIterations):
        loss, grad_l = get_loss_gradiant(frame_num, params, data_fl_frame, data_bi_gif)
        bi_loss = gif_norm(data_bi_gif[-1, :, :, :] - data_fl_frame[1, :, :, :], False)
        data_rc_gif = recover_gif(data_bi_gif, data_fl_frame, params, False, True)
        # for f in range(data_rc_gif.shape[0]):
        #     print gif_norm(data_rc_gif[f] - data_hr_gif[f], False)
        total_bi_loss = gif_norm(data_bi_gif - data_hr_gif, True)
        total_loss = gif_norm(data_rc_gif - data_hr_gif, True)
        print("Step %d : BIloss: %f | Loss: %f | Total_BIloss: %f | Total_loss: %f" % (i, bi_loss, loss, total_bi_loss, total_loss))
        # Update
        params = params - step_size * grad_l
        # print params
    return params

def recover_gif(data_bi_gif, data_fl_frame, params, keep_fl=True, use_iter=True):
    frame_num = data_bi_gif.shape[0]
    data_rc_gif = np.zeros_like(data_bi_gif)
    if keep_fl == True:
        data_rc_gif[0] = data_fl_frame[0]
        data_rc_gif[-1] = data_fl_frame[1]
        if use_iter == True:
            for i in range(1, frame_num-1):
                data_rc_gif[i] = params[0] * data_rc_gif[i-1] + params[1] * data_bi_gif[i]
        else:
            sum2 = np.zeros((frame_num-1, hr, hr, channel))
            sum2[0] += params[0]**(frame_num-1) * params[1] * data_bi_gif[1]
            for i in range(1, frame_num-1):
                sum2[i-1] = sum2[i-2] + params[0]**(frame_num-i) * params[1] * data_bi_gif[i] 
            for i in range(1, frame_num-1):
                data_rc_gif[i] = params[0]**i * data_rc_gif[0] + sum2[i-1]
    else:
        data_rc_gif[0] = data_fl_frame[0]
        if use_iter == True:
            for i in range(1, frame_num):
                data_rc_gif[i] = params[0] * data_rc_gif[i-1] + params[1] * data_bi_gif[i]        
        else:
            sum2 = np.zeros((frame_num, hr, hr, channel))
            sum2[0] += params[0]**(frame_num-1) * params[1] * data_bi_gif[1]
            for i in range(1, frame_num):
                sum2[i-1] = sum2[i-2] + params[0]**(frame_num-i) * params[1] * data_bi_gif[i] 
            for i in range(1, frame_num):
                data_rc_gif[i] = params[0]**i * data_rc_gif[0] + sum2[i-1]
    # print data_rc_gif[i]
    return data_rc_gif

def toimg(gif):
    gif = np.where(gif <= 255, gif, 255)
    return gif

def save_frame(img, dir='../data/test/', f=0):
    save_dir = dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    name = save_dir + str(f) + '.png'
    scipy.misc.imsave(name, img)
    # wait()

def save_frames(gif, dir='../data/rc_imgs/', tag='face', number='999'):
    save_dir = dir + tag + '/' + number + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    size = gif.shape
    for f in range(size[0]):
        name = save_dir + str(f) + '.png'
        scipy.misc.imsave(name, gif[f])


if __name__ == '__main__':

    number = '10'

    '''
    Step 1: Read images.
        'data_lr_gif':      read lr GIF in a array (frame X 8 X 8 X 3)
        'data_hr_gif':      read hr GIF (GT) in a array (frame X 32 X 32 X 3)
        'data_fl_frame':    read first and last frame (GT) in a array (2 X 32 X 32 X 3)
    '''
    # data_lr_gif = load_lr_gif()
    data_lr_gif = load_lr_gif(lr_dir='../../data/lr_imgs/', number=number)
    print 'data_lr_gif =', data_lr_gif.shape
    data_hr_gif = load_hr_gif(hr_dir='../../data/hr_imgs/', number=number)
    print 'data_hr_gif =', data_hr_gif.shape
    data_fl_frame = load_fl_frame(hr_dir='../../data/hr_imgs/', number=number)
    print 'data_fl_frame =', data_fl_frame.shape
    # print data_fl_frame

    '''
    Step 2: BI on each frame.
        'data_bi_gif':  bicubic interpolation on each frame (frame X 32 X 32 X 3)
        'bi_loss':      loss of the bicubic interpolation
    '''
    # data_bi_gif = load_bi_gif()
    data_bi_gif = load_bi_gif(bi_dir='../../data/bi_imgs/', number=number)
    print 'data_bi_gif =', data_bi_gif.shape
    # optical_flow(data_bi_gif)
    # data_tf_gif = temp_filter(data_bi_gif)
    # bi_loss = gif_norm(data_bi_gif[-1, :, :, :] - data_fl_frame[1, :, :, :], False)

    '''
    Step 3: Optimization.
        - Compute cost = BI cost + TR cost
        - Gradient descent
        - Next iteration
    '''
    # scaler_params = np.array([0.5, 0.5])
    # scaler_params_res = GD(data_bi_gif, data_fl_frame, scaler_params, 0.001, 100)
    mat_params = np.array([np.tile(0.5, (hr, hr, channel)), np.tile(0.5, (hr, hr, channel))])
    mat_params_res = GD(data_bi_gif, data_fl_frame, mat_params, 1e-7, 1000, data_hr_gif)

    '''
    Step 4: Recover GIF.
        'data_rc_gif':      recovered GIF (frame X 32 X 32 X 3)
        'data_rc_gif_out':  map to [0, 255] (frame X 32 X 32 X 3)
    '''
    data_rc_gif = recover_gif(data_bi_gif, data_fl_frame, mat_params_res, False, True)
    # print data_rc_gif
    total_bi_loss = gif_norm(data_bi_gif - data_hr_gif, True)
    total_loss = gif_norm(data_rc_gif - data_hr_gif, True)
    data_rc_gif_out = toimg(data_rc_gif)

    '''
    Step 5: Save GIF.
    '''    
    save_frames(gif=data_rc_gif_out, dir='../../data/rc_imgs/', number=number)





