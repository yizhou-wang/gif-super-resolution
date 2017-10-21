import numpy as np
import time

from io_data import *
from utils import *
from models import *

hr = 256
lr = 64
channel = 3
# frame_num = 0
# TEST_COUNTER = 0

def GD(data_bi_gif, data_fl_frame, params, step_size, numIterations, data_hr_gif, print_tloss=True):
    frame_num = data_bi_gif.shape[0]
    bi_loss = gif_norm(data_bi_gif[-1, :, :, :] - data_fl_frame[1, :, :, :], False)
    # print 'GD:', params.shape
    for i in range(0, numIterations):
        start_time = time.time()
        loss, grad_l = get_loss_gradiant(frame_num, params, data_fl_frame, data_bi_gif)
        if print_tloss:
            data_rc_gif = recover_gif(data_bi_gif, data_fl_frame, params, keep_fl=False, use_iter=False)
            total_bi_loss = gif_norm(data_bi_gif - data_hr_gif, True)
            total_loss = gif_norm(data_rc_gif - data_hr_gif, True)
            print("Step %d ... frame_loss: %f / %f | Total_loss: %f / %f | time: %.8s s" % (i, bi_loss, loss, total_bi_loss, total_loss, (time.time() - start_time)))
        else:
            print("Step %d ... frame_loss: %f / %f | time: %.8s s" % (i, bi_loss, loss, (time.time() - start_time)))            
        # Update
        # print np.mean(step_size * grad_l)
        params = params - step_size * grad_l
        # print np.mean(params[0]), np.mean(params[1])
    return params

if __name__ == '__main__':

    # Choose a GIF for test
    number = '1'

    '''
    Step 1: Extract frames.
    '''
    print 'Extract frames ...'
    gif2img(in_dir='../../data/raw_gifs/', number=number, out_dir='../../data/raw_imgs/')
    print 'Gerate HR & LR ...'
    gen_hr_lr(in_dir='../../data/raw_imgs/', number=number, hr_dir='../../data/hr_imgs/', lr_dir='../../data/lr_imgs/', reso=(hr, lr))

    '''
    Step 2: Read images.
        'data_lr_gif':      read lr GIF in a array (frame X 8 X 8 X 3)
        'data_hr_gif':      read hr GIF (GT) in a array (frame X 32 X 32 X 3)
        'data_fl_frame':    read first and last frame (GT) in a array (2 X 32 X 32 X 3)
    '''
    # data_lr_gif = load_lr_gif(dir='../../data/lr_imgs/', number=number, reso=lr)
    # print 'data_lr_gif =', data_lr_gif.shape
    data_hr_gif = load_hr_gif(dir='../../data/hr_imgs/', number=number, reso=hr)
    print 'data_hr_gif =', data_hr_gif.shape
    data_fl_frame = load_fl_frame(dir='../../data/hr_imgs/', number=number, reso=hr)
    print 'data_fl_frame =', data_fl_frame.shape

    '''
    Step 3: BI on each frame.
        'data_bi_gif':  bicubic interpolation on each frame (frame X 32 X 32 X 3)
        'bi_loss':      loss of the bicubic interpolation
    '''
    print 'Bicubic interpolation ...'
    bicu_inter(in_dir='../../data/lr_imgs/', number=number, out_dir='../../data/bi_imgs/', reso=(hr, lr))
    data_bi_gif = load_bi_gif(dir='../../data/bi_imgs/', number=number, reso=hr)
    print 'data_bi_gif =', data_bi_gif.shape

    '''
    Step 4: Optimization.
        - Compute cost = BI cost + TR cost
        - Gradient descent
        - Next iteration
    '''
    # scaler_params = np.array([0.5, 0.5])  
    # scaler_params_res = GD(data_bi_gif, data_fl_frame, scaler_params, 0.001, 100)
    # mat_params = np.array([0.5, 0.5])
    start_time = time.time()
    # mat_params = np.tile(0.5, (hr, hr,channel))
    mat_params = np.array([np.tile(0.9, (hr, hr, channel)), np.tile(0.1, (hr, hr, channel))])
    mat_params_res = GD(data_bi_gif, data_fl_frame, mat_params, 1e-8, 100, data_hr_gif, False)
    print("Optimization completed! Time consumed: %.8s s" % ((time.time() - start_time)))
    print np.mean(mat_params_res)
    # print mat_params_res[1,:,:,0]

    '''
    Step 5: Recover GIF.
        'data_rc_gif':      recovered GIF (frame X 32 X 32 X 3)
        'data_rc_gif_out':  map to [0, 255] (frame X 32 X 32 X 3)
    '''
    start_time = time.time()
    data_rc_gif = recover_gif(data_bi_gif, data_fl_frame, mat_params_res, keep_fl=True, use_iter=False)
    print("Recovering completed! Time consumed: %.8s s" % ((time.time() - start_time)))
    total_bi_loss = gif_norm(data_bi_gif - data_hr_gif, True)
    total_loss = gif_norm(data_rc_gif - data_hr_gif, True)
    print("Total_loss: %f / %f" % (total_bi_loss, total_loss))

    '''
    Step 6: Save GIF.
    '''    
    data_rc_gif_out = toimg(data_rc_gif)
    save_frames(gif=data_rc_gif_out, dir='../../data/rc_imgs/', number=number)
    print 'Frames saved!'
    img2gif(in_dir='../../data/hr_imgs/', number=number, out_dir='../../data/hr_gifs/')
    img2gif(in_dir='../../data/rc_imgs/', number=number, out_dir='../../data/rc_gifs/')
    img2gif(in_dir='../../data/bi_imgs/', number=number, out_dir='../../data/bi_gifs/')
    print 'GIF saved!'





