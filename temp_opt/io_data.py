import glob
import numpy as np
import scipy.ndimage, scipy.misc
import os, sys
from subprocess import call

def load_lr_gif(dir='../data/lr_imgs/', tag='face', number='999', reso=8, channel=3):
    print 'lr_path =', dir + tag + '/' + number + '/*.png'
    lr_list = glob.glob(dir + tag + '/' + number + '/*.png')
    lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print lr_list
    data_lr_gif = np.zeros((len(lr_list), reso, reso, channel))

    for idx, lr_img in enumerate(lr_list):
        im = scipy.ndimage.imread(lr_img)
        data_lr_gif[idx, :, :, :] = im

    return data_lr_gif

def load_hr_gif(dir='../data/hr_imgs/', tag='face', number='999', reso=32, channel=3):
    print 'hr_path =', dir + tag + '/' + number + '/*.png'
    hr_list = glob.glob(dir + tag + '/' + number + '/*.png')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print hr_list
    data_hr_gif = np.zeros((len(hr_list), reso, reso, channel))

    for idx, hr_img in enumerate(hr_list):
        im = scipy.ndimage.imread(hr_img)
        data_hr_gif[idx, :, :, :] = im

    return data_hr_gif

def load_fl_frame(dir='../data/hr_imgs/', tag='face', number='999',reso=32, channel=3):
    print 'hr_path =', dir + tag + '/' + number + '/*.png'
    hr_list = glob.glob(dir + tag + '/' + number + '/*.png')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print hr_list
    data_fl_frame = np.zeros((2, reso, reso, channel))
    # Load first frame
    f_frame = scipy.ndimage.imread(hr_list[0])
    data_fl_frame[0, :, :, :] = f_frame
    # Load last frame
    l_frame = scipy.ndimage.imread(hr_list[-1])
    data_fl_frame[1, :, :, :] = l_frame
    return data_fl_frame

def load_bi_gif(dir='../data/bi_imgs/', tag='face', number='999',reso=32, channel=3):
    print 'bi_path =', dir + tag + '/' + number + '/*.png'
    bi_list = glob.glob(dir + tag + '/' + number + '/*.png')
    bi_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print bi_list
    data_bi_gif = np.zeros((len(bi_list), reso, reso, channel))

    for idx, bi_img in enumerate(bi_list):
        im = scipy.ndimage.imread(bi_img)
        data_bi_gif[idx, :, :, :] = im

    return data_bi_gif

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

def img2gif(in_dir='../data/rc_imgs/', tag='face', number='999', out_dir='../data/rc_gifs/'):
    """
    Converts png frames to gif with each png containing the full image data
    """
    img_dir = in_dir + tag + '/' + number + '/'
    gif_dir = out_dir + tag + '/'
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    call(["convert", "-delay", str(0.2), "-loop", "0", img_dir + "*.png", gif_dir + number + ".gif"])

