import glob
import numpy as np
from PIL import Image
import scipy.ndimage, scipy.misc
import os, sys
from subprocess import call
import cv2
import scipy.io

def gif2img(in_dir='../data/raw_gifs/', tag='face', number='999', out_dir='../data/raw_imgs/'):
    '''
    Split GIF into images.

        The images in each GIF are stored in a folder.
        The names of images are always 6 digits, like: 000001.jpg.
        
        e.g. GIF named 001.gif, the images are stored in a folder named 001. 

    '''
    gif_dir = in_dir + tag + '/' + number + '.gif'
    img_dir = out_dir + tag + '/' + number + '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    call(["convert", "-coalesce", gif_dir, img_dir + "%06d.jpg"])

def gen_hr_lr(in_dir='../data/raw_imgs/', tag='face', number='999', hr_dir='../data/hr_imgs/', lr_dir='../data/lr_imgs/', reso=(32, 8)):
    '''
    Generate high-resolution and low-resolution images using the original images.
    '''
    hr = (reso[0], reso[0])
    lr = (reso[1], reso[1])
    img_dir = in_dir + tag + '/' + number + '/'
    hr_dir = hr_dir + tag + '/' + number + '/'
    lr_dir = lr_dir + tag + '/' + number + '/'
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    jpg_list = glob.glob(img_dir + '*.jpg')
    jpg_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # jpg_list = jpg_list[:10]
    for jpg in jpg_list:
        im1 = Image.open(jpg)
        im2 = im1.resize(hr).convert('RGB')
        im3 = im1.resize(lr).convert('RGB')
        q1 = hr_dir + jpg.split('/')[-1]
        q2 = lr_dir + jpg.split('/')[-1]
        im2.save(q1)
        im3.save(q2)

def load_lr_gif(dir='../data/lr_imgs/', tag='face', number='999', reso=8, channel=3):
    # print 'lr_path =', dir + tag + '/' + number + '/*.jpg'
    lr_list = glob.glob(dir + tag + '/' + number + '/*.jpg')
    lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print lr_list
    data_lr_gif = np.zeros((len(lr_list), reso, reso, channel))

    for idx, lr_img in enumerate(lr_list):
        im = scipy.ndimage.imread(lr_img)
        data_lr_gif[idx, :, :, :] = im

    return data_lr_gif

def load_hr_gif(dir='../data/hr_imgs/', tag='face', number='999', reso=32, channel=3):
    # print 'hr_path =', dir + tag + '/' + number + '/*.jpg'
    hr_list = glob.glob(dir + tag + '/' + number + '/*.jpg')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    # print hr_list
    data_hr_gif = np.zeros((len(hr_list), reso, reso, channel))

    for idx, hr_img in enumerate(hr_list):
        im = scipy.ndimage.imread(hr_img)
        data_hr_gif[idx, :, :, :] = im

    return data_hr_gif

def load_fl_frame(dir='../data/hr_imgs/', tag='face', number='999', reso=32, channel=3):
    # print 'hr_path =', dir + tag + '/' + number + '/*.jpg'
    hr_list = glob.glob(dir + tag + '/' + number + '/*.jpg')
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

def bicu_inter(in_dir='../data/lr_imgs/', tag='face', number='999', out_dir='../data/bi_imgs/', reso=(32, 8)):
    '''
    Generate Bicubic Interpolation of the low-resolution images.
    '''
    hr = (reso[0], reso[0])
    lr = (reso[1], reso[1])
    lr_dir = in_dir + tag + '/' + number + '/'
    bi_dir = out_dir + tag + '/' + number + '/'
    if not os.path.exists(bi_dir):
        os.makedirs(bi_dir)
    lr_list = glob.glob(lr_dir + '*.jpg')
    lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    for lr in lr_list:
        im1 = cv2.imread(lr)
        im2 = cv2.resize(im1, dsize=hr, interpolation=cv2.INTER_CUBIC)
        q = bi_dir + lr.split('/')[-1]          
        cv2.imwrite(q, im2)

def load_bi_gif(dir='../data/bi_imgs/', tag='face', number='999',reso=32, channel=3):
    # print 'bi_path =', dir + tag + '/' + number + '/*.jpg'
    bi_list = glob.glob(dir + tag + '/' + number + '/*.jpg')
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
    name = save_dir + str(f).zfill(6) + '.jpg'
    scipy.misc.imsave(name, img)
    # wait()

def save_frames(gif, dir='../data/rc_imgs/', tag='face', number='999'):
    save_dir = dir + tag + '/' + number + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    size = gif.shape
    for f in range(size[0]):
        name = save_dir + str(f).zfill(6) + '.jpg'
        scipy.misc.imsave(name, gif[f])

def img2gif(in_dir='../data/rc_imgs/', tag='face', number='999', out_dir='../data/rc_gifs/'):
    """
    Converts jpg frames to gif with each jpg containing the full image data
    """
    img_dir = in_dir + tag + '/' + number + '/'
    gif_dir = out_dir + tag + '/'
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    call(["convert", "-delay", str(0.2), "-loop", "0", img_dir + "*.jpg", gif_dir + number + ".gif"])

def savemat(hr_dir='../data/hr_imgs/', lr_dir='../data/lr_imgs/', tag='face', number='999', out_dir='../data/mats/', reso=(32, 8), channel=3):
    hr = reso[0]
    lr = reso[1]
    # print 'lr_path =', lr_dir + tag + '/' + number + '/*.jpg'
    lr_list = glob.glob(lr_dir + tag + '/' + number + '/*.jpg')
    lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    data_lr_gif = np.zeros((lr, lr, len(lr_list)))
    data_lr_gif_1 = np.zeros((lr, lr, len(lr_list)))
    data_lr_gif_2 = np.zeros((lr, lr, len(lr_list)))
    for idx, lr_img in enumerate(lr_list):
        # im = scipy.ndimage.imread(lr_img, mode='L')
        im = scipy.ndimage.imread(lr_img)
        data_lr_gif[:, :, idx] = im[:,:,0]
        data_lr_gif_1[:, :, idx] = im[:,:,1]
        data_lr_gif_2[:, :, idx] = im[:,:,2]
    # print 'hr_path =', hr_dir + tag + '/' + number + '/*.jpg'
    hr_list = glob.glob(hr_dir + tag + '/' + number + '/*.jpg')
    hr_list.sort(key=lambda f: int(filter(str.isdigit, f)))
    data_hr_gif = np.zeros((hr, hr, 1, len(hr_list)))
    data_hr_gif_1 = np.zeros((hr, hr, 1, len(hr_list)))
    data_hr_gif_2 = np.zeros((hr, hr, 1, len(hr_list)))
    for idx, hr_img in enumerate(hr_list):
        # im = scipy.ndimage.imread(hr_img, mode='L')
        im = scipy.ndimage.imread(hr_img)
        data_hr_gif[:, :, 0, idx] = im[:,:,0]
        data_hr_gif_1[:, :, 0, idx] = im[:,:,1]
        data_hr_gif_2[:, :, 0, idx] = im[:,:,2]
    data_lr_gif = data_lr_gif / 255.0
    data_hr_gif = data_hr_gif / 255.0
    data_lr_gif_1 = data_lr_gif_1 / 255.0
    data_hr_gif_1 = data_hr_gif_1 / 255.0
    data_lr_gif_2 = data_lr_gif_2 / 255.0
    data_hr_gif_2 = data_hr_gif_2 / 255.0

    if not os.path.exists(out_dir + tag + '/' + number):
        os.makedirs(out_dir + tag + '/' + number)
    filename = out_dir + tag + '/' + number + '/' + number + '_R.mat'
    scipy.io.savemat(filename, mdict={'frames': data_lr_gif, 'im_gt': data_hr_gif})
    filename = out_dir + tag + '/' + number + '/' + number + '_G.mat'
    scipy.io.savemat(filename, mdict={'frames': data_lr_gif_1, 'im_gt': data_hr_gif_1})
    filename = out_dir + tag + '/' + number + '/' + number + '_B.mat'
    scipy.io.savemat(filename, mdict={'frames': data_lr_gif_2, 'im_gt': data_hr_gif_2})
