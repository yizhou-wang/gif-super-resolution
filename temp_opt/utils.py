import numpy as np
import numpy.linalg
import cv2
from scipy.signal import butter, lfilter, freqz

def gif_norm(gif, mutli_frame=True):
    size = gif.shape
    res = 0
    if mutli_frame == True:
        frame_num = size[0]
        channel = size[3]
        for f in range(frame_num):
            for c in range(channel):
                res += numpy.linalg.norm(gif[f, :, :, c])
    else:
        channel = size[2]
        for c in range(channel):
            res += numpy.linalg.norm(gif[:, :, c])
    return res

def PSNR(y_true, y_pred):
    shape = y_true.shape
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    if len(shape) == 3:
        y_true_ycc = cv2.cvtColor(y_true, cv2.COLOR_BGR2YCR_CB)
        y_pred_ycc = cv2.cvtColor(y_pred, cv2.COLOR_BGR2YCR_CB)
        y_true_y = y_true_ycc[:, :, 0]
        y_pred_y = y_pred_ycc[:, :, 0]
    elif len(shape) == 4:
        y_true_ycc = np.zeros_like(y_true)
        y_pred_ycc = np.zeros_like(y_pred)
        for f in range(shape[0]):
            y_true_ycc[f] = cv2.cvtColor(y_true[f], cv2.COLOR_BGR2YCR_CB)
            y_pred_ycc[f] = cv2.cvtColor(y_pred[f], cv2.COLOR_BGR2YCR_CB)
        y_true_y = y_true_ycc[:, :, :, 0]
        y_pred_y = y_pred_ycc[:, :, :, 0]
    else:
        print('ERROR: Image shape incorrect!')
    y_true_y = y_true_y.astype(np.float)
    y_pred_y = y_pred_y.astype(np.float)
    # print y_pred_y.shape
    return 20.0 * np.log(255.0) / np.log(10.0) - 10.0 * np.log(np.mean(np.square(y_pred_y - y_true_y))) / np.log(10.0)
    # return 20.0 * np.log(255.0) / np.log(10.0) - 10.0 * np.log(np.mean(np.square(y_pred - y_true))) / np.log(10.0)

def wait():
    raw_input("Press Enter to continue...")

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