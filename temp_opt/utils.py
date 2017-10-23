import numpy as np
import numpy.linalg
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
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 20.0 * np.log(255.0) / np.log(10.0) - 10.0 * np.log(np.mean(np.square(y_pred - y_true))) / np.log(10.0)

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