from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import cv2

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

def optical_flow(data_gif):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    old_frame = data_gif[0, :, :, :].astype(np.uint8)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    for idx in range(data_gif.shape[0]-1):
        frame = data_gif[idx+1, :, :, :].astype(np.uint8)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # print p1, st, err
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # print frame.shape, mask.shape
        img = cv2.add(frame,mask)
        # print img.shape
        cv2.imshow('frame',img)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    # cv2.destroyAllWindows()
    cv2.imwrite('out.jpg', img)
