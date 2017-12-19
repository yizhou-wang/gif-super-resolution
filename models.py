from utils import *

def get_loss_gradiant(frame_num, params, data_fl_frame, data_bi_gif):
    n = frame_num
    rho = params[0]
    gamma = params[1]
    # print np.mean(rho)
    # print 'LG:', params.shape
    # print 'LG:', rho.shape
    # print 'LG:', gamma.shape
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += rho**(n-i) * data_bi_gif[i, :, :, :]
    Fn = rho**n * F0_gt + gamma * sum0
    # global TEST_COUNTER
    # if TEST_COUNTER % 10 == 0:
    #     save_frame(Fn, dir='../../data/test/', f=TEST_COUNTER)
    # TEST_COUNTER += 1
    loss = gif_norm(Fn - Fn_gt, False)
    # Compute Fn_rho = partial_Fn / partial_rho
    sum1 = 0
    for i in range(1, n-1):
        sum1 += (n-i) * rho**(n-i-1) * data_bi_gif[i, :, :, :]
    Fn_rho = n * rho**(n-1) * F0_gt + gamma * sum1
    # Compute Fn_gamma = partial_Fn / partial_gamma
    Fn_gamma = 0
    for i in range(1, n):
        Fn_gamma += rho**(n-i) * data_bi_gif[i, :, :, :]
    # Compute partial_rho = partial_l / partial_rho
    partial_rho = 2 * (Fn - Fn_gt) * Fn_rho
    # Compute partial_gamma = partial_l / partial_gamma
    partial_gamma = 2 * (Fn - Fn_gt) * Fn_gamma
    grad = np.array([partial_rho, partial_gamma])
    # grad = np.array([np.mean(partial_rho), np.mean(partial_gamma)])
    psnr = PSNR(Fn_gt, Fn)
    return loss, grad, psnr

def get_loss_gradiant_sum1(frame_num, params, data_fl_frame, data_bi_gif):
    n = frame_num
    rho = params
    gamma = np.ones_like(params) - rho
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += rho**(n-i) * gamma * data_bi_gif[i, :, :, :]
    Fn = rho**n * F0_gt + sum0
    loss = gif_norm(Fn - Fn_gt, False)
    # Compute Fn_rho = partial_Fn / partial_rho
    sum1 = 0
    for i in range(1, n-1):
        sum1 += ((n-i) * rho**(n-i-1) - (n-i+1) * rho**(n-i)) * data_bi_gif[i, :, :, :]
    Fn_rho = n * rho**(n-1) * F0_gt + sum1
    # Compute partial_rho = partial_l / partial_rho
    partial_rho = 2 * Fn * Fn_rho - 2 * Fn_gt * Fn_rho
    # Compute partial_gamma = partial_l / partial_gamma
    grad = partial_rho
    return loss, grad

def get_loss_gradiant_gamma(frame_num, params, data_fl_frame, data_bi_gif):
    n = frame_num
    rho = np.ones_like(params)
    gamma = params
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += data_bi_gif[i, :, :, :]
    Fn = F0_gt + gamma * sum0
    loss = gif_norm(Fn - Fn_gt, False)
    # Compute Fn_rho = partial_Fn / partial_rho
    Fn_gamma = sum0
    # Compute partial_rho = partial_l / partial_rho
    partial_gamma = 2 * (Fn - Fn_gt) * Fn_gamma
    # Compute partial_gamma = partial_l / partial_gamma
    grad = partial_gamma
    return loss, grad

def get_loss_gradiant_rho(frame_num, params, data_fl_frame, data_bi_gif):
    n = frame_num
    rho = params
    gamma = np.ones_like(params)
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += rho**(n-i) * data_bi_gif[i, :, :, :]
        # print sum0.shape
    Fn = rho**n * F0_gt + sum0
    loss = gif_norm(Fn - Fn_gt, False)
    # Compute Fn_rho = partial_Fn / partial_rho
    sum1 = 0
    for i in range(1, n-1):
        sum1 += (n-i) * rho**(n-i-1) * data_bi_gif[i, :, :, :]
    Fn_rho = n * rho**(n-1) * F0_gt + sum1
    # Compute partial_rho = partial_l / partial_rho
    partial_rho = 2 * (Fn - Fn_gt) * Fn_rho
    # Compute partial_gamma = partial_l / partial_gamma
    grad = partial_rho
    return loss, grad

def get_loss_gradiant_all(frame_num, params, data_fl_frame, data_bi_gif, alpha=0.5):
    n = frame_num
    rho = params[0]
    gamma = params[1]
    # print 'LG:', params.shape
    # print 'LG:', rho.shape
    # print 'LG:', gamma.shape
    F0_gt = data_fl_frame[0, :, :, :]
    Fn_gt = data_fl_frame[1, :, :, :]
    # Compute Fn
    sum0 = 0
    for i in range(1, n):
        sum0 += rho**(n-i) * data_bi_gif[i, :, :, :]
    Fn = rho**n * F0_gt + gamma * sum0

    Ft = np.zeros((n-1, hr, hr, channel))
    for t in range(1, n-1):
        sum0 = 0
        for i in range(1, t):
            sum0 += rho**(t-i) * data_bi_gif[i, :, :, :]
        Ft[t] = rho**t * F0_gt + gamma * sum0

    loss = gif_norm(Fn - Fn_gt, False)
    # Compute Fn_rho = partial_Fn / partial_rho
    sum1 = 0
    for i in range(1, n-1):
        sum1 += (n-i) * rho**(n-i-1) * data_bi_gif[i, :, :, :]
    Fn_rho = n * rho**(n-1) * F0_gt + gamma * sum1
    # Compute Fn_gamma = partial_Fn / partial_gamma
    Fn_gamma = 0
    for i in range(1, n):
        Fn_gamma += rho**(n-i) * data_bi_gif[i, :, :, :]

    Ft_rho = np.zeros((n-1, hr, hr, channel))
    for t in range(1, n-1):
        sum1 = 0
        for i in range(1, t-1):
            sum1 += (t-i) * rho**(t-i-1) * data_bi_gif[i, :, :, :]
            # print '!!!!!!!!!!!!', t-i
            # print 'sum1 =', np.mean(sum1)
        Ft_rho[t] = t * rho**(t-1) * F0_gt + gamma * sum1
        # print 'Ft_rho = ', np.mean(Ft_rho[t])
    Ft_gamma = np.zeros((n-1, hr, hr, channel))
    for t in range(1, n-1):
        for i in range(1, t):
            Ft_gamma[t] += rho**(t-i) * data_bi_gif[i, :, :, :]

    # Compute partial_rho = partial_l / partial_rho
    sum3 = 0
    for t in range(1, n-1):
        sum3 += 2 * (Ft[t] - data_bi_gif[t]) * Ft_rho[t]
    partial_rho = 2 * (Fn - Fn_gt) * Fn_rho + alpha / (n-1) * sum3
    # Compute partial_gamma = partial_l / partial_gamma
    sum4 = 0
    for t in range(1, n-1):
        sum4 += 2 * (Ft[t] - data_bi_gif[t]) * Ft_gamma[t]    
    partial_gamma = 2 * (Fn - Fn_gt) * Fn_gamma + alpha / (n-1) * sum4
    grad = np.array([partial_rho, partial_gamma])
    # grad = np.array([np.mean(partial_rho), np.mean(partial_gamma)])
    return loss, grad

def recover_gif(data_bi_gif, data_fl_frame, params, keep_fl=True, use_iter=False):
    frame_num = data_bi_gif.shape[0]
    data_rc_gif = np.zeros_like(data_bi_gif)
    rho = params[0]
    gamma = params[1]
    if keep_fl == True:
        data_rc_gif[0] = data_fl_frame[0]
        data_rc_gif[-1] = data_fl_frame[1]
        if use_iter == True:
            for i in range(1, frame_num-1):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]
        else:
            for i in range(1, frame_num-1):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * gamma * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + sum2
    else:
        data_rc_gif[0] = data_fl_frame[0]
        if use_iter == True:
            for i in range(1, frame_num):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]        
        else:
            for i in range(1, frame_num):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * gamma * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + sum2
    # print data_rc_gif[i]
    return data_rc_gif

def recover_gif_sum1(data_bi_gif, data_fl_frame, params, keep_fl=True, use_iter=False):
    frame_num = data_bi_gif.shape[0]
    data_rc_gif = np.zeros_like(data_bi_gif)
    rho = params
    gamma = np.ones_like(params) - rho
    if keep_fl == True:
        data_rc_gif[0] = data_fl_frame[0]
        data_rc_gif[-1] = data_fl_frame[1]
        if use_iter == True:
            for i in range(1, frame_num-1):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]
        else:
            for i in range(1, frame_num-1):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * gamma * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + sum2
    else:
        data_rc_gif[0] = data_fl_frame[0]
        if use_iter == True:
            for i in range(1, frame_num):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]        
        else:
            for i in range(1, frame_num):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * gamma * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + sum2
    # print data_rc_gif[i]
    return data_rc_gif

def recover_gif_gamma(data_bi_gif, data_fl_frame, params, keep_fl=True, use_iter=False):
    frame_num = data_bi_gif.shape[0]
    data_rc_gif = np.zeros_like(data_bi_gif)
    rho = np.ones_like(params)
    gamma = params
    if keep_fl == True:
        data_rc_gif[0] = data_fl_frame[0]
        data_rc_gif[-1] = data_fl_frame[1]
        if use_iter == True:
            for i in range(1, frame_num-1):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]
        else:
            for i in range(1, frame_num-1):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + gamma * sum2
    else:
        data_rc_gif[0] = data_fl_frame[0]
        if use_iter == True:
            for i in range(1, frame_num):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]        
        else:
            for i in range(1, frame_num):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + gamma * sum2
    # print data_rc_gif[i]
    return data_rc_gif

def recover_gif_rho(data_bi_gif, data_fl_frame, params, keep_fl=True, use_iter=False):
    frame_num = data_bi_gif.shape[0]
    data_rc_gif = np.zeros_like(data_bi_gif)
    rho = params
    gamma = np.ones_like(params)
    if keep_fl == True:
        data_rc_gif[0] = data_fl_frame[0]
        data_rc_gif[-1] = data_fl_frame[1]
        if use_iter == True:
            for i in range(1, frame_num-1):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]
        else:
            for i in range(1, frame_num-1):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + gamma * sum2
    else:
        data_rc_gif[0] = data_fl_frame[0]
        if use_iter == True:
            for i in range(1, frame_num):
                data_rc_gif[i] = rho * data_rc_gif[i-1] + gamma * data_bi_gif[i]        
        else:
            for i in range(1, frame_num):
                sum2 = 0
                for j in range(1, i):
                    sum2 += rho**(i-j) * data_bi_gif[j] 
                data_rc_gif[i] = rho**i * data_rc_gif[0] + gamma * sum2
    # print data_rc_gif[i]
    return data_rc_gif
