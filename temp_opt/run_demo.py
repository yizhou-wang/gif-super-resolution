import os
import time
import numpy as np
import scipy.io

from main import main

gif_dir = '../../data/raw_gifs/'
tag = 'yizhou_all_g5'
hr = 128
lr = 32

number_list = []
# choose_gif = []
for file in os.listdir(gif_dir + tag + '/'):
    if file.endswith(".gif"):
    	number_list.append(file.translate(None, '.gif'))

number_list.sort(key=lambda f: int(filter(str.isdigit, f)))
# number_list = number_list[4178:]
print '***', len(number_list), 'GIFs found! ***'
# number_list = number_list[:10]

number = np.zeros(len(number_list))
frame_num = np.zeros(len(number_list))
total_bi_psnr = np.zeros(len(number_list))
total_psnr = np.zeros(len(number_list))

start_time = time.time()
for idx, number in enumerate(number_list):
	# print '*** Processing ' + number + '.gif ***'
	# os.system("python main.py -n " + number)
	frame_num[idx], total_bi_psnr[idx], total_psnr[idx] = main(tag=tag, number=number, hr=hr, lr=lr, numIterations=50)
	# print 'frame_num[idx] =', frame_num[idx]
	# print 'total_bi_psnr[idx] =', total_bi_psnr[idx]
	# print 'total_psnr[idx] =', total_psnr[idx]
	print 'Current Time Consumed:', time.time() - start_time, 's'
	# if frame_num[idx] > 5 and frame_num[idx] < 500:
	# 	if total_bi_psnr[idx] < total_psnr[idx]:
	# 		# choose_gif.append(number)
	# 		thefile = open('../../data/choose_gif_g5.txt', 'a')
	# 		thefile.write("face/%s.gif\n" % (number))
	# 		thefile.close()

print 'Total Time Consumed:', time.time() - start_time, 's'
print 'Mean of Frame NUM: ', np.mean(frame_num)
print 'Mean of BI PSNR:', np.mean(total_bi_psnr)
print 'Mean of Our PSNR:', np.mean(total_psnr) 

filename = '../../data/final.mat'
scipy.io.savemat(filename, mdict={'frame_num': frame_num, 'total_bi_psnr': total_bi_psnr, 'total_psnr': total_psnr}) 
# print choose_gif


