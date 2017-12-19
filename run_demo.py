import os
import time
import numpy as np
import scipy.io

from main import main

gif_dir = '../data/raw_gifs/'
# tag = gifsr, expression, action, scene, animation, animal
tag = 'expression'
hr = 128
lr = 32

number_list = []
for file in os.listdir(gif_dir + tag + '/'):
    if file.endswith(".gif"):
    	number_list.append(file.translate(None, '.gif'))

number_list.sort(key=lambda f: int(filter(str.isdigit, f)))
# number_list = number_list[4178:]
print '***', len(number_list), 'GIFs found! ***'
# number_list = number_list[:10]

id = np.zeros(len(number_list))
frame_num = np.zeros(len(number_list))
total_bi_psnr = np.zeros(len(number_list))
total_psnr = np.zeros(len(number_list))
id_list = []
frame_num_list = []
total_bi_psnr_list = []
total_psnr_list = []
time_list = []


start_time = time.time()
for idx, number in enumerate(number_list):
	# print '*** Processing ' + number + '.gif ***'
	# os.system("python main.py -n " + number)
	id[idx] = int(number)
	cur_time = time.time()
	frame_num[idx], total_bi_psnr[idx], total_psnr[idx] = main(tag=tag, number=number, hr=hr, lr=lr, numIterations=10)
	# print 'frame_num[idx] =', frame_num[idx]
	# print 'total_bi_psnr[idx] =', total_bi_psnr[idx]
	# print 'total_psnr[idx] =', total_psnr[idx]
	tmp_time = time.time() - cur_time
	print 'Current Time Consumed:', tmp_time, 's'
	# if frame_num[idx] > 5 and frame_num[idx] < 400:
	# 	if total_bi_psnr[idx] < total_psnr[idx]:
	# 		time_list.append(tmp_time)
	# 		id_list.append(int(number))
	# 		frame_num_list.append(frame_num[idx])
	# 		total_bi_psnr_list.append(total_bi_psnr[idx])
	# 		total_psnr_list.append(total_psnr[idx])		
	# 		thefile = open('../data/choose_gif_g5.txt', 'a')
	# 		thefile.write("%s/%s.gif\n" % (tag, number))
	# 		thefile.close()

id_list = np.array(id_list)
frame_num_list = np.array(frame_num_list)
total_bi_psnr_list = np.array(total_bi_psnr_list)
total_psnr_list = np.array(total_psnr_list)
time_list = np.array(time_list)

print 'Total Time Consumed:', time.time() - start_time, 's'
print 'Mean of Frame NUM: ', np.mean(frame_num_list)
print 'Mean of BI PSNR:', np.mean(total_bi_psnr_list)
print 'Mean of Our PSNR:', np.mean(total_psnr_list) 

filename = '../result/final-' + tag + '.mat'
scipy.io.savemat(filename, mdict={'id': id, 'frame_num': frame_num, 'total_bi_psnr': total_bi_psnr, 'total_psnr': total_psnr}) 
filename = '../result/final-list-' + tag + '.mat'
scipy.io.savemat(filename, mdict={'id': id_list, 'frame_num': frame_num_list, 'total_bi_psnr': total_bi_psnr_list, 'total_psnr': total_psnr_list, 'time': time_list}) 


