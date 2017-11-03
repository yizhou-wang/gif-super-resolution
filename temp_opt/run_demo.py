import os
import time
import numpy as np

from main import main

thefile = open('../../data/choose_gif.txt', 'w')

gif_dir = '../../data/raw_gifs/'
tag = 'face'
hr = 64
lr = 16

number_list = []
choose_gif = []
for file in os.listdir(gif_dir + tag + '/'):
    if file.endswith(".gif"):
    	number_list.append(file.translate(None, '.gif'))

number_list.sort(key=lambda f: int(filter(str.isdigit, f)))
print '***', len(number_list), 'GIFs found! ***'
# number_list = number_list[:10]

total_bi_psnr = np.zeros(len(number_list))
total_psnr = np.zeros(len(number_list))

start_time = time.time()
for idx, number in enumerate(number_list):
	# print '*** Processing ' + number + '.gif ***'
	# os.system("python main.py -n " + number)
	total_bi_psnr[idx], total_psnr[idx] = main(tag=tag, number=number, hr=hr, lr=lr, numIterations=50)
	if total_bi_psnr[idx] < total_psnr[idx]:
		choose_gif.append(number)
		thefile.write("%s/%s.gif\n" % (tag, number))

print 'Total Time Consumed:', time.time() - start_time, 's'
print 'Mean of BI PSNR:', np.mean(total_bi_psnr)
print 'Mean of Our PSNR:', np.mean(total_psnr)  
print choose_gif
thefile.close()


