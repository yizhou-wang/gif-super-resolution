from shutil import copyfile
import os

filename = '../../data/choose_gif_g5.txt'

with open(filename) as f:
	data = f.readlines()

for n, line in enumerate(data):
	line = line.rstrip()
	tmp = line.split('/')[-1]
	src = '../../data/raw_gifs/' + line
	dst = '../../data/raw_gifs/yizhou_all_g5/' + tmp
	if not os.path.exists('../../data/raw_gifs/yizhou_all_g5/'):
		os.makedirs('../../data/raw_gifs/yizhou_all_g5/')
	# print src, dst
	copyfile(src, dst)