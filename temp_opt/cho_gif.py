from shutil import copyfile
import os

filename = '../../data/cho-gif-5-400-total.txt'

with open(filename) as f:
	data = f.readlines()

for n, line in enumerate(data):
	line = line.rstrip()
	tmp = line.split('/')[-1]
	src = '../../data/raw_gifs/' + line
	dst_dir = '../../data/raw_gifs/yizhou-5-400-total/'
	dst = dst_dir + tmp
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)
	# print src, dst
	copyfile(src, dst)