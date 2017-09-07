import os
import glob
from PIL import Image


def gen_hr_lr(png_dir='../data/raw_imgs/', tag='face', hr_dir='../data/hr_imgs/', lr_dir='../data/lr_imgs/'):
	'''
	Generate high-resolution and low-resolution images using the original images.

	'''

	hr = (32, 32)
	lr = (8, 8)

	if not os.path.exists(hr_dir):
		os.mkdir(hr_dir)
	if not os.path.exists(lr_dir):
		os.mkdir(lr_dir)

	if not os.path.exists(hr_dir + tag):
		os.mkdir(hr_dir + tag)
	if not os.path.exists(lr_dir + tag):
		os.mkdir(lr_dir + tag)

	png_list = glob.glob(png_dir + tag + '/*/')
	png_list.sort(key=lambda f: int(filter(str.isdigit, f)))
	# print png_list

	for idx, png in enumerate(png_list):

		if not os.path.exists(hr_dir + tag + '/' + str(idx)):
			os.mkdir(hr_dir + tag + '/' + str(idx))
		if not os.path.exists(lr_dir + tag + '/' + str(idx)):
			os.mkdir(lr_dir + tag + '/' + str(idx))

		print("Resizing pictures in %d.gif"%(idx))
		p_list = glob.glob(png + '*.png')
		p_list.sort(key=lambda f: int(filter(str.isdigit, f)))
		# print p_list

		for p in p_list:
			im1 = Image.open(p)
			im2 = im1.resize(hr)
			im3 = im1.resize(lr)
			# print p.split('/')

			q1 = hr_dir + p.split('/')[3] + '/' + p.split('/')[4] + '/' + p.split('/')[5]
			q2 = lr_dir + p.split('/')[3] + '/' + p.split('/')[4] + '/' + p.split('/')[5]

			print q1
			print q2

			im2.save(q1)
			im3.save(q2)



if __name__ == '__main__':

	PNG_DIR = '../data/raw_imgs/'
	TAG = 'face'
	HR_DIR = '../data/hr_imgs/'
	LR_DIR = '../data/lr_imgs/'

	gen_hr_lr(PNG_DIR, TAG, HR_DIR, LR_DIR)


