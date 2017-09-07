import os
import glob
import cv2


def bicu_inter(lr_dir='../data/lr_imgs/', tag='face', bicu_dir='../data/bi_imgs/'):
	'''
	Generate Bicubic Interpolation of the low-resolution images.

	'''

	hr = (32, 32)
	lr = (8, 8)

	if not os.path.exists(bicu_dir + tag):
		os.mkdir(bicu_dir + tag)

	lr_list = glob.glob(lr_dir + tag + '/*/')
	lr_list.sort(key=lambda f: int(filter(str.isdigit, f)))

	# print lr_list

	for idx, lr in enumerate(lr_list):

		if not os.path.exists(bicu_dir + tag + '/' + str(idx)):
			os.mkdir(bicu_dir + tag + '/' + str(idx))

		print("Bicubic interpolation pictures in %d.gif"%(idx))
		p_list = glob.glob(lr + '*.png')
		p_list.sort(key=lambda f: int(filter(str.isdigit, f)))

		for p in p_list:
			im1 = cv2.imread(p)
			im2 = cv2.resize(im1, dsize=hr, interpolation=cv2.INTER_CUBIC)
			q = bicu_dir + p.split('/')[3] + '/' + p.split('/')[4] + '/' + p.split('/')[5]			
			cv2.imwrite(q, im2)





if __name__ == '__main__':

	LR_DIR='../data/lr_imgs/'
	TAG='face'
	BICU_DIR='../data/bi_imgs/'

	bicu_inter(LR_DIR, TAG, BICU_DIR)





	