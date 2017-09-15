import os
import glob
from subprocess import call


def gif2img(gif_dir='../data/raw_gifs/', tag='face', png_dir='../data/raw_imgs/'):
	'''
	Split GIF into images.

		The images in each GIF are stored in a folder.
		The names of images are always 6 digits, like: 000001.png.
		
		e.g. GIF named 001.gif, the images are stored in a folder named 001. 

	'''

	if not os.path.exists(png_dir):
		os.mkdir(png_dir)

	if not os.path.exists(png_dir + tag):
		os.mkdir(png_dir + tag)

	gif_list = glob.glob(gif_dir + tag + '/' + '*.gif')
	gif_list.sort(key=lambda f: int(filter(str.isdigit, f)))

	# print gif_list

	for idx, gif in enumerate(gif_list):

		if not os.path.exists(png_dir + tag + '/' + str(idx)):
			os.mkdir(png_dir + tag + '/' + str(idx))

		print("Splitting %d.gif with tag '%s'"%(idx,tag))

		print 'GIF_DIR: ' + gif_list[idx]
		print 'PNG_DIR: ' + png_dir + tag + '/' + str(idx) + '/'

		call(["convert", "-coalesce", gif_list[idx], png_dir + tag + "/" + str(idx) + "/%06d.png"])

	print "Done with splitting."



def img2gif(
	png_dir='../data/rec_imgs/', 
	gif_dir='../data/rec_gifs/', 
	gt_dir='../data/hr_imgs/', 
	bi_dir='../data/bi_imgs/', 
	tag='face', train_size=1000):
	"""
	Converts png frames to gif with each png containing the full image data

	png_dir - path to folder where generated gifs are stored
	gif_list - path of all gifs in gif_dir

	"""

	gt_dir = gt_dir + tag + '/'
	bi_dir = bi_dir + tag + '/'
	gif_list = glob.glob(png_dir)
	gif_list.sort(key=lambda f: int(filter(str.isdigit, f)))

	if not os.path.exists(gif_dir):
		os.mkdir(gif_dir)

	print("Recovering GIF ...")
	# png_path=split_path+str(i)+"/"
	# plist = glob.glob(png_path + "*.png")
	# frames = len(plist)
	# original_gif = Image.open(gif_dir[i])
	# delay = original_gif.info['duration']/10.0
	# May need to modify the paths.
	for idx, gif_name in enumerate(gif_list):
		call(["convert", "-delay", str(0.2), "-loop", "0", gt_dir + str(idx+train_size) + "/*.png", gif_dir + "/" + str(idx+train_size) + "_gt.gif"])
		call(["convert", "-delay", str(0.2), "-loop", "0", bi_dir + str(idx+train_size) + "/*.png", gif_dir + "/" + str(idx+train_size) + "_bi.gif"])
		call(["convert", "-delay", str(0.2), "-loop", "0", gif_name + "/*.png", gif_dir + str(idx+train_size) + "_rec.gif"])

	print "Done with recovering."




if __name__ == '__main__':

	GIF_DIR = '../data/raw_gifs/'
	TAG = 'face'
	PNG_DIR = '../data/raw_imgs/'

	gif2img(GIF_DIR, TAG, PNG_DIR)

	
