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



def img2gif(png_dir='../data/rec_imgs/', tag='face', gif_dir='../data/rec_gifs/'):
	"""
	Converts png frames to gif with each png containing the full image data

	png_dir - path to folder where generated gifs are stored
	gif_list - path of all gifs in gif_dir

	"""

	png_list = glob.glob(png_dir + tag)
	png_list.sort(key=lambda f: int(filter(str.isdigit, f)))

	if os.path.exists(png_dir):
		boo = raw_input("recovered gifs already exist, overwrite? : y/n -> ")
		if boo is 'y':
			shutil.rmtree(png_dir)
			os.mkdir(png_dir)
	else:
		os.mkdir(png_dir)

	for i in range(len(gif_dir)):
		print("Recovering %d.gif"%(i))
		png_path=split_path+str(i)+"/"
		pngList=glob.glob(png_path+"*.jpg")
		frames = len(pngList)
		original_gif=Image.open(gif_dir[i])
		delay = original_gif.info['duration']/10.0
		# May need to modify the paths.
		call(["convert","-delay",str(delay),"-loop","0",png_path+"/000*.jpg", png_dir+"/"+str(i)+".gif"])

	print "Done with recovering."




if __name__ == '__main__':

	GIF_DIR = '../data/raw_gifs/'
	TAG = 'face'
	PNG_DIR = '../data/raw_imgs/'

	gif2img(GIF_DIR, TAG, PNG_DIR)

	
