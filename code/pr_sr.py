import os

os.system('python prsr/tools/create_img_lists.py --dataset=../data/hr_imgs --outfile=../data/train.txt')
os.system('python prsr/tools/train.py --device_id=0')