"""Create image-list file
Example:
python tools/create_img_lists.py 
"""
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dataset", dest="dataset",  
                  help="dataset path")

parser.add_option("--outfile", dest="outfile",  
                  help="outfile path")
(options, args) = parser.parse_args()

f = open(options.outfile, 'w')
dataset_basepath = options.dataset
for p1 in os.listdir(dataset_basepath):
	l2 = os.listdir(dataset_basepath + '/' + p1)
	l2.sort(key=lambda f: int(filter(str.isdigit, f)))
	l2 = l2[:1000]
	for p2 in l2:
		l3 = os.listdir(dataset_basepath + '/' + p1 + '/' + p2)
		l3.sort(key=lambda f: int(filter(str.isdigit, f)))
		for p3 in l3:
  			image = os.path.abspath(dataset_basepath + '/' + p1 + '/' + p2 + '/' + p3)
  			f.write(image + '\n')
f.close()
