import os

print 'Creating image lists ...'
os.system('python prsr/tools/create_img_lists.py --dataset=../data/hr_imgs --trainfile=../data/train.txt --testfile=../data/test.txt')
print 'Image list created!'
print 'Training ...'
os.system('python prsr/tools/train.py --device_id=0')
print 'Testing ...'
# os.system('python prsr/tools/test.py --device_id=0')