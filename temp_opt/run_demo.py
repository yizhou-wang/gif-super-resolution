import os

gif_dir = '../../data/raw_gifs/'
tag = 'yizhou'

number_list = []
for file in os.listdir(gif_dir + tag + '/'):
    if file.endswith(".gif"):
    	number_list.append(file.translate(None, '.gif'))

number_list.sort(key=lambda f: int(filter(str.isdigit, f)))
print number_list

for number in number_list:
	print '*** Processing ' + number + '.gif ***'
	os.system("python main.py -n " + number)
        
