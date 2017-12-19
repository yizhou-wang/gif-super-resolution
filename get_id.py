import glob
import scipy.io

list_ac = glob.glob('../../data/raw_gifs/action/*.gif')
list_al = glob.glob('../../data/raw_gifs/animal/*.gif')
list_an = glob.glob('../../data/raw_gifs/animation/*.gif')
list_ex = glob.glob('../../data/raw_gifs/expression/*.gif')
list_sc = glob.glob('../../data/raw_gifs/scene/*.gif')

# print list_ac

id_ac = [ int(x.split('/')[-1].split('.')[0]) for x in list_ac ]
id_ac.sort()
id_al = [ int(x.split('/')[-1].split('.')[0]) for x in list_al ]
id_al.sort()
id_an = [ int(x.split('/')[-1].split('.')[0]) for x in list_an ]
id_an.sort()
id_ex = [ int(x.split('/')[-1].split('.')[0]) for x in list_ex ]
id_ex.sort()
id_sc = [ int(x.split('/')[-1].split('.')[0]) for x in list_sc ]
id_sc.sort()

filename = '../../result/ids.mat'
scipy.io.savemat(filename, mdict={'id_ac': id_ac, 'id_al': id_al, 'id_an': id_an, 'id_ex': id_ex, 'id_sc': id_sc}) 

