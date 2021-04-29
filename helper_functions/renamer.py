import os

root = "./dataset/mivia/split/neg/"
prefix = 'negVideo'
suffix = ''

all_files = [f for f in os.listdir(root)]
num_files = len(all_files)
len_num_files = len(str(num_files))

for i in range(num_files):
    full_name = [name, extension] = all_files[i].split('.')

    new_name = prefix + str(i).zfill(len_num_files) + suffix

    os.rename(root + all_files[i], root + '{}.{}'.format(new_name, extension))

    print('success', name)