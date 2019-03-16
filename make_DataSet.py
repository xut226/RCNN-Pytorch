import codecs
import os
import shutil

from tqdm import tqdm

__author__ = 'xt'

images = []
labels = []

def move_file(src_path,dst_path,file,label):
    try:
        f_src = os.path.join(src_path,file)
        label_path = os.path.join(dst_path,label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        f_dst = os.path.join(label_path,file)
        shutil.move(f_src,f_dst)

    except:
        print('move error')
    finally:
        # print("save %s into classify dir: %s." % (file,label))
        pass


with codecs.open(r'data/train_list.txt','r','utf-8') as f:
    lines = f.readlines()
    for idx,line in tqdm(enumerate(lines),total=len(lines)):
        context = line.strip().split(' ')
        name = context[0].split('/')[-1]
        label = context[1]
        images.append(name)
        labels.append(label)
        src_path = r'data/17flowers/jpg'
        dst_path = r'data/17flowers'
        move_file(src_path,dst_path,name,label)
