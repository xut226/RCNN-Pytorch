import codecs
import os
import cv2
import numpy as np
import logging
from selectivesearch import selectivesearch
import shutil
import torch
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

__author__ = 'xutao'

def load_image(image,reshape=False,shape=(224,224,3)):
    img = cv2.imread(image)
    if reshape:
        img = cv2.resize(img,shape)
    return img
def clip_pic(image,rect):
    '''
    clip the rect part
    :param image:
    :return:
    '''
    x,y,w,h = rect[0],rect[1],rect[2],rect[3]
    return image[y:y+w,x:x+w,:],(x, y, x+w, y+h, w, h)

def resize_image(img,shape,mode=cv2.INTER_CUBIC):
    return cv2.resize(img,shape,interpolation=mode)

def create_proposals(image,config):
    img_lbl,regions = selectivesearch.selective_search(image,scale=100,sigma=0.9,min_size=10)
    candidates = set()
    images = []

    vertices = []
    for r in regions:
        # remove reduplicative region
        if r['rect'] in candidates:
            continue
        if r['size'] < config.proposal_min_size:
            continue
        if (r['rect'][2] * r['rect'][3]) < config.proposal_min_area:
            continue
        proposal_img, proposal_vertice = clip_pic(image, r['rect'])
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize_image(proposal_img, (config.sz,config.sz))
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])

    return images,vertices


def IOU(box_a,box_b):
    is_insersect = False
    xa1 = box_a[0]
    ya1 = box_a[1]
    xa2 = box_a[2]
    ya2 = box_a[3]

    xb1 = box_b[0]
    yb1 = box_b[1]
    xb2 = box_b[2]
    yb2 = box_b[3]

    area_a = (xa2-xa1) * (ya2 - ya1)
    area_b = (xb2-xb1) * (yb2 - yb1)

    x1 = max(xa1,xb1)
    y1 = max(ya1,yb1)
    x2 = min(xa2,xb2)
    y2 = min(ya2,yb2)

    if x2 - x1 > 0 and y2 - y1 > 0:
        is_insersect = True
        area = (x2 - x1) * (y2 - y1)
        iou = area / (area_a + area_b - area)
        return is_insersect,iou
    else:
        return is_insersect,0

class PretrainDataset(Dataset):
    def __init__(self,config):
        super(PretrainDataset, self).__init__()
        self.config = config
        self.bs = config.pretrain_batch_size
        self.nw = config.num_workers
        self.tfms = self.pretrain_transform()
        if not os.path.exists(self.config.pretrain_data_dir):
            os.mkdir(self.config.pretrain_data_dir)
        a = os.listdir(self.config.pretrain_data_dir)
        if len(os.listdir(self.config.pretrain_data_dir)) == 0:
            self.make_Dataset()
        self.Dataset = ImageFolder(config.pretrain_data_dir,transform=self.tfms)

    def move_file(self,src_path,dst_path,file,label):
        try:
            f_src = os.path.join(src_path,file)
            label_path = os.path.join(dst_path,label)
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            f_dst = os.path.join(label_path,file)
            shutil.move(f_src,f_dst)
        except:
            print('move error!')
        finally:
            # print("save %s into classify dir: %s." % (file,label))
            pass
    def make_Dataset(self):
        with codecs.open(self.config.train_list,'r','utf-8') as f:
            lines = f.readlines()
            print("-------------make pretraining dataset---------------\n")
            for idx, line in tqdm(enumerate(lines), total=len(lines)):
                context = line.strip().split(' ')
                name = context[0].split('/')[-1]
                label = context[1]
                src_path = self.config.input_data_dir
                dst_path = self.config.pretrain_data_dir
                self.move_file(src_path, dst_path, name, label)

    def pretrain_transform(self):
        tfms = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.config.sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        return tfms
    def __call__(self, *args, **kwargs):
        return DataLoader(self.Dataset,batch_size=self.bs,shuffle=True,num_workers=self.nw)


class FinetuneDataset(Dataset):
    def __init__(self,config,is_save=False,is_svm=False):
        super(FinetuneDataset, self).__init__()
        self.config = config
        self.bs = config.finetune_batch_size
        self.nw = config.num_workers
        self.tfms = self.finetune_transform()
        self.is_save = is_save
        self.is_svm = is_svm
        self.finetune_threshold = config.finetune_threshold
        self.reg_threshold = config.reg_threshold
        self.finetune_data_dir = config.finetune_data_dir
        self.SVM_and_Reg_data_dir = config.SVM_and_Reg_data_dir
        if len(os.listdir(self.finetune_data_dir))==0:
            self.make_finetuneData()
        self.Dataset = ImageFolder(config.finetune_data_dir, transform=self.tfms)

    def finetune_transform(self):
        tfms = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.config.sz),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        return tfms

    def make_finetuneData(self):
        print("-------------make finetune dataset------------")
        with codecs.open(self.config.finetune_list,'r','utf-8') as f:
            lines = f.readlines()
            image_idx = 0
            for num,line in tqdm(enumerate(lines),total=len(lines)):
                context = line.strip().split(' ')
                image_path = context[0]
                ref_rect = context[2].split(',')
                ground_truth = [int(i) for i in ref_rect]
                img = cv2.imread(os.path.join('data',image_path))
                if img is not None:
                    imgs_lbl,vertices = create_proposals(img,self.config)
                else:
                    return
                for img_float,vertice in zip(imgs_lbl,vertices):
                    is_insersect,iou_val = IOU(ground_truth, vertice)
                    if iou_val < self.finetune_threshold or not is_insersect:
                        label  = 0  #
                    else:
                        label = context[1]
                    label_dir_name = os.path.join(self.finetune_data_dir,str(label))
                    if not os.path.exists(label_dir_name):
                        os.mkdir(label_dir_name)
                    image_name = 'image_'+str(image_idx)+'.jpg'
                    if not os.path.exists(os.path.join(label_dir_name,image_name)):
                        cv2.imwrite(os.path.join(label_dir_name,image_name),img_float)
                    image_idx += 1
    # output of the trained finetune model for svm'inputs
    def make_svmData(self,net):
        print("-------------make svm dataset------------")
        with codecs.open(self.config.finetune_list,'r','utf-8') as f:
            lines = f.readlines()
            image_idx = 0
            for num,line in tqdm(enumerate(lines),total=len(lines)):
                images = []
                labels = []
                labels_bbox = []
                context = line.strip().split(' ')
                image_path = context[0]
                ref_rect = context[2].split(',')
                ground_truth = [int(i) for i in ref_rect]
                img = cv2.imread(os.path.join('data',image_path))
                if img is not None:
                    imgs_lbl,vertices = create_proposals(img,self.config)
                else:
                    return
                for img_float,vertice in zip(imgs_lbl,vertices):
                    if torch.cuda.is_available():
                        img = torch.from_numpy(img_float)   #224*224*3
                        img = img.permute(2,1,0)          #3*224*224
                        img = img.unsqueeze(0)              #1*3*224*224
                        img = Variable(img).cuda()
                    else:
                        img = torch.from_numpy(img_float)
                        img = img.permute(2,1,0)          #3*224*224
                        img = img.unsqueeze(0)
                    feature = net(img)
                    images.append(feature[0])

                    is_insersect,iou_val = IOU(ground_truth, vertice)
                    px = float(vertice[0]) + float( vertice[2] / 2.0)
                    py = float(vertice[1]) + float( vertice[3] / 2.0)
                    ph = float(vertice[3])
                    pw = float(vertice[2])

                    gx = float(ref_rect[0])
                    gy = float(ref_rect[1])
                    gw = float(ref_rect[2])
                    gh = float(ref_rect[3])
                    index = int(context[1])
                    if iou_val < self.finetune_threshold or not is_insersect:
                        label  = 0  #
                    else:
                        label = index
                    labels.append(label)

                    label_reg = np.zeros(5)
                    if pw != 0 and ph !=0:
                        label_reg[1:5] = [(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]
                    elif pw == 0 and ph != 0:
                        label_reg[1:5] = [(gx - px) / 1e-6, (gy - py) / ph, np.log(gw / 1e-6), np.log(gh / ph)]
                    elif pw != 0 and ph == 0:
                        label_reg[1:5] = [(gx - px) / pw, (gy - py) / 1e-6, np.log(gw / pw), np.log(gh / 1e-6)]

                    if iou_val <self.reg_threshold:
                        label_reg[0] = 0
                    else:
                        label_reg[0] = 1
                    labels_bbox.append(label_reg)
                if not os.path.exists(os.path.join(self.SVM_and_Reg_data_dir, str(context[1]))):
                    os.makedirs(os.path.join(self.SVM_and_Reg_data_dir, str(context[1])))

                np.save((os.path.join(self.SVM_and_Reg_data_dir, str(context[1]), context[0].split('/')[-1].split('.')[0].strip())
                                                    + '_data.npy'),[images, labels, labels_bbox])

    def __getitem__(self, item):
        pass


class SVM_Reg_Dataset(Dataset):
    def __init__(self,config,is_svm = False,is_reg = False):
        self.config = config
        self.is_svm = is_svm
        self.is_reg = is_reg
        self.input_dir = config.SVM_and_Reg_data_dir
        self.images = []
        self.labels = []
        self.label_list = os.listdir(self.input_dir)
        self.reg_data = []
        self.svm_data = []
        self.SVM_data_dic = {}
        if is_reg:
            self.epoch = config.regression_epoch
            self.cursor = 0
            self.bs = config.regression_batch_size
        self.make_svmAndregData()
    def make_svmAndregData(self):
        for label in self.label_list:
            data_list = os.listdir(os.path.join(self.input_dir,str(label)))
            for ind,d in enumerate(data_list):
                path = os.path.join(self.input_dir,label,d)
                i,l,k = np.load(path)
                for index in range(len(i)):
                    if self.is_svm:
                        self.svm_data.append([i[index],l[index]])
                    if self.is_reg:
                        self.reg_data.append([i[index],k[index]])
            if self.is_svm:
                self.SVM_data_dic[label] = self.svm_data

    def get_svm_data(self,data_dir):
        for index in range(len(self.SVM_data_dic[data_dir])):
            self.images.append(self.SVM_data_dic[data_dir][index][0].data.cpu().numpy())
            self.labels.append(self.SVM_data_dic[data_dir][index][1])
        return self.images,self.labels

    def get_reg_data(self):
        images = np.zeros((self.config.R_batch_size,4096))
        labels = np.zeros((self.config.R_batch_size,self.config.R_num_class))
        count = 0
        while (count < self.config.R_batch_size):
            images[count] = self.reg_data[self.cursor][0]
            labels[count] = self.reg_data[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.reg_data):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.reg_data)
        return images,labels

    def __getitem__(self, item):
        image = self.reg_data[item][0]
        label = self.reg_data[item][1]
        return self.reg_data[item][0],self.reg_data[item][1]    #image,label

    def __len__(self):
        if self.is_svm:
            return len(self.svm_data)
        else:
            return len(self.reg_data)


if __name__ == '__main__':
    img = load_image(image=r'E:\xt\program\Python\pytorchExercise\RCNN-Pytorch\data\\2.jpg')
    # create_proposals(img)





