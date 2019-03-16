# -*- coding:utf8 -*-
import argparse
from logging import log
__author__ = 'xt'

class Parse():
    def __init__(self):
        self.parse = argparse.ArgumentParser()

        self.parse.add_argument("--dir",type=str,default='./')
        self.parse.add_argument("--sz",type=int,default=224)
        self.parse.add_argument("--num_workers",type=int,default=4)
        self.parse.add_argument("--epoch",type=int,default=100)
        #region proposal parameter
        self.parse.add_argument("--proposal_min_size",type=int,default=250)
        self.parse.add_argument("--proposal_min_area",type=int,default=600)

        #processed data log
        self.parse.add_argument("--PrecessedDataInfo",type=str,default='data/processed/log.txt')

        #1.pre-train phase
        self.parse.add_argument("--P_num_classes",type=int,default=17)   #17 class
        self.parse.add_argument("--train_list",type=str,default=r'data/train_list.txt')
        self.parse.add_argument("--input_data_dir",type=str,default=r'data/17flowers/jpg')
        self.parse.add_argument("--pretrain_data_dir",type=str,default='data/processed/pretrain')    #input
        self.parse.add_argument("--Pretrain_model",type=str,default='model/alexnet-owt-4df8aa71.pth')
        self.parse.add_argument("--P_model",type=str,default='output/pretrain/pretrain_Alexnet.pth')  #output ,save model
        #para
        self.parse.add_argument("--pretrain_batch_size",type=int,default=128)
        self.parse.add_argument("--pretrain_lr",type=int,default=5e-5,choices=[2e-5,1e-5,1e-6])
        self.parse.add_argument("--pretrain_epoch",type=int,default=10)

        #2.finetune phase
        self.parse.add_argument("--finetune_threshold",type=float,default=0.2)
        self.parse.add_argument("--F_num_classes",type=int,default=3)   #2 class and 1 background
        self.parse.add_argument("--finetune_list",type=str,default=r'data/finetune_list.txt')
        self.parse.add_argument("--finetune_data_dir",type=str,default='data/processed/finetune')    #input
        self.parse.add_argument("--F_model",type=str,default='output/finetune/finetune.pth')  #output ,save model
        #para
        self.parse.add_argument("--finetune_batch_size",type=int,default=64)
        self.parse.add_argument("--finetune_lr",type=int,default=1e-4,choices=[1e-4,5e-4,1e-3])
        self.parse.add_argument("--finetune_epoch",type=int,default=10)

        #2.1 pre svm ,save finetune feature matrix 4096 * N
        self.parse.add_argument("--reg_threshold",type=int,default=0.3,choices=[0,0.3,0.5])
        self.parse.add_argument("-SVM_and_Reg_data_dir",type=str,default='data/processed/presvm')
        self.parse.add_argument("--Pre_SVM_model",type=str,default='output/pre_svm/pre_svm.pth')  #output ,save model
        #3.SVM and non-maximum suppression phase
        self.parse.add_argument("--output",type=str,default='output')
        self.parse.add_argument("--svm_model_dir",type=str,default=r'output/SVM_model')
        self.parse.add_argument("--S_num_class",type=int,default=3)
        self.parse.add_argument("--svm_batch_size",type=int,default=64)

        #4.regression
        self.parse.add_argument("--R_num_class",type=int,default=5)
        self.parse.add_argument("--R_batch_size",type=int,default=64)
        self.parse.add_argument("--R_model",type=str,default='output/reg/reg.pth')
        #para
        self.parse.add_argument("--regression_batch_size",type=int,default=256)
        self.parse.add_argument("--regression_lr",type=int,default=1e-4,choices=[1e-4,5e-4,1e-3])
        self.parse.add_argument("--regression_epoch",type=int,default=100)

        #5.test
        self.parse.add_argument("--test_example",type=str,default='data/test/image_1281.jpg')
        self.config = self.parse.parse_args()

        print(self.config)

    def __call__(self, *args, **kwargs):
        return self.parse.parse_args()
