# -*- coding:utf8 -*-

'''
surpervised pre-training 部分
数据集是dataset (ILSVRC 2012)
#alexnet 预训练数据集为ImageNet
'''
import torch
import torch.nn as nn
from torchvision.models import AlexNet, alexnet

__author__ = 'xutao'


class AlexNetModel(nn.Module):
    def __init__(self,config,is_pretrain=True,is_finetune=False,is_presvm=False,
                 is_model_save=True,is_output_save=False):
        super(AlexNetModel, self).__init__()
        self.config = config
        self.is_pretrain = is_pretrain
        self.is_finetune = is_finetune
        self.is_presvm = is_presvm
        self.is_model_save = is_model_save
        self.is_output_save = is_output_save

        if is_pretrain:
            encoder = alexnet(pretrained=False)
            self.pre_weight = config.Pretrain_model
            #只加载模型的参数
            encoder.load_state_dict(torch.load(self.pre_weight, map_location=lambda storage, loc: storage))
            self.num_classes = config.P_num_classes
            self.body = nn.Sequential(*list(encoder.children())[:-1])
            self.head = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Linear(4096, self.num_classes))
            # for layer in self.body:
            #     for p in layer.parameters():
            #         p.requires_grad=False
            self.model_save_path = config.P_model
        if is_finetune:
            if torch.cuda.is_available():
                encoder = torch.load(config.P_model)
            else:
                encoder = torch.load(config.P_model,map_location="cpu")
            self.num_classes = config.F_num_classes
            self.body = nn.Sequential(*list(encoder.children())[:-1])
            self.head = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Linear(4096, self.num_classes))
            for layer in self.body:
                for p in layer.parameters():
                    p.requires_grad=False

            self.model_save_path = config.F_model
        if is_presvm:  #train 4096 * num_class featrues for next phase:svm
            if torch.cuda.is_available():
                encoder = torch.load(self.config.F_model)
            else:
                encoder = torch.load(self.config.F_model,map_location="cpu")
            self.body = nn.Sequential(*list(encoder.children())[:-1])
            self.head = nn.Sequential(nn.Dropout(0.5),
                                      nn.Linear(in_features=9216,out_features=4096,bias=True),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(in_features=4096,out_features=4096))
            self.model_save_path = config.Pre_SVM_model

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    net = AlexNetModel()
