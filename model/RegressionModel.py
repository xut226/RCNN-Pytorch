# -*- coding:utf8 -*-

import torch
import torch.nn as nn
#
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class RegressionModel(nn.Module):
    def __init__(self,config,is_model_save = True):
        super(RegressionModel, self).__init__()
        self.config = config
        self.output_classes = config.R_num_class
        self.is_model_save = is_model_save

        # self.model = nn.Sequential(
        #     nn.Linear(in_features=4096,out_features=4096),
        #     nn.Sigmoid(),
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=4096,out_features=self.output_classes)
        # )
        self.fc1 = nn.Linear(in_features=4096,out_features=4096)
        self.active = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=4096,out_features=self.output_classes)
        # for layer in self.model.modules():
        #     if isinstance(layer,nn.Linear):
        #         param_shape = layer.weight
        #         layer.weight.data  = torch.from_numpy(np.random.normal(0,0.5,shape=param_shape))
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.active(x1)
        res = self.fc2(x2)
        return res

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self,pred,target):
        # target = target.float()
        pred = pred.double()
        no_object_loss = torch.pow( torch.pow( (1 - target[:,0]) * pred[:,0] ,2 ),0.25).mean()
        is_object_loss = torch.pow( torch.pow( target[:,0] * (pred[:,0] - 1),2),0.25 ).mean()
        c =pred[:,1:5]
        d = target[:,1:5]
        a = torch.pow(torch.pow((target[:,1:5] - pred[:,1:5]),2),0.25)
        b = torch.sum(a,dim=1)
        reg_loss = torch.mean(target[:,0]) * torch.pow(torch.pow((target[:,1:5] - pred[:,1:5]),2),0.25).sum()
        return no_object_loss + is_object_loss + reg_loss
