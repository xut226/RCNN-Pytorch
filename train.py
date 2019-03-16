import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from DataProcess.dataload import PretrainDataset, FinetuneDataset, SVM_Reg_Dataset
from config import Parse
from model.Classify_SVM import SVM
from model.RegressionModel import RegressionModel, RegLoss
from model.pretrain_finetune import AlexNetModel

__author__ = 'xutao'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Solver():
    def __init__(self,config,model,dataset,
                 is_pretrain = False,
                 is_finetune = False,
                 is_presvm = False,
                 is_svm = False,
                 is_reg = False):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.is_pretrain = is_pretrain
        self.is_finetune = is_finetune
        self.is_presvm = is_presvm
        self.is_svm = is_svm
        self.is_reg = is_reg

        if self.is_reg:
            self.dataloader = DataLoader(self.dataset,batch_size=dataset.bs,shuffle=True)
        else:
            self.dataloader = DataLoader(self.dataset.Dataset,batch_size=dataset.bs,shuffle=True)
        if self.is_pretrain:
            self.epoch = config.pretrain_epoch
            self.load_model = None
            self.model_save_path = self.config.P_model
            self.cirterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(model.parameters(),lr=self.config.pretrain_lr)

        if self.is_finetune:
            self.epoch = config.finetune_epoch
            self.load_model = self.config.P_model
            self.model_save_path = self.config.F_model
            self.cirterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.finetune_lr)
        if self.is_presvm:
            self.load_model = self.config.F_model
            self.model_save_path = self.config.Pre_SVM_model
        if self.is_reg:
            self.epoch = config.regression_epoch
            self.model_save_path = config.R_model
            self.cirterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.regression_lr)

    def train(self):
        total_loss=0.0
        model = self.model.to(device)
        for epoch in range(self.epoch):
            length = len(self.dataloader)
            if not self.is_reg:
                self.adjust_learning_rate(self.config.pretrain_lr,self.optimizer,epoch)
            for idx,data in tqdm(enumerate(self.dataloader),total=len(self.dataloader)):
                image,label = data[0],data[1]
                if torch.cuda.is_available():
                    image,label = Variable(image).cuda(),Variable(label).cuda()

                output = model(image)
                if self.is_reg:
                    output = output.double()
                loss = self.cirterion(output,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (idx+1) % length == 0:
                    print('[epoch:%d ,step:%5d] loss: %.3f' % (epoch + 1, idx + 1, total_loss / length))
                    total_loss = 0.0
        if self.model.is_model_save:
            torch.save(self.model,self.model_save_path)    #


    def predict(self):
        if torch.cuda.is_available():
            model = self.model.cuda()
        else:
            model = self.model

        if self.is_presvm:
            # self.dataset.make_svmData(model)
            if self.model.is_model_save:
                torch.save(self.model,self.model_save_path)

        if self.is_finetune:
            pass
        if self.is_svm:
            pass
        if self.is_reg:
            pass


    def adjust_learning_rate(self,lr,optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
#supervised pre-training

def get_solver(config):

    #pretrain
    P_dataset = PretrainDataset(config)
    P_model = AlexNetModel(config,is_pretrain=True,is_finetune=False,is_presvm=False)
    P_solver = Solver(config,P_model,P_dataset,is_pretrain=True)
    P_solver.train()

    #Domain-specific fine-tuning
    F_dataset = FinetuneDataset(config)
    F_model = AlexNetModel(config, is_pretrain=False, is_finetune=True, is_presvm=False)
    F_solver = Solver(config, F_model, F_dataset, is_finetune=True)
    F_solver.train()

    # save outputdata for svm and reg
    F_model = AlexNetModel(config, is_pretrain=False, is_finetune=False, is_presvm=True)
    F_solver = Solver(config, F_model, F_dataset,is_presvm=True)
    F_solver.predict()

    #object category classifies,svm
    S_data = SVM_Reg_Dataset(config,is_svm=True)
    S_model = SVM(config,S_data)
    S_model.train()

    #regression
    R_data = SVM_Reg_Dataset(config,is_reg=True)
    R_model = RegressionModel(config)
    R_solver = Solver(config,R_model,R_data,is_reg=True)
    R_solver.train()



if __name__ == '__main__':

    parse = Parse()
    config = parse.config

    get_solver(config)
