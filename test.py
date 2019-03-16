# -*- coding:utf8 -*-
import cv2
import joblib
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from DataProcess.dataload import create_proposals
from config import Parse
from model.RegressionModel import RegressionModel
from model.pretrain_finetune import AlexNetModel
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RCNN():
    def __init__(self,config):
        self.config = config
        self.S_model = []
        self.load_model()


    def load_model(self):
        self.F_model = torch.load(self.config.Pre_SVM_model,map_location=device)

        a = [file for file in os.listdir(self.config.svm_model_dir)]
        self.S_model = [joblib.load(os.path.join(self.config.svm_model_dir,file))
                                 for file in os.listdir(self.config.svm_model_dir)]
        a = self.config.R_model
        self.R_model = torch.load(self.config.R_model,map_location=device)
    def test(self,image):
        imgs_lbl,vertices = create_proposals(image,config)

        finetune_features = []
        for img_float in imgs_lbl:
            if torch.cuda.is_available():
                img = torch.from_numpy(img_float)   #224*224*3
                img = img.permute(2,1,0)          #3*224*224
                img = img.unsqueeze(0)              #1*3*224*224
                img = Variable(img).cuda()
            else:
                img = torch.from_numpy(img_float)
                img = img.permute(2,1,0)          #3*224*224
                img = img.unsqueeze(0)
            feature = self.F_model(img)
            finetune_features.append(feature)

        results = []
        results_old = []
        results_label = []
        count = 0
        for f in finetune_features:
            for svm in self.S_model:
                pred = svm.predict(f.data.cpu().numpy())
                # not background
                # f = torch.from_numpy(f)
                if pred[0] != 0:
                    results_old.append(vertices[count])
                    R_result = self.R_model(f)
                    if R_result[0][0] > 0.5:
                        px, py, pw, ph = vertices[count][0], vertices[count][1], vertices[count][2], vertices[count][3]
                        old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0

                        x_ping, y_ping, w_suo, h_suo = R_result[0][1], \
                                                       R_result[0][2], \
                                                       R_result[0][3], \
                                                       R_result[0][4]
                        new__center_x = x_ping * pw + old_center_x
                        new__center_y = y_ping * ph + old_center_y
                        new_w = pw * np.exp(w_suo.data.cpu().numpy())
                        new_h = ph * np.exp(h_suo.data.cpu().numpy())
                        a = new__center_x.data.cpu().numpy()
                        b = new__center_y.data.cpu().numpy()
                        a = float(a)
                        b = float(b)
                        new_verts = [a, b, new_w, new_h]
                        results.append(new_verts)
                        results_label.append(pred[0])
                        print('find proposal, label:',pred[0],"rect:",new_verts)
            count += 1

    def NMS(self):
        pass

if __name__ == '__main__':
    parse = Parse()
    config = parse.config
    image = cv2.imread(config.test_example)
    rcnn = RCNN(config=config)
    rcnn.test(image)