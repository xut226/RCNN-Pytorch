#-*- coding:utf8 -*-
import os
import joblib
from sklearn import svm

from sklearn.svm import SVC

#为每个类单独训练svm，输入n×4096维数据（n张图片，4096维是CNN训练得来的extracted features ）
class SVM():
    def __init__(self,config,data):
        self.data = data
        self.config = config
        self.output = config.output
        self.SVM_and_Reg_data_dir = config.SVM_and_Reg_data_dir
    def train(self):
        svms = []
        data_dirs = os.listdir(self.SVM_and_Reg_data_dir)
        for data_dir in data_dirs:
            images, labels = self.data.get_svm_data(data_dir)
            clf = svm.LinearSVC()
            clf.fit(images, labels)
            svms.append(clf)
            SVM_model_path = os.path.join(self.output, 'SVM_model')
            if not os.path.exists(SVM_model_path):
                os.makedirs(SVM_model_path)
            joblib.dump(clf, os.path.join(SVM_model_path,  str(data_dir)+ '_svm.pkl'))

