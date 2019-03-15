# RCNN-Pytorch
This is an experimental Pytorcn implementation of RCNN - a convnet for object detection with a region proposal network. For details about R-CNN please refer to the paper:Rich feature hierarchies for accurate object detection and semantic segmentation.  
## Requirment   
1.pytorch 1.0.0  
2.opencv3-python 3.4.4.19  
## Training Model  
1.Download the dataset  
17flowers,http://www.robots.ox.ac.uk/~vgg/data/flowers/17/  
2.make training data format  
python make_DataSet.py  
3.train  
python train.py  
4.test  
python test.py  

## Reference
blog:(http://www.cnblogs.com/edwardbi/p/5647522.html)  
code:(https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN)
   
  
