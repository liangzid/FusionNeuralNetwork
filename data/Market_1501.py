'''
for FNN read dataset

using pytorch to rebuild it

reference to Caffe Code...

Liangzia,2018,4,27
'''
'''
new changes:==========

modify again on 5,22,2018

change the way of read

try to use cuda

the dataset now is turn to Market 1501
========================================
备注：
在正则化的过程中，需要的参数并没有进行调节，这一部分已经被注释掉了。当实际调试时要记住用啊。。。
'''

import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torchvision import transforms as trans
import string
import cv2


class Market_1501(data.Dataset):
    '''
    read the dataset Market-1501 and get the data and label
    这些是作者要求加上去的，可以不用看
    @inproceedings{zheng2015scalable,
    title={Scalable Person Re-identification: A Benchmark},
    author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
    booktitle={Computer Vision, IEEE International Conference on},
    year={2015}
        }
    '''
    def __init__(self,path,transforms=None,train=True,test=False,quyang=True):
        
        self.test=test
        self.train=train
        #path:
        # if train:
        # path: ../dataset/Market1501/bounding_box_train/
        # 0002_c1s1_000451_03.jpg
        # if test:
        #  path: ../dataset/Market1501/bounding_box_test/
        # 0000_c1s1_000151_06.jpg 
        imgs=[os.path.join(path,img) for img in os.listdir(path)]
        
        #sorted:
        if self.train:
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1].split('_')[0]))
        if self.test:
            imgs=sorted(imgs,key=lambda x:int(x.split('.')[-2].split('/')[-1].split('_')[0]))
        imgs_num=len(imgs)

        #shuffle imgs
        np.random.seed(933)
        imgs=np.random.permutation(imgs)

        if self.test:
            self.imgs=imgs
        elif self.train:
            self.imgs=imgs[:int(0.7*imgs_num)]
        
        if transforms is None:
            normalize=trans.Normalize(mean=[0.48,0.407,0.456],
            std=[0.229,0.225,0.224])
        '''
            if self.test or not self.train:
                self.transforms=trans.Compose([
                    trans.Scale(224),
                    trans.CenterCrop(224),
                    trans.ToTensor,
                    normalize
                ])
            else:
                self.transforms = trans.Compose([
                    trans.Scale(256),
                    trans.RandomSizedCrop(224),
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                    normalize,
                    ]) 
        '''
    def __getitem__(self,index):
        '''
        得到指定图片的数据
        '''
        img_path=self.imgs[index]
        label = int(self.imgs[index].split('.')[-2].split('/')[-1].split('_')[0])
        data=Image.open(img_path)

        #data=cv2.imread(img_path)
      #  data=self.transforms(data)

        return data,label

    def __len__(self):
        return len(self.imgs)









