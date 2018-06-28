'''
for FNN read dataset

using pytorch to rebuild it

reference to Caffe Code...

Liangzia,2018,4,27
'''

import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torchvision import transforms as trans
import string

class VIPeR(data.Dataset):
    def __init__(self,path,transforms=None,train=True,test=False,quyang=True):
        '''
        对标签进行处理
        '''

        self.test=test
        def getpath(path):
            path_new=[os.path.join(path,dir) for dir in os.listdir(path)]
            return path_new

        #load the path+name of each image
        path_a,path_b=getpath(path)
        imgs_a=[os.path.join(path_a,img) for img in os.listdir(path_a)]
        imgs_b=[os.path.join(path_b,img) for img in os.listdir(path_b)]

        # train: ./VIPeR/cam_a/101_45.bmp
        #sort:
        if self.test:
            #还没想好该怎么写
            pass
        else:
            imgs_a=sorted(imgs_a,key=lambda x : x.split('.')[-2].split('/')[-1].split('_')[-2])
            imgs_b=sorted(imgs_b,key=lambda x : x.split('.')[-2].split('/')[-1].split('_')[-2])

        imgs_number=len(imgs_a)

        #shuffle them
        #此处的打乱是总体打乱，每一个索引上双方是一致的
        np.random.seed(3933)
        imgs_a=np.random.permutation(imgs_a)
        np.random.seed(3933)
        imgs_b=np.random.permutation(imgs_b)

        #取样
        if quyang:
            if self.test:
                pass
            elif train:
                self.imgs_a=imgs_a[int(imgs_number*0.7):]
                self.imgs_b=imgs_b[int(imgs_number*0.7):]

        #进行正则化，也就是数据的处理：
        if transforms is None:
            #此处的均值和方差都是瞎编的，可自定义
            normalize=trans.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

        #下面进行简单的图像处理
        if self.test:
            pass
        else:
            #原图片大小 128*48pixel
            self.tranforms=trans.Compose([
                trans.Scale(50),#此处瞎写的比例
                trans.RandomCrop(size=48),
                #trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                normalize
            ])

    def __getitem__(self,index,camera=0):
        '''
        一次返回一张图片的数据
        :param self:
        :param index:
        :return:
        '''
        if camera==0:
            imgs_a_path=self.imgs_a[index]
            if self.test:
                pass
            else:
                label=imgs_a_path.split('.')[-2].split('/')[-1].split('_')[-2]
            data=Image.open(imgs_a_path)
            data=self.tranforms(data)
            return data ,label
        else:
            imgs_b_path = self.imgs_b[index]
            if self.test:
                pass
            else:
                label = imgs_b_path.split('.')[-2].split('/')[-1].split('_')[-2]
            data = Image.open(imgs_b_path)
            data = self.tranforms(data)
            return data, label

    def __len__(self):
        return len(self.imgs_a)



