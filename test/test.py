# -*- coding: utf-8 -*-
##这一部分很多都和train.py相同,因此加以注释的地方不多,请先阅读训练脚本.
from __future__ import print_function, division#同train.py,可以使用print()函数,除法变为精确除法

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from Model.deep_feature.model import fusion_net

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Test!')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../../Market1501/pytorch',type=str,
                    help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save Model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')

opt = parser.parse_args()


#判断GPU型号,采用第一个
str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
#预处理
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms)
                  for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16)
               for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load Model
#---------------------------
def load_network(network):
    save_path = os.path.join('../model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained Model.
#
def fliplr(img): #flip的意思是轻弹,快速翻转
    '''flip horizontal''' #水平翻转???
    #torch.arange(start,end,step=1,out=None)函数可以返回一个一维张量,起始为start,终于end,步长为step.
    #.long()方法用来将Tensor类型转化为Long类型
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W #数目*频道*高度*宽度??
    img_flip = img.index_select(3,inv_idx)#这里应该就是实现了所谓的翻转了,仍然是针对第三个维度
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size() #What???我怎么觉得这里似乎有问题,你确定那个n==1吗?还是说这里的img是一个
                                #id文件夹下的图像数据组成的张量
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_() #其实这里的1024与2048为什么我是搞不清楚的
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have four parts #对的,要被切成6块
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f     #相当于是翻转的结论与不进行翻转的结论相加
        # norm feature
        if opt.PCB:
            # feature size (n,2048,4)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)#这里是返回了张量在指定维度上的p范数,并保存
                                                            #那个维度
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            ff = ff.div(fnorm.expand_as(ff)) ## 不理解...
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained Model
print('-------test-----------')

model_structure = fusion_net(751)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
model.model.fc = nn.Sequential()     #为什么要移除啊! 为了提取特征?
model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()                     #??
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,
          'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,
          'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)
