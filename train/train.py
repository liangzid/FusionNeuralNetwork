# -*- coding: utf-8 -*-

from __future__ import print_function, division  # print_function导入后会使得可以在python2里面使用print
import argparse  # 命令行与参数解析                 #函数，而division导入后会使得除法变为精确除法，即两个整数
import torch                                     # 相除得到的可以使小数（这在C里面是不可能的）
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler  # 提供了几种方法来根据epoches的数量调整学习率
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from Model.deep_feature.model import fusion_net
import json

######################################################################
# Options
# --------
#for example: python3 train.py --FNN --gpu_ids 0,1,2,3 --name 'liangzizuishaui' --train_all --batchsize 2 --color_jitter --which_epoch last
parser = argparse.ArgumentParser(description='Training')  # 创建一个解析器并告诉他们会有什么对象在内。
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--read_name', default='ft_ResNet50', type=str, help='input Model name')
parser.add_argument('--name', default='FusionNeuralNetwork', type=str, help='output Model name')
parser.add_argument('--data_dir', default='../../Market1501/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
# color jitter 颜色抖动
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
# batchsize的尺寸
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
# 是否使用fusion_net
parser.add_argument('--FNN', action='store_true', help='use FusionNeuralNetwork')
#选择需要加载的模型
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
read_name=opt.read_name
#设置GPU,目前仅支持单GPU实际情况
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)  # 这一步取整感觉没有什么意思啊，可能是为了安全？或者说保证类型是整形？
    if gid >= 0:
        gpu_ids.append(gid)  # 这一步主要是由于如果id的值为-1，代表纯粹使用cpu

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])  # 这一步就是设置了所使用的GPU吧！
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------

##这里是进行的对图像的预处理操作
# 训练集的预处理方式
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
# 验证集的预处理方式
transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


# 如果使用颜色抖动
if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)

# 相当于一个字典，‘train’映射到训练，‘val’映射到验证
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}

# 对于不同类别的文件放在不同的文件夹，使用datasets.ImageFolder。这样就可以实现读取文件夹，并且对文件夹里面的东西
# 实现预处理。然后将处理后的结果存入image_datasets字典中。
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])
# 使用DataLoader对进行预处理完毕的图像进行多线程读取
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=16)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# iter()是对可迭代对象产生一个迭代器，next（）是对迭代器读取其结果。只是第一个？
inputs, classes = next(iter(dataloaders['train']))

######################################################################
# Training the Model
# ------------------
#
# Now, let's write a general function to train a Model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best Model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []



######################################################################
# Finetuning the convnet #调整神经网络
# ----------------------
#
# Load a pretrainied Model and reset final fully connected layer.#加载提前训练好的模型并且重新设置最
# 终的全连接层
#

if opt.FNN:
    model = fusion_net(len(class_names))

print(model)

#=============================================================================================
#此处尝试对model初始化,由于储存的是cpu格式,所以先加载,再cuda
read_path = os.path.join('../model', read_name, 'net_%s.pth' % opt.which_epoch)

model.load_state_dict(torch.load(read_path).state_dict(),strict=False) #使得对应不上的参数被抛弃
#=============================================================================================

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，因为最后一层是softmax

##各种设置,下面是重灾区

# list()函数用于将元组转化为列表；map(func,iter)函数用于将可迭代对象iter中的每一个元素都代入函数func中得到
# 输出的列表或元组；id(a)函数用于返回自变量a的地址
ignored_params = list(map(id, model.model.fc.parameters())) + \
                 list(map(id, model.classifier.parameters()))  # 记录全连接和分类器的参数地址
# filter(func，iter)函数的意思是将可迭代对象iter中的元素代入函数func，将返回值是true的所有元素作为一个列表
# 输出。下面的那行代码的意思就是：如若模型参数不是ignored_params,那就是base_params。
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

# 下面实在定义优化方法.使用的不是SGD,而是动量优化(nesterov算法),对与conv_para和fc_para使用了两个不同的学习
# 率
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.0001},
    {'params': model.model.fc.parameters(), 'lr': 0.0001},
    {'params': model.classifier.parameters(), 'lr': 0.05}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
# 使用了lr_scheduler策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#

# 建立存储模型的地址
dir_name = os.path.join('../model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# save opts
# 这里是用来保存了所有的使用过的命令行选项.vars(opt)函数可以以字典的形式返回opt的所有属性,而dump()函数用来将所有
# 的非字符串形式的数据保存在json中.indent是缩进的意思.

with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase  #每一个周期都有训练和验证的环节
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set Model to training mode
            else:
                model.train(False)  # Set Model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # print(inputs.shape)
                # 2018,8,20,这几行是临时起意加上去的
                # nowbatchsize,c,h,w=inputs.shape
                # if nowbatchsize<opt.batchsize:
                #    continue

                # 下面的几行不知道pytorch 0.4.0还是否支持
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # print(inputs)

                # forward
                outputs = model(inputs)

                # 这一行需要注意，由于model的输出即为softmax的输出，也就是代表了分类的概率，所以
                # 在这里max它的意思就是找到最大的概率对应的索引，将之传送到preds里面
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()  # 反向传播
                    optimizer.step()  # 优化器优化

                # statistics
                running_loss += loss.data.item()  # 将所有的loss加和
                print('{0}*****************************************{1}'.format(preds,labels.data))
                running_corrects += torch.sum(preds == labels.data)  # 将所有预测正确的次数加和

            epoch_loss = running_loss / dataset_sizes[phase]  # 输出损失？？？？
            epoch_acc = running_corrects / dataset_sizes[phase]  # 精确度？？？？

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            # deep copy the Model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)  # 每隔10次保存一次模型
                draw_curve(epoch)  # 蓝色代表‘train’，红色代表‘val’。

        print()

    time_elapsed = time.time() - since  # 相当于计算运行之时间
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  # 计算多少min多少s，//应该是代表整除，%代表取余数。
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best Model weights
    model.load_state_dict(last_model_wts)  # 加载验证集中的最后的模型
    save_network(model, 'last')  # 保存训练完成后最后的模型
    return model



model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=60)




######################################################################
# Save Model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('../model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()  # legend图例
        ax1.legend()
    fig.savefig(os.path.join('../model', name, 'train.jpg'))


