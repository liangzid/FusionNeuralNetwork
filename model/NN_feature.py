'''
warning: this file may be wrong in somewhere, be careful if you use it..

refer to Caffe Code..

liangzia,2018,4,27
new changes:========================================
modify in 2018, 5, 22

some wrong go away.
NN feature extraction
input is a vector whose dim is 256, and the output is 4096
We need to use it.
'''

import torch as t
import numpy as np
from BasicModule import BasicModule
from torch import nn


class deep_feature(BasicModule):
    '''
    项目中提取深度特征的部分，输入是一个256的向量，输出一个4096维度的特征向量
    if wrong,you must letter to liangzid ...
    '''

    def __init__(self):
        
        super(deep_feature,self).__init__()

        self.model_name='DeepFeatureExtration'

        self.feature=t.nn.Sequential(
            #这里的in_channal存在疑问

            nn.Conv2d(256,96,kernel_size=11,stride=4), # 这里没有找到卷积核的初始化方式该如何给出
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.CrossMapLRN2d(size=5,alpha=0.0001,beta=0.75),

            nn.Conv2d(96,256, kernel_size=5, stride=1,padding=2,groups=2,bias=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(256,384, kernel_size=3, stride=1, padding=1, groups=1, bias=0),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2, bias=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2, bias=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Linear(256,4096,bias=1),

            nn.Dropout(p=0.5)
        )

        self.classifier1=nn.Sequential(

            nn.Linear(256*6*6,4096),
            nn.ReLU(),

            nn.Dropout(0.5)
        )
        

    def forward(self, x):
        self.feature(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        self.classifier1(x)
        return x




#这一部分往下毫无用处，甚至还有一些多余，已经被删掉
#此处有未解决的疑惑，我写下的是我认为我是对的的代码
        #self.cat=t.cat((),1) 这一步是在main函数里
        #self.calssifier2=nn.Sequential(

         #   nn.Linear(4096,4096),
          #  nn.ReLU(),
           # nn.Dropout(0.5)
        #)
        #预测精确度的方法未封装,以后看看有没有必要放在模型里，还是在主函数中
        #self.accurary=????????

