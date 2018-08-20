'''
New file in vsCode,windows10 X64,pytorch,python3.
(please check it if using it on linux)
build the fusion part of NN
liangzia,2018,5,23
'''
import torch as t
import numpy as np
from .BasicModule import BasicModule
from torch import nn

class Fusion_feature(BasicModule):
    def __init__(self):
        
        super(Fusion_feature,self).__init__()

        self.model_name='FusionFeature'

        self.ffusion=t.nn.Sequential(
            nn.Linear(8192,4096,bias=1),
            nn.Linear(4096,4096,bias=1),
            nn.ReLU(),
            nn.Dropout(0.5)

        )

    def view_as(self,feature):
        return feature.view(feature.size(0),-1)

    def cat(self,feature1,feature2):
        feature=t.cat((feature1,feature2),0)
        return feature

    def forward(self,feature):
        feature=self.ffusion(feature)
        return feature

        














