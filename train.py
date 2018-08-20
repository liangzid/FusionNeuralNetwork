"""
this file belongs to DFnn Project.
it is used to train the data,connect the whole thing which written before.
liangzid,2018,7,26

"""

#导入官方库
import os
import sys
import torch
import torchvision as tv
import numpy as np

#导入自己写的函数
import data.Market_1501 as mk1501
import model.hand_on_feature.LOMO as LOMO
import model.Fusion_feature as FNN
import model.NN_feature as CNNfeature
import model.PCA as PCA

















