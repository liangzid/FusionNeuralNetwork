"""
this file belongs to the DFNN Project, it is used to extracted mainly feature
from LOMO hand_crafted feature,which turns 26960 -d to 4096 -d for Fusion.

liangzia,2018,7,26...

"""
import torch as t
import numpy as np

def sp_PCA(lomo_feature,dAfter=4096):
    '''

    :param lomo_feature:默认是numpy向量
    :param dAfter:      默认留下来的维度的数目
    :return:            默认是返回的主成分
    '''

    X=t.from_numpy(lomo_feature)

    #正则化
    Xmean=X-t.mean(X,0).expand_as(X)

    U,S,V=t.svd(t.t(Xmean))
    return t.mm(Xmean,U[:,dAfter]) #这一步很灵性，torch.mm实现的是一个矩阵和一个向量的相乘
