'''
=========================================================
更新于2018,8,25

更新内容:
1.由于batchsize的影响,将之前的输入变量由向量改变为张量对输入的张量
进行降维处理;
2.对于特征表示上,去除过去的均值向量+方差直接结合,改为均值方差均值方
差交替
3.输出直接改为张量

liangzia,2018,8,25
=========================================================
'''


import numpy as np
import torch as t

from torch.autograd import Variable

'''
用来进行降维，利用均值和方差来代表一组数据，从而将整体的特征数据降维表出

liangzid,2018,8,20
'''

def dd_one(x,dAfter):
    d_after=dAfter*2
    length=len(x)
    if length%(d_after/2)!=0:
        print('*********************\nERROR!   \nERROR:降维数值有误\n**************************\n')
        return -1
    else:
        block_size=int(length/(d_after/2))
        dd1=np.zeros(int(d_after/2))
        dd2=np.zeros(int(d_after/2))
        
        for i in range(int(d_after/2)):
            xx=x[i*block_size:(i+1)*block_size]
            dd1[i]=get_mean(xx)
            dd2[i]=get_std(xx)
            
        result=np.concatenate((dd1,dd2))
        resultt=t.from_numpy(result).type(t.cuda.FloatTensor)    

        resultt=Variable(resultt,requires_grad=False)
        resultt=resultt.reshape(2,-1)
        
        return resultt


def dd(x,dAfter):

    a,b=x.shape

    feature=np.zeros((a,dAfter))

    for i in range(a):
        per_size=dAfter/2
        if b % (per_size) != 0:
            print('*********************\nERROR!   \nERROR:降维数值有误\n**************************\n')
            return -1
        else:
            block_size=int(b/per_size)
            for j in range(per_size):
                #先提取那一个块出来
                xx=x[i,block_size*j:(block_size+1)*j]
                feature[i,j*2]=get_mean(xx)
                feature[i,j*2+1]=get_std(xx)

    featureture=LOMO_FEATURE2Tensor(feature)

    return featureture


def LOMO_FEATURE2Tensor(feature):
    Tensorr = t.from_numpy(feature).type(t.cuda.FloatTensor)

    Tensorrr = Variable(Tensorr, requires_grad=True)
    return Tensorrr


















def get_mean(x):
    return x.mean()



#说是标准差，实际是方差
def get_std(x):
    mean=get_mean(x)
    std=((x-mean)**2).sum()
    return std



'''
A=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

B=dd(A,d_after=10)
print(B)

'''




