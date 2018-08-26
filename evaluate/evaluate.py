import scipy.io  #实现python对mat数据的读写的库
import torch
import numpy as np
import time

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):     #这里的起名规则应该是:q代表query,g代表gallery;f是feature,l是label
    query = qf                       #c是cam
    score = np.dot(gf,query)         #将两个feature进行点乘 为什么?
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)   #找到他们标签相同的
    camera_index = np.argwhere(gc==qc)  #找到他们摄像机相同的

    #np.setdiff1d(x,y)函数的作用是返回在x中但不在y中的元素的集合
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)#找到所有在qi中而不再ci中
    junk_index1 = np.argwhere(gl==-1)                                       #的索引
    junk_index2 = np.intersect1d(query_index, camera_index)    #公共元素,与setdiff1d类似
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True) #invert为true时表示寻找不同的值,为False时表示寻找相
                                                   #同的值
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()               #总之就是确定rows就是一个向量,flatten就是让嵌套消失
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat('pytorch_result.mat') #读取提取出来的特征
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
