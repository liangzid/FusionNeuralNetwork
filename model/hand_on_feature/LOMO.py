'''
********************=========0=========******************
belong to bigChuang Project: Person Re-ID

please take this python file to the Project directory

reference from: ......

liangzia,2018,5,6 finally

warning: make sure python version =3.x, and make sure you have installed numpy,opencv3,cv2(python)
********************===================******************
how to use it
###########___________1___________##########
加载这两个库
import numpy as np
import cv2
加载本文件
import LOMO
设置图片路径
path=
读取,特征提取
img=cv2.imread(path)
lomo=LOMO.LOMO(img)

print(lomo,lomo.shape)

如果是多个文件：
额外加载
import os
path='./data/VIPeR/cam_a'

img_list=os.listdir(path)
for img_name in img_list:

    img_path=os.path.join(path,img_name)

    img=cv2.imread(img_path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lomo=LOMO.LOMO(img)
    print(lomo)

###########_________2__________############
para:参数解释

img:输入图像,一张,因此读取的时候如若是照片流则需要用for循环，这里和MATLAB版本的LOMO有所不同
c_list: MSR中的方差的列表或元祖
low_clip：MSRCP中使用的剪裁尺寸，下同
high_clip：
R_list=SILTP算法的参数，下同
tau=
blocksize= size of the sub-window for histogram counting.
block_step= sliding step for the sub-windows.
hsv_bin_size number of bins for HSV channels.
'''

import numpy as np
import cv2

'''
image retinex algothrim
it has:
SSR, MSR, MSRCR, MSRCP, and so on.
'''
def Retinex_SingleScale(img,c):
    retinex=np.log10(img)-np.log10(cv2.GaussianBlur(img,(0,0),c))
    return retinex

def Retinex_MultiScale(img,c_list):
    retinex=np.zeros_like(img)
    for c in c_list:
        retinex+=Retinex_SingleScale(img,c)
    retinex=retinex/len(c_list)
    return retinex

def Retinex_MSRCR_ColorRestoration(img,alpha,belta):

    img_sum=np.sum(img,axis=2,keepdims=True)

    return belta*(np.log10(alpha*img)-np.log10(img_sum))

def Retinex_SimplistColorBalance(img,lowclip,highclip):

    total=img.shape[0]*img.shape[1]

    for i in range(img.shape[2]):

        unique,counts = np.unique(img[:,:,i],return_counts=True)
        #unique函数的作用是找到张量中不同元素的值,将其赋予unique(从小到大排序)，然后将索引赋予count
        current=0

        for u,c in zip(unique,counts):
            if float(current)/total<lowclip:
                low_val=u
            if float(current)/total<highclip:
                high_val=u

            current+=c
        img[:,:,i]=np.maximum(np.minimum(img[:,:,i],high_val),low_val)

    return img

def Retinex_MSRCR(img,c_list,G,b,alpha,belta,low_clip,high_clip):

    img=np.float(img)+1.0
    img_retinex=Retinex_MultiScale(img,c_list,)
    img_color=Retinex_MSRCR_ColorRestoration(img,alpha,belta)

    img_msrcr=G*(img_retinex*img_color+b)

    img_msrcr=np.uint8(np.minimum(np.maximum(img_msrcr,0),255))
    img_msrcr=Retinex_SimplistColorBalance(img_msrcr,low_clip,high_clip)
    return img_msrcr

def Retinex_AutomatedMSRCR(img, sigma_list):
    img = np.float(img) + 1.0

    img_retinex = Retinex_MultiScale(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.05:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.05:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def Retinex_MSRCP(img, sigma_list, low_clip, high_clip):
    img = img + 1.0

    intensity = np.sum(img, 2) / img.shape[2]

    retinex = Retinex_MultiScale(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = Retinex_SimplistColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp
'''

SILTP algothrim(CVPR2010)

LBP对噪声敏感,LTP对光照敏感,SILTP是二者的改进版

'''
def SILTP(img,R,tau):
    if len(img.shape)>2: # 在这里img如果是彩图就应该是3维张量,如果是2维张量就是灰度图
        imgcpy=np.uint8(img)
        img=cv2.cvtColor(imgcpy,cv2.COLOR_BGR2GRAY)

    img_pad=np.pad(img,R,'edge') #pad 函数是填充函数，'edge'代表边缘填充，填充多少取决于R

    R_=-1*R
    img_u=img_pad[:2*R_,R:R_]
    img_d=img_pad[2*R:,R:R_]
    img_l=img_pad[R:R_,2*R:]
    img_r=img_pad[R:R_,:2*R_]
    up_limit=(1+tau)*img
    low_limit=(1-tau)*img

    siltp=((img_u<low_limit)+(img_u>up_limit)*2)+((img_d<low_limit)+(img_d>up_limit)*2)*3+\
          ((img_r<low_limit)+(img_r>up_limit)*2)*(9)+((img_l<low_limit)+(img_l>up_limit)*2)*27

    return siltp
'''

LOMO algothrim

'''
def jointHistogram(img,boundary,bin_size):

    interval=(boundary[1]-boundary[0]+1)/bin_size

    if len(img.shape)>2:
        histsize=bin_size**(img.shape[2])
        img_bin=np.zeros([img.shape[0],img.shape[1]],np.int32)
        for i in range(img.shape[2]):
            img_bin=img_bin+((img[:,:,i]-boundary[0])/interval)/(bin_size**i)
    else:
        histsize=bin_size
        img_bin=(img-boundary[0])/interval

    unique,counts=np.unique(img_bin,return_counts=True)

    histogram=np.zeros([histsize])

    for u,c in zip(unique,counts):
        histogram[int(u)]=int(c)

    return histogram

def averagePooling(img):

    if img.shape[0]%2 !=0:
        img=img[:-1]
    if img.shape[1]%2 !=0:
        img=img[:,:-1]

    img_pool=img[::2]+img[1::2]
    img_pool=img_pool[:,::2]+img_pool[:,1::2]

    img_pool=img_pool/4

    return img_pool

def LOMO(img,c_list=[5,20],low_clip=0.1,high_clip=0.9,
         R_list=[3,5],tau=0.3,hsv_bin_size=8,blocksize=10,block_step=5):


    img_retinex=Retinex_MSRCP(img,c_list,low_clip,high_clip)
    siltp_feat=np.array([])
    hsv_feat=np.array([])
    for pool in range(3):
        row_num=int((img.shape[0]-(blocksize-block_step))/block_step)
        col_num=int((img.shape[1]-(blocksize-block_step))/block_step)
        for row in range(row_num):
            for col in range(col_num):
                img_block=img[
                    row*block_step:row*block_step+blocksize,
                    col*block_step:col*block_step+blocksize
                ]

                siltp_hist=np.array([])
                for R in R_list:
                    siltpp=SILTP(img_block,R,tau)
                    unique,count=np.unique(siltpp,return_counts=True)
                    siltp_hist_r=np.zeros([3**4])
                    for u,c in zip(unique,count):
                        siltp_hist_r[u]=c
                    siltp_hist=np.concatenate([siltp_hist,siltp_hist_r],0)

                img_block=img_retinex[
                    row * block_step:row * block_step + blocksize,
                    col * block_step:col * block_step + blocksize
                    ]
                img_block_copy=np.uint8(img_block)
                img_hsv=cv2.cvtColor(img_block_copy,cv2.COLOR_BGR2HSV)
                hsv_hist=jointHistogram(
                    img_hsv,
                    [0,255],
                    hsv_bin_size
                )

                if col==0:
                    siltp_feat_col=siltp_hist
                    hsv_feat_col=hsv_hist
                else:
                    siltp_feat_col=np.maximum(siltp_feat_col,siltp_hist)
                    hsv_feat_col=np.maximum(hsv_feat_col,hsv_hist)

            siltp_feat=np.concatenate([siltp_feat,siltp_feat_col],0)
            hsv_feat=np.concatenate([hsv_feat,hsv_feat_col],0)


        img=averagePooling(img)
        img_retinex=averagePooling(img_retinex)

    siltp_feat=np.log(siltp_feat+1.0)
    siltp_feat[:int(siltp_feat.shape[0]/2)]/=np.linalg.norm(siltp_feat[:int(siltp_feat.shape[0]/2)])
    siltp_feat[int(siltp_feat.shape[0] / 2):] /= np.linalg.norm(siltp_feat[int(siltp_feat.shape[0] / 2):])

    hsv_feat=np.log(hsv_feat+1.)
    hsv_feat/=np.linalg.norm(hsv_feat)

    lomo=np.concatenate([siltp_feat,hsv_feat],0)

    return lomo





