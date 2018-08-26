'''
从现在开始尝试搞清楚这些代码并在此基础上把自己写的代码成功RUN出来
'''



import os
from shutil import copyfile

# You only need to change this line to your dataset download path
download_path = '../Market1501'#注意：这是需要下载的数据集所在的路径,默认为是与FusionNeuralNetwork相平行的文件夹

if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch'#这是需要下载的数据集所要保存的路径
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#query                          #似乎就是多的意思
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = query_path + '/' + name        #存放的是所有的图片的路径的列表
        dst_path = query_save_path + '/' + ID[0]  #存放的是所有图片对应的id的列表
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)                    #新建了无数个文件夹，位置是/Market1501/pytorch/query/001/等
        copyfile(src_path, dst_path + '/' + name) #将所有的文件都拷贝到上面的那些文件夹里

#-----------------------------------------
#gallery
gallery_path = download_path + '/bounding_box_test' #测试集里面的图片的路径
gallery_save_path = download_path + '/pytorch/gallery'#新的保存后的路径
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True): #这里执行的操作和上面一样，把所有的东西都
    for name in files:                                        #放置在了新的文件夹，每个文件夹对应一个id
        if not name[-3:]=='jpg':                              #里面全部是关于这个id的图像
            continue
        ID  = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

#---------------------------------------
#train_all
train_path = download_path + '/bounding_box_train'        #同上
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


#---------------------------------------
#train_val
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:]=='jpg':
            continue
        ID  = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):             #val是指validation，验证集。用来调节超参数
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
