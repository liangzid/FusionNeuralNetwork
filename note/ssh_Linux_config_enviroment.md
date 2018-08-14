>这篇文章在2018年8月14日进行了改动。改动原因：深度学习服务器网址变化。

# 利用ssh远程登录深度学习服务器并且配置行人再识别环境笔记
1. 本地环境：Ubuntu18.04；
2. 远程服务器环境：Ubuntu14.04；
3. 目标配置：python深度学习环境，pytorch，opencv2.4.3，cv3，使用python3.6，cuda 8.0
4. 注释：不存在root权限



## 连接上远程服务器
### 使用VNC VIEWER

VNC VIEWER是一个GUI界面的连接器，在搜索栏输入网址和端口，按照要求来即可。这里给出网址和端口：

219.216.72.117:70
### 使用ssh

打开shell，输入 ssh hostname@web 在提示中输入password即可

```commandline
ssh liangzi@219.216.72.117
```

即可连接上远程服务器

## 上传与下载文件
### 上传文件

在本地shell下使用
```commandline
scp -r /home/liangzi/FusionNeuralNetwork liangzi@219.216.72.117:/home/liangzi/
```
注意，这里的地址需要改成自己的。如果需要指定端口，需要使用-P（此处的P一定是要大写，小写的已经被占用了）

> 这样一来，就将代码上传到了远程服务器中。如若打开远程服务器，进入home下，可以发现很多学长的名字，这应该都是他们的文件夹……

### 下载文件

在本地shell下使用
```commandline
scp -r liangzi@219.216.72.117:/path.filename /home/liangzi/cc
```
注意，这里的-r指的是连着文件夹和里面的文件一起copy

## 环境配置（只是配置了梁子本身使用了的深度学习环境，使用者可以自定义）
### 安装conda
#### 下载anaconda的镜像
[清华大学镜像](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)，自己选择系统、版本。比较新的都在下面。
#### 配置路径
这一步可以省略，我就省略了。
参见这个[教程](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)

运行如下代码（可能需要管理员身份，所以没有root权限的可以试一试，像我就没试）
```commandline
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# for legacy win-64
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
```
这样conda就应该安装完毕（所谓的安装不过是一个文件夹）

### 安装pytorch

进入[pytorch官网](https://pytorch.org/)，找到自己电脑的型号，系统Linux，选用conda安装，cuda8

关于cuda的版本可以从这里找到：
```commandline
cat /usr/local/cuda/version.txt
```
cudnn的版本查询方式
```commandline
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
在服务器shell下输入官网上给出的指令，即可开始安装。
> 可能会出现一些问题，就是http报错，这个问题的解决是使用pip安装……这要求之前安装conda是加入路径。
使用pip安装的话可以加上下面这句话：

```commandline
--index https://pypi.mirrors.ustc.edu.cn/simple/
```
> 或者
```commandline
--default-timeout=100 future
```
### 安装opencv

#### 检查是否已经安装了opencv
首先检查一下是否电脑已经配置了opencv，在shell输入：
```commandline
pkg-config --modversion opencv
```
深度学习服务器显示2.4.13版本，说明opencv已经安装完毕。

####安装cv2

输入
```commandline
pip install opencv-python
```
> 到此，所有的安装都已经完成！！！开始享受GPU的飞速吧！或者赶紧把代码写完......

## 使用过程中的可能问题：
### conda安装之torch不存在
这是由于重启之后conda被覆盖，需要添加conda路径。可以将这一指令添加到开机或者开启命令行的.rc文件中，代码为
```commandline
export PATH=/home/liangzi/anaconda3/bin:$PATH
```
此外，深度学习服务器无法展示命令行之外的东西。














