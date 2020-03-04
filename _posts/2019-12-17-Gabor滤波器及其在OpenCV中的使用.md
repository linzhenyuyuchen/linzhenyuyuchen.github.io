---
layout:     post
title:      Gabor滤波器及其在OpenCV中的使用
subtitle:   Gabor在特征提取上的应用
date:       2019-12-17
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 特征提取
    - OpenCV
---

# Gabor滤波器

## 概念

Fourier 变换是一种信号处理中的有力工具，可以将图像从空域转换到频域，并提取到空域上不易提取的特征。但是 Fourier 变换缺乏时间和位置的局部信息。

Gabor 变换是一种加窗短时 Fourier 变换（简单理解起来就是在特定时间窗内做 Fourier 变换），是短时 Fourier 变换中当窗函数取为高斯函数时的一种特殊情况。其频率和方向表达与人类视觉系统类似。

Gabor与脊椎动物视觉皮层感受野响应的比较如下所示：

![](/img/853143432354364.JPEG)

图中第一行是脊椎动物的视觉响应，第二行是Gabor滤波器的响应，可以看到，二者相差极小。

Gabor特征是一种可以用来描述图像纹理信息的特征。此外，Gabor小波对于图像的边缘敏感，能够提供良好的方向选择和尺度选择特性，可以在频域不同尺度、不同方向上提取相关的特征。Gabor滤波器可以提取不同方向上的纹理信息。Gabor滤波器对于光照变化不敏感,能够提供对光照变化良好的适应性，能容忍一定程度的图像旋转和变形，对光照、姿态具有一定的鲁棒性。

基于以上特性，Gabor滤波器被广泛应用于人脸识别的预处理。

## 定义

在二维空间中，使用一个三角函数(如正弦函数)与一个高斯函数相乘就得到了一个Gabor滤波器。

![gabor filter](/img/20200304140640.png)

- x,y分别表示像素坐标位置
- λ 表示滤波的波长；（波长越大，黑白相间的间隔越大）
- θ 表示Gabor核函数图像的倾斜角度；
- ϕ表示相位偏移量，范围是-180~180；（ϕ=0时白条为中心，ϕ=180时，黑条为中心 ）
- σ表示高斯函数的标准差；（σ增大，条纹数量越多）
- γ表示长宽比，决定这Gabor核函数图像的椭圆率。（γ越小，核函数图像会越高）


在OpenCV中的getGaborKernel函数里需要传入的参数除了上述5个外，还需要传入卷积核的大小。

## 在线演示系统

`http://www.cs.rug.nl/~imaging/simplecell.html`

`http://matlabserver.cs.rug.nl/cgi-bin/matweb.exe`



# Python OpenCV

## cv2.getGaborKernel

```python
- Size  ksize,   #Size of the filter returned.

- double  sigma,    #Standard deviation of the gaussian envelope.

- double  theta,    #Orientation of the normal to the parallel stripes of a Gabor function.

- double  lambd,    #Wavelength of the sinusoidal factor.

- double  gamma,    #Spatial aspect ratio.

- double  psi = CV_PI *0.5, #Phase offset.

- int ktype = CV_64F    #Type of filter coefficients. It can be CV_32F or CV_64F .

```

## cv2.filter2D

```python
- InputArray  	src,    #input image. 
- OutputArray  	dst,    #output image of the same size and the same number of channels as src. 
- int  	ddepth, #desired depth of the destination image
- InputArray  	kernel, #convolution kernel (or rather a correlation kernel), a single-channel floating point matrix; if you want to apply different kernels to different channels, split the image into separate color planes using split and process them individually. 
- Point  	anchor = Point(-1,-1),
- double  	delta = 0,
- int  	borderType = BORDER_DEFAULT 
```

## Gabor滤波器python代码

```python
import cv2,os
import numpy as np
import matplotlib.pyplot as plt


def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


#构建Gabor滤波器
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # gabor尺度，6个
     lamda = np.pi/2.0         # 波长
     for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
         for K in range(6):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     plt.figure(1)

     #用于绘制滤波器
     for temp in range(len(filters)):
         plt.subplot(4, 6, temp + 1)
         plt.imshow(filters[temp])
     plt.show()
     return filters

#Gabor特征提取
def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))

    #用于绘制滤波效果
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
    plt.show()
    return res  #返回滤波结果,结果为24幅图，按照gabor角度排列


if __name__ == '__main__':
    filters = build_filters()
    '''
    input_Path = './content'
    img_paths = get_img(input_Path)
    for img in img_paths:
        img = cv2.imread(img)
        getGabor(img, filters)
    '''
    img = cv2.imread('/data1/input.png')
    getGabor(img, filters)
```

