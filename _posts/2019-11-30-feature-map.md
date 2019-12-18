---
layout:     post
title:      Feature Map
subtitle:   卷积神经网络中的 feature map
date:       2019-11-30
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Feature Map
---

[Reference]()

# Feature Map

在每个卷积层，数据都是以三维形式存在的。你可以把它看成许多个二维图片叠在一起，其中每一个称为一个feature map。
 
在输入层，如果是灰度图片，那就只有一个feature map。如果是彩色图片，一般就是3个feature map (RGB)。层与层之间会有若干个卷积核（kernel），上一层的每个feature map跟每个卷积核做卷积，都会产生下一层的一个feature map。

# 卷积

通过图像卷积后，新图像的大小跟原来一样，或者变小。

## Same

滑动步长为1，图片大小为N1xN1，卷积核大小为N2xN2，卷积后图像大小：N1xN1

![](/img/full_conv.png)

## Valid

滑动步长为S，图片大小为N1xN1，卷积核大小为N2xN2，卷积后图像大小：(N1-N2)/S+1 x (N1-N2)/S+1

![](/img/2019121821507.gif)

滑动步长为1，图片大小为5x5，卷积核大小为3x3，卷积后图像大小：3x3

## Full (反卷积)

滑动步长为1，图片大小为N1xN1，卷积核大小为N2xN2，卷积后图像大小：N1+N2-1 x N1+N2-1

![](/img/full_conv_deconv.gif)

滑动步长为1，图片大小为2x2，卷积核大小为3x3，卷积后图像大小：4x4

# 反卷积

## 第一种

上面的Full

## 第二种

![](/img/20191218260158.png)

假设原图是3X3，首先使用上采样让图像变成7X7，可以看到图像多了很多空白的像素点。使用一个3X3的卷积核对图像进行滑动步长为1的valid卷积，得到一个5X5的图像。

使用上采样扩大图片，使用反卷积填充图像内容，使得图像内容变得丰富，这也是CNN输出end to end结果的一种方法。
