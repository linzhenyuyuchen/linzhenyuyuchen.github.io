---
layout:     post
title:      ROI Pooling Align
subtitle:   ROI Align & ROI Pooling
date:       2019-12-08
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
    - 目标检测
---

# ROI Pooling

## ROI Pooling 的输入

-由 Backbone 产生的 Feature maps

- RPN 产生的 ROIs

## ROI Pooling 的参数

- pooled_width

- pooled_height

- spatial_scale

## 计算过程

1. 在feature map中找到roi位置

2. 将roi位置等分为pooled_width X pooled_height的网格

3. 对网格中每一格都做max pooling

![](/img/v2-0c6eb6731e92957b9e23ae727ffc2f21_b.webp.gif)

## ROI Pooling操作中两次量化造成区域不匹配

在two stage中，ROI Pooling的作用是根据预选框的位置坐标在特征图中将相应区域池化为`固定尺寸`的特征图，以便进行后续的分类和包围框回归操作。

由于预选框的位置通常是由模型回归得到的，一般来讲是浮点数，而池化后的特征图要求尺寸固定。故ROI Pooling这一操作存在两次量化的过程。

- 将候选框边界量化为整数点坐标值。

- 将量化后的边界区域平均分割成 k x k 个单元(bin),对每一个单元的边界进行量化。

因此量化后的结果与最开始的位置会有一定的偏差，也就是misalignment

# ROI Align

misalignment对于小目标影响较大，ROI Align 正好可以解决misalignment的问题

1. Conv layers使用的是VGG16，feat_stride=32(即表示，经过网络层后图片缩小为原图的1/32),原图800 * 800,最后一层特征图feature map大小:25 * 25

2. 假定原图中有一region proposal，大小为665 * 665，这样，映射到特征图中的大小：665/32=20.78,即20.78 * 20.78，此时，没有像RoiPooling那样就行取整操作，保留浮点数

3. 假定pooled_w=7,pooled_h=7,即pooling后固定成7 * 7大小的特征图，所以，将在 feature map上映射的20.78 * 20.78的region proposal 划分成49个同等大小的小区域，每个小区域的大小20.78/7=2.97,即2.97 * 2.97

4. 假定采样点数为4，即表示，对于每个2.97*2.97的小区域，平分四份，每一份取其中心点位置，而中心点位置的像素，采用双线性插值法进行计算，这样，就会得到四个点的像素值，如下图

![](/img/202030388125.png)

上图中，四个红色叉叉x的像素值是通过双线性插值算法计算得到的

最后，取四个像素值中最大值作为这个小区域(即：2.97*2.97大小的区域)的像素值，如此类推，同样是49个小区域得到49个像素值，组成7*7大小的feature map
