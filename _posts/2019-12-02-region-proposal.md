---
layout:     post
title:      Region Proposal
subtitle:   Region Proposal
date:       2019-12-02
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Feature Map
    - 目标检测
---

[Reference]()

# Region Proposal

> Region Proposal 是目标检测框架中two stage的其中一步

以Faster R-CNN为例：

![](/img/20191221100915.png)

Input image 经过CNN提取得到的feature map，输入到Region Proposal Network，RPN生成可能包含Objects的预定义数量的Region(边界框)。

## 边框的位置表示

目标检测之所以难，是因为每一个物体的区域大小是不一样的，每一个区域框有着不同的大小size(也称之为scale)和不同的长度比（aspect ratios)

1. 若采用(xmin,ymin,xmax,ymax),预测xmin和xmax时，需要保证xmin < xmax 当图片的尺寸和长宽比不一致时，良好训练模型来预测，会非常复杂。

2. 采用`(xcenter,ycenter,width, height)`,学习相对于参考boxes的偏移量


## 如何生成Region?

基于深度学习的目标检测中，可能最难的问题就是生成长度不定（variable-length)的边界框列表。

在RPN中，通过采用`anchors`来代替以往RCNN中`selective search`的方法解决边界框列表长度不定的问题，即，在原始图像中统一放置固定大小的参考边界框，不同于直接检测objects的位置，这里将问题转化为2部分：

1. anchor是否包含相关的object?

对于一个region输出一个binary值p，当p>threshold，则认为该region属于所有类别中的某一类，被选取出来的Region又叫做ROI（Region of Interests）

RPN同时也会在feature map上框定这些ROI感兴趣区域的大致位置，即输出Bounding-box

2. 如何调整anchor以更好的拟合相关的object?

精确的微调：根据每个box中的具体内容微微调整一下这个box的坐标，即输出第一张图中右上方ROI pooling的Bounding-box regression

## RPN的可行性

1. 假设RPN的输入是13x13x256的特征图

2. RPN在该特征图上进行3x3x256的卷积运算，3*3为一个滑动窗口(sliding window)

3. 使用3x3x256的卷积核进行卷积运算，最后依然会得到一个 a x a x 256 的特征图

4. 选定3种不同`scale`和3种不同宽高比`(aspect ratios)`的矩形框作为`基本候选框`: 三种scale size是{128，256，512} | 三种比例是{1：1， 1：2， 2：1}

5. 由于采用卷积默认进行边界填充，那么每一个特征图上一共有13 x 13 = 169个像素点，每一个像素点都可以做一次卷积核的中心点，那么整个卷积下来相当于是有169个卷积中心，这169个卷积中心在原始图像上会有169个对应的anchor锚点，然后每个锚点会有9个默认大小的基本候选框，这样相当于原始图像中一共有169*9=1521个候选框，这1521个候选框中心分布均匀且有9种不同的尺度，所以足以覆盖了整个原始图像上所有的区域，甚至还有大量的重复区域。



