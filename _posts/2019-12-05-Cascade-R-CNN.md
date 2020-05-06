---
layout:     post
title:      Cascade R-CNN
subtitle:   Cascade R-CNN
date:       2019-12-05
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
    - 目标检测
---

[Reference](https://arxiv.org/abs/1712.00726)

# Cascade R-CNN

## 前言

当使用较低IOU阈值训练时，通常会导致noisy detection，而当提高阈值后性能会有所提升，原因有二：

1. 选取正负样本的方式主要利用候选框与ground truth的IOU占比，常用的比例是50%.当提升阈值后，正样本数量会减少，然而模型较大，会导致过拟合

2. detector肯定是在某一个IOU阈值上训练的，但是inference时，生成anchor box之后，这些anchor box跟ground truth box会有各种各样的IOU,已经优化的模型所适应的IOU和输入的proposal不匹配


## 出发点

![](/img/compareiou.JPG)

左图比右图多了很多bbox, 且多的bbox大多无意义

Cascade R-CNN就是使用不同的IOU阈值，训练了多个级联的检测器。

1. 生成某个IOU的一系列proposal, 其后对和其IOU对应的detector进行训练

2. 每一个单独的detector只对一个对应的单独IOU进行优化

## 改进

目标检测其实主要干的就是两件事，一是对目标分类，二是标出目标位置。为了实现这两个目标，在训练的时候，一般会首先提取候选proposal，然后对proposal进行分类，并且将proposal回归到与其对应的groud truth上面，分类最常用的做法是利用IOU（proposal与ground truth的交并比），常用的阈值是0.5，可是0.5是最好的吗？作者通过实验证实了不同IOU对于网络的影响。

![](/img/iounumscp.JPG)

是否可以将阈值提高，以达到优化输出精度的效果呢？

作者又做了不同阈值下网络精度的实验，结果如图所示，可以发现，对于阈值为0.5以及0.6的时候，网络精度差距不大，甚至提升了一点，但是将精度提升到0.7后，网络的精度就急速下降了，(COCO数据集上：AP：0.354->0.319)，这个实验说明了，仅仅提高IoU的阈值是不行的，因为提高阈值以后，实际上网络的精度（AP）反而降低了。

为什么会下降呢？

1. 由于提高了阈值，导致正样本的数量呈指数减低，导致了训练的过拟合。

2. 在inference阶段，输入的IOU与训练的IOU不匹配也会导致精度的下降。所以才会出现，u=0.7的曲线在IOU=0.5左右的时候，差距那么大。

实验证明不能使用高的阈值来进行训练，但是实验也呈现出了另一个事实，那便是：回归器的输出IOU一般会好于输入的IOU。并且随着u的增大，对于在其阈值之上的proposal的优化效果还是有提升的。

既然这样，可以采用级联的方式逐步提升即首先利用u=0.5的网络，将输入的proposal的提升一些，假如提升到了0.6，然后在用u=0.6的网络进一步提升，加入提升到0.7，然后再用u=0.7的网络再提升。

## 模型对比

![](/img/20191221396569.png)