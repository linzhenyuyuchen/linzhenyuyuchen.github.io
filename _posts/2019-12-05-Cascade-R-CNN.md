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

1. 当提升阈值后，正样本数量会减少，然而模型较大，会导致过拟合

2. detector肯定是在某一个IOU阈值上训练的，但是inference时，生成anchor box之后，这些anchor box跟ground truth box会有各种各样的IOU,已经优化的模型所适应的IOU和输入的proposal不匹配


## Intro

![](/img/compareiou.JPG)

左图比右图多了很多bbox, 且多的bbox大多无意义

1. 生成某个IOU的一系列proposal, 其后对和其IOU对应的detector进行训练

2. 每一个单独的detector只对一个对应的单独IOU进行优化

![](/img/iounumscp.JPG)


![](/img/20191221396569.png)