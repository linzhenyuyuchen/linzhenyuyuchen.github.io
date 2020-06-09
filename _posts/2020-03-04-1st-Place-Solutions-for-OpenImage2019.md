---
layout:     post
title:      OpenImage2019
subtitle:   1st Place Solutions for OpenImage2019
date:       2020-03-04
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 目标检测
    - 实例分割
---

[arxiv](https://arxiv.org/abs/2003.07557)

![](/img/2020053101.png)

# 创新点

1. Decoupling Head (DH)

2. Controllable Margin Loss (CML)

3. Adj-NMS

4. Auto Ensemble

# Decoupling Head

Revisiting the Sibling Head in Object Detector `https://arxiv.org/abs/2003.07540`

Faster RCNN中的分类和回归共享从backbone提取的特征，通常会导致回归的效果较差。Double-Head R-CNN提出了分离feature map到两个分支，减少了共享的参数，分别进行分类任务和回归任务。这样在detection head进行分离能提升一些性能，但是进入两个分支的特征都是有ROI Pooling从同样的proposal生成的。

图像的一些突出区域的特征可能具有丰富的分类信息，而边界周围的特征可能有助于边界框回归，Decoupling Head将从backbone提取的特征分离到两个分支。


# Controllable Margin Loss

![](/img/202005229.png)

# Adj-NMS

先用threshold=0.5的NMS进行筛选，然后采用soft-nms进行筛选。

![](/img/202005221.png)


# Auto Ensemble

集成以下四个模型：

HRNet trained with random scale from 200 to 300

HRNet trained with fixed scale [256,600]

HRNet trained with fixed scale[112,112]

HRNet trained with [256,600] and test with flip mechanism