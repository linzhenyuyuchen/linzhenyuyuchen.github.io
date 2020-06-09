---
layout:     post
title:      Double Head R-CNN
subtitle:   全连接层用于分类 卷积层用于回归
date:       2020-02-24
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 深度学习
    - 目标检测
    - MMDetection
---

# Double Head R-CNN

文章来源：[Rethinking Classification and Localization for Object Detection](https://arxiv.org/abs/1904.06493)

> 分类任务和回归任务采用了不同的head，其中fc-head 比conv-head更具有spatial sensitivity，fc-head能区分完整对象和部分对象的能力，而卷积层能回归目标对象

![](/img/20200521123.png)

## Double-Head 的四种搭配

- Double-FC

- Double-Conv

- Double-Head

- Double-Head-Reverse

![](/img/2020052134.png)

### Unfocused Task Supervision

![](/img/2020052122.png)

![](/img/2020052133.png)

### Complementary  Fusion  of  Classifiers

![](/img/2020052155.png)

s是分类score，分别来自conv和fc层

## Experiment

![](/img/202005219.png)

# MMDetection

[reference](https://mmdetection.readthedocs.io/en/latest/tutorials/new_modules.html#add-new-heads)


修改config文件：

```python

    bbox_head=[
        dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))

```