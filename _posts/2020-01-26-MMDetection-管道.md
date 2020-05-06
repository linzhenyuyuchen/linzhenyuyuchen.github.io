---
layout:     post
title:      MMDetection 管道
subtitle:   数据管道，模型和迭代管道
date:       2020-01-26
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - MMDetection
---

[Reference](https://github.com/open-mmlab/mmdetection)

[Doc](https://mmdetection.readthedocs.io/)

# MMDetection

## 数据管道

蓝色块是管道操作。随着管道的进行，每个操作员可以向结果字典添加新键（标记为绿色）或更新现有键（标记为橙色）。 

![](/img/20200506023.jpg)

这些操作分为数据加载，预处理，格式化和测试时间扩充(TTA) MultiScaleFlipAug。

## 迭代管道

对单机和多机采用分布式训练。假设服务器具有8个GPU，则将启动8个进程，并且每个进程都在单个GPU上运行。

每个过程都保持隔离的模型，数据加载器和优化器。模型参数在开始时仅同步一次。在向前和向后传递之后，所有GPU之间的梯度都将减小，优化器将更新模型参数。

由于所有梯度均减小，因此迭代后所有过程的模型参数均保持不变。
