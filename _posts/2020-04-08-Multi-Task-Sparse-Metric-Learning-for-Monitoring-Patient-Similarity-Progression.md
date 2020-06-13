---
layout:     post
title:      Multi-Task Sparse Metric Learning for Monitoring Patient Similarity Progression
subtitle:   用于监测患者相似性进展的多任务稀疏度量学习
date:       2020-04-08
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - KDD
    - 数据挖掘
---

# mtTSML

[Referrence](https://www.researchgate.net/publication/330028201_Multi-task_Sparse_Metric_Learning_for_Monitoring_Patient_Similarity_Progression)

首先采用low-rank transformation matrices分解马氏矩阵，这样马氏距离可以作为对所有task和特定task的距离。在transformation matrix中每一列可以当作是衡量对应特征的向量。然后对每一个task在transformation matrix上利用L2,1正则化进行特征选择，将无关信息列置为0。考虑到疾病标签的相似性，提出triplet constraints，在相似对和不相似对之间设置一个margin，这样不仅能够使有相同标签的病人相似，不同标签的病人不相似，也能够获得有序标签的关系。

## 主要贡献

- 提出mtTSML可以监控病人相似性进展，同时学习到在不同未来时间节点上病人的相似性距离。利用学习到的距离矩阵，可以将临床研究转为监控相似变量的变化趋势。

- 在不同task学习过程中进行sparse feature selection，对每个task中高维输入空间中的无代表性的特征进行删除。

- 引入相似等级信息，在距离约束条件中关注疾病标签的有序关系。这样可以充分反应疾病的严重等级。

- 应用实际医疗健康数据证明mtTSML是SOTA。

## Proposed Method

![](/img/vgg1123.png)

因为病人的状态是一直改变的，我们关注于一些特定的时间点。学习一个timestamp的病人相似性叫做one task，如图右上角的每个task合起来可以看作是这些病人相似性学习的multi-task

## Low-Rank Metric Formulation


## Sparse Feature Selection






