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


# Faster R-CNN

Faster R-CNN的训练，是在已经训练好的model（如VGG_CNN_M_1024，VGG，ZF）的基础上继续进行训练。实际中训练过程分为6个步骤：

![](/img/20201589965646.png)

1. 卷积层CNN等基础网络，提取特征得到feature map

2. RPN层，再在经过卷积层提取到的feature map上用一个3x3的slide window，去遍历整个feature map,在遍历过程中每个window中心按rate，scale（1:2,1:1,2:1）生成9个anchors，然后再利用全连接对每个anchors做二分类（是前景还是背景）和初步bbox regression，最后输出比较精确的300个ROIs

3. Roi Pooling层固定feature map输入全连接层的维度

4. ROIs映射到经Roi Pooling的feature map上，对得到的proposal feature maps进行bbox回归和分类

5. bbox分类: 通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量

6. bbox回归：利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框

## 四个 Loss

在训练Faster RCNN的时候有四个损失：

（1）RPN 分类损失：anchor是否为前景（二分类）

（2）RPN位置回归损失：anchor位置微调

（3）RoI 分类损失：RoI所属类别（多分类，加上背景）

（4）RoI位置回归损失：继续对RoI位置微调

四个损失相加作为最后的损失，反向传播，更新参数。

## 三个 Creator

（1）AnchorTargetCreator ： 负责在训练RPN的时候，根据ground truth从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出用于训练的位置参数目标。 即返回gt_rpn_loc和gt_rpn_label。

（2）ProposalCreator： 在RPN中，从上万个anchor中，选择一定数目（2000或者300），调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。

（3）ProposalTargetCreator： 负责在训练RoIHead/Fast R-CNN的时候，从RoIs选择一部分(比如128个)用以训练。同时给定训练目标, 返回（sample_RoI, gt_RoI_loc, gt_RoI_label）

## 训练流程

![](/img/2020scjtfo10ql.jpeg)

蓝色箭头的线代表着计算图，梯度反向传播会经过。而红色部分的线不需要进行反向传播

## 总结

- 在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍

- 在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍

- 在RPN阶段分类是二分类（正负样本），而Fast RCNN阶段是多分类

## 解释mismatch问题

- 在training阶段，由于我们知道gt，所以可以很自然的把与gt的iou大于threshold（0.5）的Proposals作为正样本，这些正样本参与之后的bbox回归学习。

- 在inference阶段，由于我们不知道gt，所以只能把所有的proposal都当做正样本，让后面的bbox回归器回归坐标。

我们可以明显的看到training阶段和inference阶段，bbox回归器的输入分布是不一样的，training阶段的输入proposals质量更高(被采样过，IoU>threshold)，inference阶段的输入proposals质量相对较差,没有被采样过，可能包括很多IoU < threshold的，这就是论文中提到mismatch问题，这个问题是固有存在的，通常threshold取0.5时，mismatch问题还不会很严重。

# 模型对比

![](/img/20191221396569.png)

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

## 总结

RPN提出的proposals大部分质量不高，导致没办法直接使用高阈值的detector，Cascade R-CNN使用cascade回归作为一种重采样的机制，逐stage提高proposal的IoU值，从而使得前一个stage重新采样过的proposals能够适应下一个有更高阈值的stage。

- 每一个stage的detector都不会过拟合，都有足够满足阈值条件的样本。

- 更深层的detector也就可以优化更大阈值的proposals。

- 每个stage的H不相同，意味着可以适应多级的分布。

- 在inference时，虽然最开始RPN提出的proposals质量依然不高，但在每经过一个stage后质量都会提高，从而和有更高IoU阈值的detector之间不会有很严重的mismatch。
