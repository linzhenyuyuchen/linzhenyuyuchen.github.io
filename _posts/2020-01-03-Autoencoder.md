---
layout:     post
title:      Autoencoder
subtitle:   自编码器 栈式自编码器 去噪自编码器
date:       2020-01-03
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 特征提取
---

# Autoencoder

> 自动编码器本质上是学习输入数据低维特征表示的神经网络结构。

autoencoder是多层神经网络，其中输入层和输出层表示相同的含义，具有相同的节点数，训练的目的是得到与输入相同的输出。

![](/img/22619e9c.png)

但是，这样可以得到维度较低的中间层，这一层能较好地代表输入，起到了降维的作用。

# Stacked Autoencoder (SAE)

接着去掉Layer L3，把L1和L2看作整体作为新的输入，重复刚才的做法。

这样便是栈式自编码器。

# Stacked Denoising Autoencoder (SDAE)

在输入的过程中加入噪声。


# 总结

自编码器并不是一个真正的无监督学习的算法，而是一个自监督的算法。自监督学习是监督学习的一个实例，其标签产生自输入数据。要获得一个自监督的模型，你需要想出一个靠谱的目标跟一个损失函数，那么问题来了，仅仅把目标设定为重构输入可能不是明智的选择。
