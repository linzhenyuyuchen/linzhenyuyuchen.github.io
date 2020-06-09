---
layout:     post
title:      KDD RETAIN
subtitle:   An Interpretable Predictive Model for
Healthcare using Reverse Time Attention Mechanism / Knowledge-Discovery in Databases
date:       2020-03-08
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 数据挖掘
    - KDD
---

# RETAIN

RETAIN基于两级神经注意模型，以相反的时间顺序处理EHR数据来模拟医生操作。准确度和可解释性是成功的预测模型的两个主要特征。RETAIN有详细的可解释性，准确度和速度。通过具体的案例研究和可视化方法，展示了RETAIN如何提供直观的解释。

## encounter sequence modeling (ESM)

一个病人的某次病例是一个医学代码的集合 {c1, c2, . . . , cn}

给定一个病例序列x1, . . . , xT，ESM的目标是输出下一次的病例x2, . . . , xT+1 和 标签的数量 s = |C|

##  learning to diagnose (L2D)

L2D可以看作是一个特殊的ESM，它在病例序列的最后做一个简单的预测，预测特定疾病（s=1）或多个疾病（s>1）的发病情况。

## Overview

RETAIN一个中心特性是将相当大一部分预测责任交给生成注意权重的过程。这在一定程度上是为了解决解释rnn的困难，在rnn中，递归权重将过去的信息传递给隐藏层。因此，为了同时考虑访问级别和变量级别（单个坐标xi）的影响，我们使用输入向量xi的线性嵌入。

![](/img/2020052801.png)

其中Wemb∈R表示输入向量xi的嵌入，m是嵌入向量维数的大小,可以使用MLP进行学习表示

针对visit-level attention 和 variable-level attention 有两套分别的权重

![](/img/2020052802.png)

## Evaluation measures

- Negative log-likelihood that measures the model loss on the test set. The loss can be calculated
by Eq (1)

- Area Under the ROC Curve (AUC) of comparing ybi with the true label yi
. AUC is more robust to imbalanced positive/negative prediction labels, making it appropriate for evaluation of classification accuracy in the heart failure prediction task.


## 超参

visit embedding size m, RNNα’s hidden
layer size p, RNNβ’s hidden layer size q, L2 regularization coefficient, and drop-out rates.

L2 regularization was applied to all weights except the ones in RNNα and RNNβ. Two separate
drop-outs were used on the visit embedding vi and the context vector ci. We performed the random
search with predefined ranges m, p, q ∈ {32, 64, 128, 200, 256}, L2 ∈ {0.1, 0.01, 0.001, 0.0001},
dropout vi, dropout ci ∈ {0.0, 0.2, 0.4, 0.6, 0.8}. We also performed the random search with m, p
and q fixed to 256.

The final value we used to train RETAIN for heart failure prediction is m, p, q = 128, dropoutvi = 0.6,dropout ci = 0.6 and 0.0001 for the L2 regularization coefficient.

## 结论

复杂模型可以提供更高的预测精度和更精确的解释能力。考虑到RNN对序列数据的分析能力，提出了RETAIN，它在保留RNN预测能力的同时，拥有更高程度的解释。RETAIN算法的核心思想是通过复杂的注意力来提高预测精度，同时保持表示学习部分简单易懂，使得整个算法准确易懂。以相反的时间顺序保留两个RNN序列，以有效地生成适当的注意力变量。