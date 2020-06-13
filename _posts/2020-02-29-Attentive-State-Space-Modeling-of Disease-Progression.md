---
layout:     post
title:      Attentive State-Space Modeling of
Disease Progression
subtitle:   Attentive State-Space Modeling of
Disease Progression
date:       2020-02-29
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - KDD
    - RNN
---

[Github](https://github.com/AnyLeoPeace/state_space)

# Introduction

作者开发了一个疾病进展的深度概率模型，能够提供具有临床意义和非马尔可夫状态变化，关注于可解释结构表示和深度学习方法预测能力。可设计为任意的复杂度而不失可解释性。模型以无监督的方法学习隐含疾病状态，这适合于没有label的EHR数据。

注意力机制关注于病人的慢性病史，学习注意力权重来决定之前的疾病状态对未来的影响。

使用a sequence-to-sequence RNN architecture [8]来实现动态注意力机制，捕获复杂的动态，同时保持结构的可解释性。

模型是非马尔可夫，疾病的推断是复杂的，而且没法通过前后向传播获得。为了解决这一问题，设计了另一个结构化的推断网络，共享权重，共同训练。

实验数据： breast cancer  (UK Cystic Fibrosis registry)

本文与RETAIN的注意力机制区别在于the latent state space 和 observable sample space , 本文的模型关注于潜在的疾病动态，从而能对隐含的疾病进展提供更好的解释

# Attentive State-Space Models

![](/img/vmm20.png)

对于一个时间步t来说，zt代表xt的隐含状态，zt无法直接被观察到，需要通过无监督学习来得到

![](/img/uns2020.png)

## Attentive state transitions

- transition probability 根据一个病人的所有历史状态xt-1和zt-1计算t时刻他的健康状态，这与正常的马尔可夫只关注前一个状态zt-1不同

![](/img/vn221.png)

## Emission distribution

![](/img/vnn222.png)

# Sequence-to-sequence Attention Mechanism

普通seq2seq注意力模型的attention是在解码过程中用作中间表示，而本文的seq2seq注意力对每个病人的向量XT中的xt都学习attention weights

![](/img/vmm123.png)

# Experiments


## Implementation

- attention network: LSTM 2 hidden layer of size 100

- ADAM with lr = 5X10-4

- mini-batch = 100

- 5-fold cross-validation

## Data description

a cohort of patients enrolled in the UK CF registry

10,263 patients over the period from 2008 and 2015 with a total of 60,218 hospital visits

Each patient is associated with 90 variables, including information on 36 possible treatments, diagnoses for 31 possible comorbidities and 16
possible infections, in addition to biomarkers and demographic information.

The FEV1 biomarker (a measure of lung function) is the main measure of illness severity in CF patients.

