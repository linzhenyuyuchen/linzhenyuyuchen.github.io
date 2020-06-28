---
layout:     post
title:      Multimodal Attentional Neural Networks for Diagnosis Prediction
subtitle:   Multimodal Attentional Neural Networks for Diagnosis Prediction
date:       2020-05-23
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - KDD
---

# 摘要

结合文本记录和医学代码进行多模态的数据融合

# 数据

包含病人医学代码和文本记录的连续记录

![](/img/808.png)

# MNN 结构

![](/img/3934.png)

# 特征提取

## Medical code

离散的医学代码通常用二值特征表示为multi-hot向量，使用以下公式将其转换为dense vector:

![](/img/171.png)

## Clinical text

临床文本特征提取包括两部分，一是是基于原始临床文本的纯文本特征提取信息，另一种是医学上下文感知文本特征嵌入，可将文本数据与医学代码相关联补偿离散的医疗代码

## Pure Text Feature Extraction

note j， sentence u， word 1:n ，word dimension r

对于句子s表示，我们通过使用具有不同窗口大小的多个过滤器的卷积神经网络来捕获单词级特征的不同粒度。

对于文档d表示，为了利用在不同情况下不同的重要性，我们使用具有注意力机制的双向递归神经网络将隐藏状态整合到最终文档表示中。

对于单词表示w，使用预训练的word embedding对文本中的单词进行转换，得到num_words X r_dimensional的矩阵：

![](/img/4627.png)

然后使用卷积神经网络学习句子表征

对于每个句子表示Suj，我们使用双向GRU来学习两个句子方向的文本信息：

![](/img/45053.png)

对于每个文档表示中的所有句子，我们使用注意力机制来获取重要的信息：

![](/img/15171.png)


## Medical Context Aware Text Feature Embedding

根据住院状态，医生为病人标注不同的医疗代码。

![](/img/45528.png)

## 深层特征混合

### 显式特征

![](/img/6047.png)

### 隐式特征

先将 textual feature representation τj and medical code feature representation πj 进行concatenate，然后使用DNN提取隐含的相互特征。

![](/img/46068.png)

# 注意力双向RNN (BiRNN)

![](/img/3346278.png)





