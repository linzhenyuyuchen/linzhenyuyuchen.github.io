---
layout:     post
title:      Keypoint Graph Based Bounding Boxes
subtitle:   Multi-scale Cell Instance Segmentation with Keypoint Graph Based Bounding Boxes
date:       2020-05-31
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - MICCAI
---

# Multi-scale Cell Instance Segmentation with Keypoint Graph Based Bounding Boxes

[github](https://github.com/yijingru/KG_Instance_Segmentation)

[arxiv](https://arxiv.org/abs/1907.09140)

## 细胞分割的难点

- 细胞边界的低分辨率

- 背景杂质

- 细胞粘附

- 细胞聚集

## 相关工作

DCAN 提出将细胞轮廓叠在分割结果上，由于touching cell间的模糊轮廓，这会产生over-segmentation

STARDIST 提出使用凸多边形来分隔细胞，但是它假设细胞形状应该是凸的

CosineEmbedding 提出通过对pixel-embeddings进行聚类来分离细胞，但是这会导致单独聚类的细胞产生大量误报false-positive

box-free的实例分割缺乏全局目标特征，box-based的方法先用边界框定位，再在该框内进行单独的细胞分割任务，不仅能学到局部像素信息，还能学到全局目标特征，但是box-based的maskrcnn有正负anchor boxes不平衡的问题

## 
