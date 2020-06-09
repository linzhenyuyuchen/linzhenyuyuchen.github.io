---
layout:     post
title:      Deformable Convolutional Networks
subtitle:   Deformable Convolutional Networks
date:       2019-12-04
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
---

[Reference](https://arxiv.org/abs/1703.06211)

[Github](https://github.com/msracver/Deformable-ConvNets)

[Pytorch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch)

# Deformable Convolutional Networks

3×3标准可变形卷积的采样位置的说明

![](/img/20191221154936.png)

(a)标准卷积的规则采样网格（绿点）。

(b)变形的采样位置（深蓝点），在可变形卷积中具有增强偏移（浅蓝色箭头）。

(c)和(d)是(b)的特殊情况，表明变形卷积概括了各种尺度变换、（各向异性）纵横比和旋转。

---

3×3可变形卷积的说明

![](/img/20191221155324.png)

---

3×3可变形RoI池的说明

![](/img/20191221155427.png)

---

3×3可变形PS ROI池的说明

![](/img/20191221155642.png)

---

在标准卷积a中的固定感受野和可变形卷积b中的自适应感受野

![](/img/20191221160002.png)
