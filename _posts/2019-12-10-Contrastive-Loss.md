---
layout:     post
title:      Contrastive Loss
subtitle:   对比损失函数
date:       2019-12-10
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 损失函数
    - 神经网络
---

# Contrastive Loss

对比损失函数满足以下准则：
- 近似样本之间的距离越小越好
- 不似样本之间的距离如果小于m，则通过互斥使其距离接近m

![](/img/0200218171521.png)
其中 W 是网络权重；Y是成对标签

如果X1，X2这对样本属于同一个类，Y=0，调整参数最小化X1与X2之间的距离。

如果属于不同类则 Y=1 ,如果X1与X2之间距离大于m，则不做优化（省时省力）；如果 X1 与 X2 之间的距离小于 m, 则增大两者距离到m。

Dw 是 X1 与 X2 在潜变量空间的欧几里德距离
