---
layout:     post
title:      MMDetection 多尺度训练测试
subtitle:   多尺度训练/测试
date:       2020-02-13
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 目标检测
    - MMDetection
---

# 多尺度训练/测试

> 输入图片的尺寸对检测模型的性能影响相当明显，事实上，多尺度是提升精度最明显的技巧之一。在基础网络部分常常会生成比原图小数十倍的特征图，导致小物体的特征描述不容易被检测网络捕捉。通过输入更大、更多尺寸的图片进行训练，能够在一定程度上提高检测模型对物体大小的鲁棒性，仅在测试阶段引入多尺度，也可享受大尺寸和多尺寸带来的增益。

训练时，预先定义几个固定的尺度，每个epoch或每个batch随机选择一个尺度进行训练。

测试时，生成几个不同尺度的feature map，在不同的feature map上也有不同的尺度。选择最接近某一固定尺寸（即检测头部的输入尺寸）的Region Proposal作为后续的输入。

![](/img/20206d43440w.jpg)

选择单一尺度的方式被Maxout（element-wise max，逐元素取最大）取代：随机选两个相邻尺度，经过Pooling后使用Maxout进行合并。

# MMDetection

修改`train_pipeline`和`test_pipeline`中Resize的img_scale部分即可（换成[(), ()]或者[(), (), ().....]）。

影响：train达到拟合的时间增加、test的时间增加

参数解析：

`train_pipeline`中dict(type='Resize', img_scale=(1228, 921), keep_ratio=True)的keep_ratio解析。

假设原始图像大小为（4912， 3684）

```python
max_long_edge = max(img_scale)
max_short_edge = min(img_scale)
# 取值方式:  大值/长边    小值/短边   谁的比值小   按谁来计算缩放比例
scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
```

scale_factor = min(4912/1228,3684/921) = 4

- 当keep_ratio=True时，img_scale的多尺度最多为两个。假设多尺度为[(2000, 1200), (1333, 800)]，则代表的含义为：首先将图像的短边固定到800到1200范围中的某一个数值假设为1100，那么对应的长边应该是短边的ratio=1.5倍，且长边的取值在1333到2000的范围之内。如果大于2000按照2000计算，小于1300按照1300计算。

简要概括就是： 利用 (小值/短边) 和 (大值/长边) 的比值小的为基准，然后通过图片比例来计算另一边的长度。

```python
scale_w = int(w * float(scale_factor ) + 0.5),
scale_h = int(h * float(scale_factor ) + 0.5)
```

- 当keep_ratio=False时，img_scale的多尺度可以为任意多个。假设多尺度为[(2000, 1200), (1666, 1000),(1333, 800)]，则代表的含义为：随机从三个尺度中选取一个作为图像的尺寸进行训练。

`test_pipeline`中img_scale的尺度可以为任意多个，含义为对测试集进行多尺度测试（可以理解为TTA）

在Resize之后，注意配置文件里还有个Pad操作，将Resize之后的图片Pad成size_divisor=32的倍数，具体逻辑是

```python
pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
```

