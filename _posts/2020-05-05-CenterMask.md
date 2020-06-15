---
layout:     post
title:      CenterMask
subtitle:   Real-Time Anchor-Free Instance Segmentation
date:       2020-05-05
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 实例分割
    - 目标检测
---

[arxiv](https://arxiv.org/abs/1911.06667)

# 创新点

![](/img/202006032.png)

- a novel spatial attention-guided mask (SAG-Mask) branch to anchor-free one stage object detector

## SAG-Mask

在分割任务中在预测mask的卷积层中加入注意力机制 SAM

- Backbone : VoVNetV2,  with two effective strategies: (1) add residual connection into each OSA module to ease optimization for alleviating the optimization problem of larger VoVNet [19] and (2)effective Squeeze-Excitation (eSE) dealing with the channelinformation  loss  problem  of  original  SE.

## OSA

> One-Shot Aggregation

![](/img/20200404220720207.png)

DenseNet在目标检测任务上表现很好。因为它通过聚合不同receptive field特征层的方式，保留了中间特征层的信息。它通过feature reuse 使得模型的大小和flops大大降低，但是，实验证明，DenseNet backbone更加耗时也增加了能耗：dense connection架构使得输入channel线性递增，导致了更多的内存访问消耗，进而导致更多的计算消耗和能耗。

在OSA module中，每一层产生两种连接，一种是通过conv和下一层连接，产生receptive field 更大的feature map，另一种是和最后的输出层相连，以聚合足够好的特征。

## eSE

Squeeze-Excitation  (SE)  [13]  channel  attention module 减少了通道数，这样虽然减少了计算成本，但是造成了通道信息损失

提出eSE模块，用channel-wise global average pooling保留通道维度，然后接1个C维度的全连接层

计算公式：

![](/img/202005.png)

# 关键点

one stage / anchor free / attention module

# 组成部分

(1) backbone for feature extraction

## VoVNetV2

在VoVNet基础上增加了 residual  connection 和 eSE注意力模块

![](/img/vmm11.png)

(2) FCOS  [33] ：detection  head

> an anchor-free and proposal-free object detection in a per-pixel prediction manner as like FCN

(3) mask head ：The  procedure  of masking  objects  is  composed  of detecting objects from the FCOS [33] box head and then predicting segmentation masks inside the cropped regionsin a per-pixel manner

## Adaptive RoI Assignment Function

根据RoI scales对RoIs映射到不同层次的feature map上，大尺度的roi映射到高层的feature level

对应映射关系可计算：

![](/img/202006031.png)

# 安装


```
pip install cython
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install -i https://pypi.douban.com/simple/ pyyaml==5.1.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

git clone https://github.com/youngwanLEE/centermask2.git

```

修改数据集地址: `/usr/local/lib/python3.6/dist-packages/detectron2/data/datasets/builtin.py`

```python
# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")

改为

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "/xxx/xxx/")
```

Config:

`/home/centermask2/configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml`


Train:

`CUDA_VISIBLE_DEVICES=2,3 python train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 2`

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 4`

Test:

`python train_net.py --config-file "configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml" --num-gpus 1 --eval-only MODEL.WEIGHTS output/centermask/CenterMask-V-39-ms-3x/model_0019999.pth`


