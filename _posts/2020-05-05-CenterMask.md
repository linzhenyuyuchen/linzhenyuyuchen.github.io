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

注意方法已经广泛应用于目标检测，因为它有助于聚焦重要的特征，抑制不必要的特征

受空间注意机制的启发，提出在分割任务中在预测mask的卷积层中加入注意力机制 SAM，进行聚焦有意义的像素和抑制不必要像素。

![](/img/a6.png)

对于从RoiAlign得到的14x14xC的输入进行多层卷积，然后经过SAM，即先在C channel维度上分别进行avg和max pooling，然后将pooling结果串联，然后用激活函数为sigmoid的3*3卷积核卷积。然后在原始输入Xi上应用attention，即用Asag和原始输入Xi进行像素点的乘法，得到14x14xC的输出，再用2*2的decon反卷积来上采样得到28x28xC的输出，最后使用1*1的卷积核预测类别的的mask。

## OSA

> One-Shot Aggregation

![](/img/20200404220720207.png)

DenseNet在目标检测任务上表现很好。因为它通过聚合不同receptive field特征层的方式，保留了中间特征层的信息。它通过feature reuse 使得模型的大小和flops大大降低，但是，实验证明，DenseNet backbone更加耗时也增加了能耗：dense connection架构使得输入channel线性递增，导致了更多的内存访问消耗，进而导致更多的计算消耗和能耗。

在OSA module中，每一层产生两种连接，一种是通过conv和下一层连接，产生receptive field 更大的feature map，另一种是和最后的输出层相连，以聚合足够好的特征。

## Residual connection

VovNetV1-99随着卷积层的增加，梯度的反向传播逐渐变难，基于resnet的方法，添加identity mapping 到OSA模块中，将输入路径连接到OSA的尾部使得梯度能够更好的反向传播

![](/img/11643.png)

## eSE

Squeeze-Excitation  (SE)  [13]  先用全局平均池化压缩空间依赖性，得到通道的特定描述子，然后用全连接层将通道数减少至C/r,r是缩小比例,再接一个全连接层重新扩展通道数为C，channel  attention module 减少了通道数，这样虽然减少了计算成本，但是造成了通道信息损失

提出eSE模块，用channel-wise global average pooling保留通道维度，然后接1个C维度的全连接层

计算公式：

![](/img/202005.png)

# 关键点

one stage / anchor free / attention module

# 组成部分

(1) backbone for feature extraction

Backbone : VoVNetV2,  with two effective strategies: (1) add residual connection into each OSA module to ease optimization for alleviating the optimization problem of larger VoVNet [19] and (2)effective Squeeze-Excitation (eSE) dealing with the channelinformation  loss  problem  of  original  SE.

## VoVNetV2

在VoVNet基础上增加了 residual  connection 和 eSE注意力模块

![](/img/vmm11.png)

(2) FCOS  [33] ：detection  head

> an anchor-free and proposal-free object detection in a per-pixel prediction manner as like FCN

(3) mask head ：The  procedure  of masking  objects  is  composed  of detecting objects from the FCOS [33] box head and then predicting segmentation masks inside the cropped regionsin a per-pixel manner

## Adaptive RoI Assignment Function

Maskrcnn中的RoiAlign没有考虑到输入尺寸大小，会导致低尺寸的roi被分配到高层的feature level上，所以提出根据RoI scales对RoIs映射到不同层次的feature map上，大尺度的roi映射到高层的feature level

对应映射关系可计算：

![](/img/202006031.png)

# 消融实验

## Adaptive RoI Assignment Function

与公式1相比，提出的公式2考虑到输入规模，改进了0.4%的APmask

![](/img/a81.png)

## Spatial Attention Guided Mask

FCOS baseline的APbox只有37.8%,运行时间是57ms

加入maskrcnn中的mask head可以改进0.5%的APbox，得到33.4%APmask

使用Adaptive RoI Assignment Function可以再提升0.4%的APmask和APbox

加入Spatial Attention Guided Mask可以再提升0.2%

在SAG-mask中使用mask scoring对预测的mask IoU进行分数校准，则可以对APmask提升0.7%。因为mask scoring在evaluation阶段调整了mask结果的顺序，所以它无法提升APbox

## Feature selection

![](/img/a037.png)

FCOS采用的是P3~P7的特征，但是由于P7的feature map过于小，以至于无法提取精细特征，所以本文采用的是P3~P5的特征，也就是说对于mask预测来说分辨率越高越好

## VoVNetV2

![](/img/a4581.png)
对于残差连接来说，VoVNet-99 对AP值的提升比VoVNet-39/57大是因为更多的OSA模块可以产生更多的残差连接效果，从而缓解梯度问题

对于SE和eSE来说，因为损失了通道信息，所以SE要么对效果没有提升，要么反而使效果更差。而eSE保留了通道信息，使得APmask和APbox都有所提升

## Comparison to other backbones

从表中可以得出VoVNetV2是准确率和速度均衡的backbone

尽管VoVNetV1-39 已经超过了其它backbone, VoVNetV2-39 比ResNet-50/HRNet-W18在速度上分别提升了1.2%/2.6%. APbox分别提升了1.5%/3.3%.

## Comparison with state-of-the-arts methods

CenterMask on V100 GPU

YOLACT on Titan Xp GPU

Dataset : COCO test-dev2017

Mask R-CNN, RetinaMask 和 CenterMask 部署在 maskrcnn benchmark

CenterMask* 部署在 Detectron2

![](/img/20200617.png)

ResNet-101作为backbone CenterMask(APmask, APbox) and speed的效果比其他的更好，特别和有相同结构的RetinaMask(i.g.,
one-stage detector + mask branch)相比, CenterMask 在APmask上有3.6%的提升，而且与the dense sliding window method,
TensorMask 相比只要用不到一半的epoch就能达到1.2% APmask的提升，5倍speed的提升。

CenterMask with VoVNetV2-99 是首个能够达到40%的APmask提升且FPS大于10.

另外，将centermask部署在Detectron2能够得到0.8% APmask和0.7% APbox的提升

最后，YOLACT是代表性的实时实例分割方法，本文采用四种backbone(e.g., MobileNetV2,VoVNetV2-19,VoVNetV2-39, and ResNet-50)对centermask进行实验，如图，CenterMask-Lite相对于YOLACT，在准确率和速度上都有较好的表现。

---

ResNet-101作为backbone的Mask R-CNN比CenterMask的小目标的APmask好，猜测是Mask R-CNN采用了更大的feature maps(P2),相对CenterMask采用的P3，它能够目标更精细的空间结构。


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


