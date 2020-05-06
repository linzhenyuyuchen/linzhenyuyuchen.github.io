---
layout:     post
title:      MMDetection 组件
subtitle:   组件
date:       2020-01-23
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - MMDetection
---

[Reference](https://github.com/open-mmlab/mmdetection)

[Doc](https://mmdetection.readthedocs.io/)

# MMDetection

## 组件

1. 骨干网：通常是FCN网络，用于提取特征图，例如ResNet，MobileNet。

2. 脖子：骨干和头部之间的组件，例如FPN，PAFPN。

3. head：用于特定任务的组件，例如bbox预测和掩码预测。

4. roi提取器：用于从要素图（例如RoI Align）中提取RoI要素的部分。

## 开发新的组件

### backbone

创建一个新文件`mmdet/models/backbones/mobilenet.py`

```python
import torch.nn as nn

from ..registry import BACKBONES

@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

在中导入模块`mmdet/models/backbones/__init__.py`

```python
from .mobilenet import MobileNet
```

在配置文件中使用它

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```


## 添加模型部件

创建一个新文件`mmdet/models/necks/pafpn.py`

```python
from ..registry import NECKS

@NECKS.register
class PAFPN(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    def forward(self, inputs):
        # implementation is ignored
        pass
```

导入模块`mmdet/models/necks/__init__.py`

```python
from .pafpn import PAFPN
```

修改配置文件

```python
neck=dict(
    type='PAFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

## 添加两阶段检测器

`mmdet/models/detectors/two_stage.py`

```python
extract_feat() # 给定形状为（n，c，h，w）的图像批处理，提取特征图。

forward_train() # 训练模式的前进方法

simple_test() # 无扩展的单规模测试

aug_test() # 增强测试（多尺度，翻转等）
```
