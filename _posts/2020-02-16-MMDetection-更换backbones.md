---
layout:     post
title:      MMDetection 更换backbones
subtitle:   使用新的backbones
date:       2020-02-16
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 目标检测
    - MMDetection
---

# Add new backbones

创建文件 `mmdet/models/backbones/mobilenet.py`

```python
import torch.nn as nn

from ..builder import BACKBONES

@BACKBONES.register_module()
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        super(MobileNet, self).__init__()
        pass

    def forward(self, x):  # should return a tuple 需要返回多尺度特征
        pass
        return [x1,x2,x3,x4]

    def init_weights(self, pretrained=None): # 初始化权重
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

```

在 `mmdet/models/backbones/__init__.py` 添加

```python
from .mobilenet import MobileNet
```

`../configs/configname.py`

```python
model = dict(
    ...
    pretrained='../pretrained_model.pth'
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```