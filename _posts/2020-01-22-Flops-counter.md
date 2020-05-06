---
layout:     post
title:      Flops counter
subtitle:   Flops counter for convolutional networks in pytorch framework
date:       2020-01-22
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 模型参数
---

# Flops counter

## Install the latest version

```
pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
```

## Example

```python
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.densenet161()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
    print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
```
