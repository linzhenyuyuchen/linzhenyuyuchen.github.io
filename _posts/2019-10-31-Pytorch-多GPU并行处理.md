---
layout:     post
title:      Pytorch 多GPU并行处理
subtitle:   Python
date:       2019-10-31
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pytorch
---

# DataParallel

>单主机多GPUs训练

```python
import torch

model = ###

model = torch.nn.DataParallel(model)

```

# DistributedParallel

>分布式训练
