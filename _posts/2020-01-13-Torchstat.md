---
layout:     post
title:      Torchstat
subtitle:   自动计算模型参数量、FLOPs、乘加数以及所需内存等数据
date:       2020-01-13
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - 模型参数
---

# Torchstat

`https://github.com/Swall0w/torchstat`

安装

```
pip install torchstat
```

python中使用


```python
from torchstat import stat
import torchvision.models as models

model = models.resnet18()
stat(model, (3, 224, 224))
```

## OTHER

```python
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
```
