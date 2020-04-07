---
layout:     post
title:      Pytorch Tensorboard
subtitle:   B
date:       2019-11-04
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Tensorboard
    - Pytorch
---

# Tensorboard

[Source](https://pytorch.org/docs/stable/tensorboard.html)

`https://blog.csdn.net/bigbennyguo/article/details/87956434`

`https://github.com/fusimeng/framework_benchmark`

`https://shenxiaohai.me/2018/10/23/pytorch-tutorial-TensorBoard/`

## Install

首先要安装好tensorflow

```
pip install tensorflow
pip install tensorflow-gpu
```

```
pip install tensorboard
conda install tensorboard
```


# Application


## scalar

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir= 'tests')

for epoch in range(199):
    writer.add_scalar('scalar/test',np.random.rand(),epoch)
    writer.add_scalars('scalar/scalars_test',{'xsinx':epoch*np.sin(epoch),'xcsox':epoch*np.cos(epoch)},epoch)
writer.close()
```

```
!tensorboard  --logdir='tests' --bind_all
```

---

```python
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir= 'train')

for epoch in range(epoch_n):
    ####
    writer.add_Scalar('train',loss,epoch)
    ###

writer.add_graph(model,(inputs,))

predicted = model(x_test)

plt.plot(x_train,y_train,'ro',label = 'original data')
plt.plot(x_test,preditcted,label = 'Fitted line')

plt.legend()
plt.show()

writer.close()

```

## graph

```python
import torch
import torchvision
model = torchvision.models.resnet18()
#print (model)
inputs = torch.rand(64,3,7,7)

with SummaryWriter(log_dir='densenet',comment='densenet') as w:
    w.add_graph(model,inputs)

```

```
!tensorboard  --logdir='densenet' --bind_all
```

# Run

传递`--bind_all`参数支持外部访问

```
tensorboard --logdir=runs --bind_all
```


# Error

非root用户在启动时发生写数据权限不足 `PermissionDeniedError`

`Solution:` 找到对于py文件修改tmp相对路径为子用户下的可写绝对路径
