---
layout:     post
title:      Pytorch How to adjust Learning Rate
subtitle:   several methods to adjust the learning rate based on the number of epochs
date:       2019-10-15
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pytorch
---

[Reference](https://pytorch.org/docs/stable/optim.html)


# torch.optim.lr_scheduler

>根据epoch调整学习率

Learning rate scheduling should be applied after optimizer’s update

```
>>> scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```
## .StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)

Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs. When last_epoch=-1, sets initial lr as lr.

- ptimizer (Optimizer) – Wrapped optimizer.

- step_size (int) – Period of learning rate decay.

- gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.

- last_epoch (int) – The index of last epoch. Default: -1.

## .MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

Set the learning rate of each parameter group to the initial lr decayed by gamma once the number of epoch reaches one of the milestones. When last_epoch=-1, sets initial lr as lr.

- optimizer (Optimizer) – Wrapped optimizer.

- milestones (list) – List of epoch indices. Must be increasing.

- gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.

- last_epoch (int) – The index of last epoch. Default: -1.

```
>>> # Assuming optimizer uses lr = 0.05 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```


