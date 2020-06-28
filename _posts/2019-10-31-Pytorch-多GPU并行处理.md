---
layout:     post
title:      Pytorch 多GPU并行处理
subtitle:   单主机多GPUs训练
date:       2019-10-31
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pytorch
---

# DataParallel

>DataParallel 使用起来非常方便，只需要用 DataParallel 包装模型，再设置一些参数即可。需要定义的参数包括：参与训练的 GPU 有哪些，device_ids=gpus；用于汇总梯度的 GPU 是哪个，output_device=gpus[0] 。DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总

```python
import torch

# batch size = gpu数目的倍数
# 图片尺寸可以适当缩小

model = ###

model = torch.nn.DataParallel(model)
# or
model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
```

# DistributedParallel

>分布式并行训练,与 DataParallel 的单进程控制多 GPU 不同，在 distributed 的帮助下，我们只需要编写一份代码，torch 就会自动将其分配给n个进程，分别在n个 GPU 上运行。

```python
import torch
import torch.distributed as dist

gpu = -1

dist.init_process_group(backend='nccl')
torch.cuda.set_device(gpu)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ###
model.cuda(gpu)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

```

在使用时，调用 torch.distributed.launch 启动器启动：

`python -m torch.distributed.launch --nproc_per_node=GPU数量 main.py`

# apex

`https://github.com/NVIDIA/apex`

>Apex 是 NVIDIA 开源的用于混合精度训练和分布式训练库。Apex对混合精度训练的过程进行了封装，从而大幅度降低显存占用，节约运算时间。此外，Apex 也提供了对分布式训练的封装，针对 NVIDIA 的 NCCL 通信库进行了优化。

## Install

`git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## 混合精度训练

在`混合精度训练`上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。

```
from apex import amp

# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
```

## 分布式训练

在`分布式训练`的封装上，Apex 在胶水层的改动并不大，主要是优化了 NCCL 的通信。因此，大部分代码仍与 torch.distributed 保持一致。使用的时候只需要将 torch.nn.parallel.DistributedDataParallel 替换为 apex.parallel.DistributedDataParallel 用于包装模型。

```
from apex.parallel import DistributedDataParallel

model = DistributedDataParallel(model)
# torch.distributed
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

```
在正向传播计算 loss 时，Apex 需要使用 amp.scale_loss 包装，用于根据 loss 值自动对精度进行缩放：

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
   scaled_loss.backward()
```

汇总代码：

```
import torch
import torch.distributed as dist

from apex.parallel import DistributedDataParallel

gpu = -1

dist.init_process_group(backend='nccl')
torch.cuda.set_device(gpu)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model, optimizer = amp.initialize(model, optimizer, opt_level='01')
model = DistributedDataParallel(model, device_ids=[gpu])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
         scaled_loss.backward()
      optimizer.step()
```

在使用时，调用 torch.distributed.launch 启动器启动：

`UDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py`


[Reference Github](https://github.com/tczhangzhi/pytorch-distributed)