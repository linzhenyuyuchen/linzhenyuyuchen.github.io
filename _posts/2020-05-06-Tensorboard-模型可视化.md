---
layout:     post
title:      Tensorboard 模型可视化
subtitle:   模型可视化
date:       2020-05-06
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Tensorboard
    - Pytorch
---

# Tensorboard模型可视化

```python
model = xxx()

x=torch.autograd.Variable(torch.rand(1,3,224,224)) 
writer=SummaryWriter("./logs/")  #定义一个tensorboardX的写对象 
writer.add_graph(model,x,verbose=True) 
```
