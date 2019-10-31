---
layout:     post
title:      Pytorch cuda out of memory
subtitle:   显存不足分析和解决
date:       2019-10-12
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pytorch
---

# 分析

`out of memory` 的原因是显存装不下那么多模型权重和中间变量


# 解决方法

不要一次性读入全部数据

及时清空中间变量 del var..

total_loss= float(loss)  即仅将loss的值传给total_loss

torch.cuda.empty_cache()  可清理缓存，应该是最有效最便捷的

减少batch size

优化代码

减少输入图像的尺寸

多使用下采样，池化层

在预测时禁用梯度计算，Context-manager that disabled gradient calculation

```python
with torch.no_grad():
    output = net(input,inputcoord)
```

# 数据量

- float32 单精度浮点型
- int32 整型

- 1 G = 1000 MB
- 1 M = 1000 KB
- 1 K = 1000 Byte
- 1 B = 8 bit

假设有一张RGB三通道图片，尺寸为512x512，数据类型为单精度浮点型，那么这张图所占显存大小为512x512x3x4B

# 显存占用

占用显存的层一般是：

- 卷积层，通常的conv2d
- 全连接层，也就是Linear层
- BatchNorm层
- Embedding层

而不占用显存的则是：

- 刚才说到的激活层Relu等
- 池化层
- Dropout层

具体计算方式：

- Conv2d(Cin, Cout, K): 参数数目：Cin × Cout × K × K
- Linear(M->N): 参数数目：M×N
- BatchNorm(N): 参数数目： 2N
- Embedding(N,W): 参数数目： N × W

额外的显存：

- 模型中的参数(卷积层或其他有参数的层)
- 模型在计算时产生的中间参数(也就是输入图像在计算时每一层产生的输入和输出)
- backward的时候产生的额外的中间参数
- 优化器在优化时产生的额外的模型参数
