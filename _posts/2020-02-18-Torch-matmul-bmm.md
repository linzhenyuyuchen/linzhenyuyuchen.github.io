---
layout:     post
title:      Torch matmul bmm
subtitle:   批量矩阵乘法 torch.matmul torch.bmm
date:       2020-02-10
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Pytorch
---

# 批量矩阵乘法

```python
import torch

# a=(2*3*4)
a = torch.rand((2,3,4))
# b=(2,2,4)
b = torch.rand((2,2,4))

b=b.transpose(1, 2)


# res=(2,3,2),对于a*b，是第一维度不变，而后[3,4] x [4,2]=[3,2]
#res[0,:]=a[0,:] x b[0,;];   res[1,:]=a[1,:] x b[1,;] 其中x表示矩阵乘法
res = torch.matmul(a, b)  # 维度res=[2,3,2]
print(res)  # res2的值等于res


res2 = torch.bmm(a, b)  # 维度res2=[2,3,2]
print(res2)

print((res == res2).all())

"""
tensor(True)
"""
```
