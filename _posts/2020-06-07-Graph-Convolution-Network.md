---
layout:     post
title:      Graph Convolution Network
subtitle:   图卷积网络 常用库
date:       2020-06-07
author:     LZY
header-img: img/bg-20210718.jpg
catalog: true
tags:
    - GNN
    - 图神经网络
    - 图卷积
---


# Graph Convolution Network



#### 基础版本

[github](https://github.com/tkipf/pygcn)

[dataset](https://relational.fit.cvut.cz/dataset/CORA)

`wget https://data.deepai.org/Cora.zip`

```python
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # input [in_features]
        # adj [num]
        support = torch.mm(input, self.weight)
        # support [out_feature]
        output = torch.spmm(adj, support)
        # output [out_feature]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
```



#### torch-geometric



**依赖**

```shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
```

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
```



**cuda**

[版本选择](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#cudnn-versions-804-805)

```shell
conda install cudatoolkit=11.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/

conda install cudnn=7.4.1 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/

```



