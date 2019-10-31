---
layout:     post
title:      Torchvision 模型微调
subtitle:   Pytorch 迁移学习：微调和特征提取Torchvision模型
date:       2019-10-14
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Pytorch
    - Torchvision
---

# 迁移学习

[Reference](https://segmentfault.com/p/1210000018024703/read)

**步骤**

- 初始化预训练模型
- `重组最后一层`，使其具有与新数据集类别数相同的输出数
- 为优化算法定义想要在训练期间更新的`参数`
- 运行训练步骤

```
import torchvision
model = torchvision.models.densenet(pretrained=True)
```

## 微调

>在微调中，从一个预训练模型开始，然后为新任务更新所有的模型参数，实质上就是重新训练整个模型。

## 特征提取

>在特征提取中，从预训练模型开始，只更新产生预测的最后一层的权重。它被称为特征提取是因为使用预训练的CNN作为固定的特征提取器，并且仅改变输出层。

### 设置模型参数的.requires_grad属性

`feature_extract` 是定义我们选择微调还是特征提取的布尔值。 如果`feature_extract = False`，将微调模型，并更新所有模型参数。 如果`feature_extract = True`，则仅更新最后一层的参数，其他参数保持不变。

默认情况下，当我们加载一个预训练模型时，所有参数都是 `.requires_grad = True`，如果我们从头开始训练或微调，这种设置就没问题。 但是，如果我们要运行特征提取并且只想为新初始化的层计算梯度，那么我们希望所有其他参数不需要梯度变化。

```python
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
```

### 创建只更新所需参数的优化器

现在模型结构是正确的，微调和特征提取的最后一步是创建一个只更新所需参数的优化器。 回想一下，在加载预训练模型之后，但在重塑之前，如果`feature_extract = True`，我们手动将所有参数的`.requires_grad = False`。然后重新初始化默认为`.requires_grad = True`的网络层参数。所以现在我们知道应该优化所有具有 `.requires_grad = True`的参数。接下来，我们列出这些参数并将此列表输入到SGD算法构造器。

要验证这一点，可以查看要学习的参数。微调时，此列表应该很长并包含所有模型参数。但是，当进行特征提取时，此列表应该很短并且仅包括重塑层的权重和偏差。

```python
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
```
