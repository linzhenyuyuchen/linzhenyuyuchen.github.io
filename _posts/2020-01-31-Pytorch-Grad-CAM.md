---
layout:     post
title:      Pytorch Grad-CAM
subtitle:   Grad-CAM
date:       2020-01-31
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Pytorch
---

# Grad-CAM implementation in Pytorch

> What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':

[Github](https://github.com/jacobgil/pytorch-grad-cam)


默认使用的模型是VGG19 from torchvision, 这个模型有features/classifier方法

Usage: 

```
python gradcam.py --image-path <path_to_image>
```

To use with CUDA:

```
python gradcam.py --image-path <path_to_image> --use-cuda
```

# 应用于自己的模型

```python
grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args.use_cuda)
```

feature_module 是想要输出的特征层

target_layer_names 是目标层的最后一层索引

修改ModelOutputs类中__call__方法：

"avgpool" 池化层名字

```python
elif "avgpool" in name.lower():
    x = module(x)
```