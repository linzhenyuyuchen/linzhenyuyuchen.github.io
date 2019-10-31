---
layout:     post
title:      ResNext WSL Pytorch Hub
subtitle:   ResNext models trained with billion scale weakly-supervised data
date:       2019-10-10
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - ResNext
---

[Reference](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/)

![](/img/wsl-image.png)

# Load

```python
import torch
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
# or
# model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
# or
# model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
# or
#model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
model.eval()
```

# Application

All pre-trained models expect input images `normalized in the same way`, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be `at least 224`.

The images have to be loaded in to a range of `[0, 1]` and then normalized using mean = `[0.485, 0.456, 0.406]` and std = `[0.229, 0.224, 0.225]`.

```python
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))
```

