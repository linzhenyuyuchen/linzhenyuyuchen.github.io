---
layout:     post
title:      Python Albumentations
subtitle:   Python库 - Albumentations 图片数据增强库
date:       2019-10-11
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 1
---

[Reference](https://www.aiuai.cn/aifarm422.html)

# Albumentations 特点

- 基于高度优化的 OpenCV 库实现图像快速数据增强.
- 针对不同图像任务，如分割，检测等，超级简单的 API 接口.
- 易于个性化定制.
- 易于添加到其它框架，比如 PyTorch.

# Install

```
pip install albumentations
# 或
pip install -U git+https://github.com/albu/albumentations
# Kaggle GPU kernels
pip install albumentations > /dev/null
# Conda
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
```

# Transforms

`albumentations.augmentations.transforms`

[API](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#module-albumentations.augmentations.transforms)

`.CenterCrop(height, width, always_apply=False, p=1.0)`

裁剪输入图片的中心部分

**p** (float) – probability of applying the transform. Default: 1.

`.Resize(height, width, interpolation=1, always_apply=False, p=1)`

调整输入图片的尺寸

`.HorizontalFlip(always_apply=False, p=0.5)`

关于Y轴水平翻转

`.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5)`

随机调整亮度和对比度

**brightness_limit** ((float, float) or float) – factor range for changing brightness. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).

**contrast_limit** ((float, float) or float) – factor range for changing contrast. If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).

**brightness_by_max** (Boolean) – If True adjust contrast by image dtype maximum, else adjust contrast by image mean.

`.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)`

随机应用变换：平移、缩放和旋转输入

shift_limit ((float, float) or float) – shift factor range for both height and width. If shift_limit is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).

scale_limit ((float, float) or float) – scaling factor range. If scale_limit is a single float value, the range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).

rotate_limit ((int, int) or int) – rotation range. If rotate_limit is a single int value, the range will be (-rotate_limit, rotate_limit). Default: (-45, 45).

`.MotionBlur(blur_limit=7, always_apply=False, p=0.5)`

运动模糊，在摄像时相机和被摄景物之间有相对运动而造成的图像模糊则称为运动模糊

`.GaussianBlur(blur_limit=7, always_apply=False, p=0.5)`

高斯模糊


# More

[Github](https://github.com/albu/albumentations)

[Example](https://github.com/albu/albumentations/blob/master/notebooks/showcase.ipynb)

[Doc](https://albumentations.readthedocs.io/en/latest/)

# TORCHVISION.TRANSFORMS

`https://pytorch.org/docs/stable/torchvision/transforms.html`

torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

torchvision.transforms.functional.normalize(tensor, mean, std, inplace=False)

**Example** `https://www.kaggle.com/phantomakame/pytorch-fast-ai-top-1-or-good-results`