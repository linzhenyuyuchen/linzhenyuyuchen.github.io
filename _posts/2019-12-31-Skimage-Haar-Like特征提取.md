---
layout:     post
title:      Skimage Haar Like特征提取
subtitle:   Haar特征 / 矩形特征
date:       2019-12-31
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Skimage
    - 特征提取
---

# Haar Like

Haar特征本身并不复杂，就是用图中黑色矩形所有像素值的和减去白色矩形所有像素值的和。

![](/img/20200405594.png)

特定的窗口在图像中以步长为1滑动遍历图像。

## 积分图

积分图是（Integral Image）类似动态规划的方法，主要的思想是将图像从起点开始到各个点所形成的矩形区域像素之存在数组中，当要计算某个区域的像素和时可以直接从数组中索引，不需要重新计算这个区域的像素和，从而加快了计算。

# DEMO

`https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.haar_like_feature`

```python
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
def haarlike(img_path):
    feature_types = ['type-2-x', 'type-2-y']
    img = cv2.imread(img_path)
    img=cv2.cvtColor(cv2.resize(img,(25,25),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],feature_type=feature_types,feature_coord=None)
```
