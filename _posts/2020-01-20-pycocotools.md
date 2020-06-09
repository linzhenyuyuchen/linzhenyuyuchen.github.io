---
layout:     post
title:      pycocotools
subtitle:   pycocotools
date:       2020-01-20
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - pycocotools
---

# pycocotools

`https://pypi.org/project/pycocotools/#files`

```python
# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".
```


```python
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random
```



```python
# 实例化对象
annFile = '/data2/lzy/cocojson_train_1.json'
print(f'Annotation file: {annFile}')
coco=COCO(annFile)
```



```python
# 利用getCatIds函数获取某个类别对应的ID，
ids = coco.getCatIds('labelname')[0]
print(f'"labelname" 对应的序号: {ids}')

# 利用loadCats获取序号对应的类别名
cats = coco.loadCats(1)
print(f'"1" 对应的类别名称: {cats}')

```



```python
# 获取包含0的所有图片
imgIds = coco.getImgIds(catIds=[0])
print(f'包含0的图片共有：{len(imgIds)}张')
```


```python
# 获取包含dog的所有图片
id = coco.getCatIds(['dog'])[0]
imgIds = coco.catToImgs[id]
print(f'包含dog的图片共有：{len(imgIds)}张, 分别是：')
print(imgIds)
```


```python
# 展示图片信息

imgId = imgIds[10]

imgInfo = coco.loadImgs(imgId)[0]
print(f'图像{imgId}的信息如下：\n{imgInfo}')

imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name'])
im = cv2.imread(imPath)
plt.axis('off')
plt.imshow(im)
plt.show()
```


```python
# 显示原图
plt.imshow(im); plt.axis('off')

# 获取该图像对应的anns的Id
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
print(f'图像{imgInfo["id"]}包含{len(annIds)}个ann对象，分别是:\n{annIds}')
anns = coco.loadAnns(annIds)

# 显示标注
coco.showAnns(anns)
```


```python
# 显示mask图
print(f'ann{annIds[3]}对应的mask如下：')
mask = coco.annToMask(anns[3])
plt.imshow(mask); plt.axis('off')
```
