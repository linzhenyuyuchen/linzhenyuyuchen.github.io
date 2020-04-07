---
layout:     post
title:      Skimage HOG特征提取
subtitle:   histogram of oriented gradient 梯度方向直方图特征
date:       2019-12-22
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 机器学习
    - Skimage
    - 特征提取
---

# HOG特征

> 作为提取基于梯度的特征, HOG 采用了统计的方式(直方图)进行提取. 其基本思路是将图像局部的梯度统计特征拼接起来作为总特征. 局部特征在这里指的是将图像划分为多个Block, 每个Block内的特征进行联合以形成最终的特征.


1. 将图像分块: 以Block为单位, 每个Block以一定的步长在图像上滑动, 以此来产生新的Block.

2. Block作为基本的特征提取单位, 在其内部再次进行细分: 将Block 划分为(一般是均匀划分)NxN的小块, 每个小块叫做cell.

3. cell是最基本的统计单元, 在cell内部, 统计每个像素的梯度方向, 并将它们映射到预设的M个方向的bin里面形成直方图.

4. 每个Block内部的所有cell的梯度直方图联合起来并进行归一化处理(L1-norm, L2-Norm, L2-hys-norm, etc), 这样可以使特征具有光照不变性. 光照属于加性噪声, 归一化之后会抵消掉光照变化对特征的影响.

5. 所有Block的特征联合起来, 就是最终的HOG特征


# skimage 函数与参数

```python
from skimage.feature import hog
features = hog(image,  # input image
                  orientations=ori,  # number of bins
                  pixels_per_cell=ppc, # pixel per cell
                  cells_per_block=cpb, # cells per blcok
                  block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                  transform_sqrt = True, # power law compression (also known as gamma correction)
                  feature_vector=True, # flatten the final vectors
                  visualise=False) # return HOG map
```

- image: input image, 输入图像

- orientation: 指定bin的个数. scikit-image 实现的只有无符号方向.
(根据反正切函数的到的角度范围是在-180°~ 180°之间, 无符号是指把 -180°~0°这个范围统一加上180°转换到0°~180°范围内. 有符号是指将-180°~180°转换到0°~360°范围内.)

也就是说把所有的方向都转换为0°~180°内, 然后按照指定的orientation数量划分bins. 比如你选定的orientation= 9, 则bin一共有9个, 每20°一个:

[0°~20°, 20°~40°, 40°~60° 60°~80° 80°~100°, 100°~120°, 120°~140°, 140°~160°, 160°~180°]

- pixels_per_cell : 每个cell的像素数, 是一个tuple类型数据,例如(20,20)

- cell_per_block : 每个BLOCK内有多少个cell, tuple类型, 例如(2,2), 意思是将block均匀划分为2x2的块

- block_norm: block 内部采用的norm类型.

- transform_sqrt: 是否进行 power law compression, 也就是gamma correction. 是一种图像预处理操作, 可以将较暗的区域变亮, 减少阴影和光照变化对图片的影响.

- feature_vector: 将输出转换为一维向量. True or False

- visualise: 是否输出HOG image, (应该是梯度图)

- scikit-image 版的HOG 没有进行cell级别的gaussian 平滑, 原文对cell进行了gamma= 8pix的高斯平滑操作.

# Demo

```python
def hog_image(img_path):
    img = cv2.imread(img_path)
    gimg=cv2.cvtColor(cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    fd,hog_image = hog(gimg,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True)
    #hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,0.02))
    return fd,hog_image
```


