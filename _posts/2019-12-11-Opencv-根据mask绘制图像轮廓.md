---
layout:     post
title:      Opencv 根据标注信息绘制图像轮廓
subtitle:   轮廓发现 findContours
date:       2019-12-11
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - opencv
    - 医学影像
---

# 轮廓发现 findContours

contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contours：`Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).`

hierarchy：`Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology.`

mode：`Contour retrieval mode, see cv::RetrievalModes`

method：`Contour approximation method, see cv::ContourApproximationModes`

# 轮廓绘制 drawContours

img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

contourIdx：`Parameter indicating a contour to draw. If it is negative, all the contours are drawn.`

color：`Color of the contours. RGB (#,#,#)`

thickness：`Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.` (粗细or填充)

# 根据标注信息绘制病灶区域

```
import cv2
from matplotlib import pylab

im = cv2.imread("/img/pat08_im1_ACHD.png")
msk = cv2.imread("/img/pat08_im1_.bmp")

imgray = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

pylab.imshow(img[:,:,::-1])
```

`original image :`

![](/img/pat08_im1_ACHD.png)

`mask :`
![](/img/pat08_im1_.bmp)

`combine :`

![](/img/pat08_combine.png)

