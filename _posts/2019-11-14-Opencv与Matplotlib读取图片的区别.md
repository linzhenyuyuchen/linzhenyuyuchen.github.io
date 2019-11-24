---
layout:     post
title:      Opencv与Matplotlib读取图片的区别
subtitle:   二者读取图片的顺序不同
date:       2019-11-14
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Opencv
    - Matplotlib

---

# 原因

opencv读取图片默认的顺序是BGR

matplotlib则是RGB

# 解决方法

## Method 1

```
b,g,r = cv2.split(img)

img2 = cv2.merge([r,g,b])

plt.imshow(img2)
plt.show()

```


## Method 2

```
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

```
