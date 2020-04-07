---
layout:     post
title:      Sklearn KNN分类
subtitle:   KNeighborsClassifier
date:       2019-12-20
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 机器学习
    - Sklearn
---

# K近邻算法

```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
train_img = []
for t in train_dataset:
    img = Image.open(t).convert('RGB')
    img =np.array(img).flatten()
    train_img.append(img)
knn.fit(train_img, train_label)

test_img = []
for t in test_dataset:
    img = Image.open(t).convert('RGB')
    img =np.array(img).flatten()
    test_img.append(img)

pred = knn.predict(test_img)

```