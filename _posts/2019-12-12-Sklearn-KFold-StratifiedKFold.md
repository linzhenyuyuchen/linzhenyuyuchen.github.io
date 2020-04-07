---
layout:     post
title:      Sklearn KFold StratifiedKFold
subtitle:   StratifiedKFold和KFold生成交叉验证数据集的区别
date:       2019-12-12
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Sklearn
    - 深度学习
---

# KFold

KFold交叉采样：将训练/测试数据集划分n_splits个互斥子集，每次只用其中一个子集当做测试集，剩下的（n_splits-1）作为训练集，进行n_splits次实验并得到n_splits个结果。

```
sklearn.model_selection.KFold(n_splits=3,shuffle=False,random_state=None)
```

n_splits：表示将数据划分几等份
shuffle：在每次划分时，是否进行洗牌
若为False，其效果相当于random_state为整数(含零)，每次划分的结果相同
若为True，每次划分的结果不一样，表示经过洗牌，随机取样的
random_state：随机种子数，当设定值(一般为0)后可方便调参，因为每次生成的数据集相同

# StratifiedKFold

分层采样，用于交叉验证：与KFold最大的差异在于，StratifiedKFold方法是根据标签中不同类别占比来进行拆分数据的。

```
sklearn.model_selection.StratifiedKFold(n_splits=3,shuffle=False,random_state=None)
```
