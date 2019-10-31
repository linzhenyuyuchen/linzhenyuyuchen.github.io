---
layout:     post
title:      Some Tricks for beginner
subtitle:   about training model at the beginning
date:       2019-10-03
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - 深度学习
---

[Reference](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110671#latest-638850)

**At the begenning:**

image size : smaller firstly

batch size : depend on image size eg. 224x224->64

full data : don't use full data eg. 1k positive and 1k negatice subset

model : smaller model of certain model eg. efficientnet b0 rather than efficientnet b4

**Improvements**

Image augmentation

Different learning rate and learning rate schedule

Increased input size

Train longer

Add dense layer and regularization (e.g. keras.layers.Dropout() before the output layer)

Adding some optimal windowing
