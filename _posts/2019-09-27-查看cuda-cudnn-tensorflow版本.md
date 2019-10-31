---
layout:     post
title:      查看cuda cudnn tensorflow版本
subtitle:   json command
date:       2019-09-2
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
---

# 查看cuda cudnn tensorflow版本




1、查看cuda版本

```
cat /usr/local/cuda/version.txt
```

2、查看cudnn版本

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

3、查看tensorflow版本

```
python
import tensorflow as tf
tf.__version__
tf.__path__
```