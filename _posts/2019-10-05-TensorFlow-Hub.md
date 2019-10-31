---
layout:     post
title:      TensorFlow Hub
subtitle:   TensorFlow Hub 是一个针对可重复使用的机器学习模块的库
date:       2019-10-05
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - TensorFlow
---

[Reference](https://tensorflow.google.cn/hub/)

TensorFlow Hub 是一个库，用于发布、发现和使用机器学习模型中可重复利用的部分。模块是一个独立的 TensorFlow 图部分，其中包含`权重`和`资源`，可以在一个进程中供不同任务重复使用（称为`迁移学习`）。迁移学习可以：

- 使用较小的数据集训练模型，
- 改善泛化效果，以及
- 加快训练速度。

```
!pip install "tensorflow_hub==0.4.0"
!pip install "tf-nightly"
```

```python
  import tensorflow as tf
  import tensorflow_hub as hub

  tf.enable_eager_execution()

  module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
  embed = hub.KerasLayer(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])
  print(embeddings.shape)  #(3,128)
```