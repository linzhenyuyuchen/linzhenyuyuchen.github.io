---
layout:     post
title:      Google Colab & Drive / Kaggle
subtitle:   Google Colab 读取/存储 云盘内的文件
date:       2019-09-20
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Colab
    - Kaggle
---

# 挂载云盘

```python
import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive"

os.chdir(path)
os.listdir(path)
```

点击输出的连接并登录Google账号，复制验证码到输出位置

# 从drive读写文件

```python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open("1.jpg")
print(img.mode)
plt.figure("img")
plt.imshow(img)
plt.show()

```

# Kaggle数据集下载

**在Kaggle my account 下载new API Token**

**复制kaggle.json内容到以下命令中**

```python
!pip install -U -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"lzy","key":"a9c058"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

**下载数据集**

```python
!kaggle datasets download -d taindow/rsna-train-stage-1-images-png-224x
```


**Trick**

- tensorflow 采用 `TPU` 训练

- 独立出一个页面把你要配的环境的代码都写在该页面下，下次打开只需要运行所有单元格就可以完成环境配置

- 挂载只有12个小时，也就是说12小时之后你就需要重现挂载一次，所以就需要我们在进行模型训练的时候记得要加上checkpoint

**云盘超时**
为什么 `drive.mount()` 有时会失败，并提示“超时”？为什么在通过 drive.mount() 装载的文件夹中执行的 `I/O` 操作有时会失败？

当文件夹中的文件或子文件夹数量太多时，Google 云端硬盘操作可能会出现超时问题。如果有成千上万个项目直接包含在“我的云端硬盘”顶级文件夹中，那么装载该云端硬盘可能会超时。重复尝试可能最终会取得成功，因为在超时之前，失败的尝试会在本地缓存部分状态。如果您遇到此问题，`请尝试将直接包含在“我的云端硬盘”中的文件和文件夹移至子文件夹`。如果在 drive.mount() 运行成功后从其他文件夹中读取数据，可能会出现类似问题。访问含有许多项目的任何文件夹中的项目都可能会导致错误，例如 OSError: [Errno 5] Input/output error (python 3) 或 IOError: [Errno 5] Input/output error (python 2)。同样，您只需将直接包含的项目移至子文件夹中，便可解决此问题。

请注意，通过将文件或子文件夹移入回收站来将其删除可能还不足够；如果执行上述操作后问题仍未解决，请务必再清空回收站

# 上传文件到本地

```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
```

# 下载文件到本地

```python
from google.colab import files
files.download('./weight.best.hdf5')
```

# 设置TPU加速

[tensorflow.keras](https://tensorflow.google.cn/guide/keras?hl=zh-cn)

[Reference](https://tensorflow.google.cn/guide/using_tpu?hl=zh-cn)

[Cloud TPU](https://cloud.google.com/tpu/docs/?hl=zh-cn)

1. notebook设置为TPU模式

2. 初始化TPU环境

```python
# 准备TPU环境
import os
# This address identifies the TPU we'll use when configuring TensorFlow.
TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tf.logging.set_verbosity(tf.logging.INFO)
```

3. 将keras模型转化为TPU模型

以keras为例，假设已经定义好了一个模型model，需要将其转换为TPU类型的模型

```python
model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
```

**tricks**

- 如果要训练的 batch size 过大，可以慢慢减小 batch size，直到它适合 TPU 内存，只需确保总的 batch size 为 64 的倍数（`每个核心的 batch size 大小应为 8 的倍数`，这是为了使输入样本在 8 个 TPU 核心上均匀分布并运行）

- 使用较大的 batch size 进行训练也同样有价值：通常可以稳定地提高优化器的学习率，以实现更快的收敛

4. 使用标准的 Keras 方法来训练并保存权重

```python
history = tpu_model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128 * 8,
                        validation_split=0.2)
tpu_model.save_weights('./tpu_model.h5', overwrite=True)
tpu_model.evaluate(x_test, y_test, batch_size=128 * 8)
```

```
WARNING:tensorflow:Keras support is now deprecated in support of TPU Strategy. Please follow the distribution strategy guide on tensorflow.org to migrate to the 2.0 supported version.
INFO:tensorflow:Querying Tensorflow master (grpc://10.121.251.178:8470) for TPU system metadata.
INFO:tensorflow:Found TPU system:
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, -1, 1117769489232922289)

...

INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 17179869184, 10216600743192387167)
```