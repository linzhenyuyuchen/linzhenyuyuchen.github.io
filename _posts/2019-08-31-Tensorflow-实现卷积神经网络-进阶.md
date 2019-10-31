---
layout:     post
title:      Tensorflow 实现卷积神经网络 进阶
subtitle:   CIFAR-10数据集
date:       2019-08-31
author:     LZY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Tensorflow
    - 卷积神经网络
    - CNN
---

# Tensorflow 实现卷积神经网络（进阶）

>CIFAR-10数据集包含60000张32x32的彩色图像，其中训练集50000张，测试集10000张。数据标注为10类，每类6000张，每张图片只包含该类物体不包含多类物体。

### 设计CNN模型

下载Tensorflow Models库

```

```

导入常用库，载入Tensorflow Models中自动下载的CIFAR-10类

```
import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
```

初始化weight函数 截断的正态分布初始化权重 并加上L2正则化处理

- 为了解决特征过多导致的过拟合，一般可以通过减少特征或者惩罚不重要特征的权重来缓解问题
- L1 正则 会制造稀疏的特征 大部分无用特征都会置为0
- L2 正则 会使得权重比较平均 让特征权重不过大

标准差stddev 正则系数w1
暂存为losses，后加到全体loss中

```
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
```

下载数据集，解压到默认位置

```
cifar10.maybe_download_and_extract()
```

distorted_inputs 和 inputs 都产生数据，前者对数据进行了Data Augmentation 数据增强

数据增强操作 扩大样本量有助于提高准确率 增大CPU使用时间
- 随机的水平翻转
- 随机剪切一块24x24大小的图片
- 设置随机的亮度及对比度
- 数据的标准化 减去均值除以方差 保证数据的零均值方差为1

```
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)   
```


创建输入数据的placeholder

```
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
```

创建卷积层

- Local Response Normalization(LRN),即局部响应归一化层,LRN函数类似DROPOUT和数据增强作为ReLU激励之后防止数据过拟合而提出的一种处理方法。对局部神经元的活动创造竞争环境，使得响应大的值相对变大，抑制反馈较小的神经元，增强模型的泛化能力。

- LRN 对ReLU这种没有上限边界的激活函数会比较有用，因为它会从附近挑选出响应大的反馈，但不适合Sigmoid这种有固定边界并且能抑制过大值的激活函数。

```
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
```

创建全连接层 先把卷积结果全部flatten 再使用ReLU激活函数进行非线性化 （非零w1防止过拟合）

```
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))                                      
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
```

计算softmax主要是为了计算loss，这里把softmax操作整合到后面计算loss部分

```
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

```

>设计CNN主要就是安排卷积层，池化层，全连接层的分布和排序，以及其中超参数的设置，trick的使用等。设计性能良好的模型是有规律可循，但是针对特定问题设计最合适的网络结构是需要大量实践摸索的。


### 计算CNN的loss 选择优化器

定义loss 将softmax和cross entropy loss 二者的计算合在一起后再求均值 加入losses后计算cross entropy和weight L2 loss 的总和

```
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
```

调用loss函数 使用Adam优化器优化loss

```
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #0.72
```

>tf.nn.in_top_k(predictions, targets, k, name=None) 这个函数的作用是返回一个布尔向量，说明对于某一行的predictions的最k大的索引是否有索引targets。

 **EXAMPLE**
 ```
 targets
 [1,0,1]

 predictions
 [[0.24530381 0.8667579  0.19730636 0.78550184]
 [0.87574047 0.18356404 0.36066866 0.6147065 ]
 [0.2539505  0.17513384 0.3122791  0.60933316]]
 
 results
 [ True  True False]
```

求出结果中top k的准确率 1 -> 输出分数最高的那一类的准确率

```
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
```



### 训练

初始化全部模型参数

```
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
```

图片数据增强需要用到的线程队列 一共使用16个线程来进行加速

```
tf.train.start_queue_runners()
```

每10个step输出训练信息便于监控整个训练过程

```
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _, loss_value = sess.run([train_op, loss],feed_dict={image_holder: image_batch, 
                                                         label_holder:label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
    
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
```


### 评测模型在测试集上的准确率

先获取每个batch的测试集 再执行top_k_op计算模型在这个batch的top 1 上预测正确的样本数 最后汇总所有预测正确的结果

```
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0  
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
                                                 label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

```

输出在测试集上的准确率

```
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
```

# 总结
>持续增大max_steps可以期望准确率逐渐增加，如果max_steps比较大则推荐使用学习速率衰减(decay)的SGD训练，这样训练过程能达到的准确率峰值会比较高，大致接近86%。而其中的L2正则和LRN层的使用一定程度缓解了过拟合，提升了模型的泛化性。


# Q & A

- 卷积层加池化层的作用？
- 全连接层的作用？
- 防止CNN过拟合有哪些方法？
- 优化算法有哪些？

---

- 图像识别的标准组件 提取高阶特征
- 对特征进行组合匹配 进行分类
- 加快收敛速度 提高泛化性
- [深度学习优化算法总结](https://linzhenyuyuchen.github.io/2019/09/01/深度学习优化算法总结/)