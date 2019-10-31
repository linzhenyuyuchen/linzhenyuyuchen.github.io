---
layout:     post
title:      Tensorflow 安装及识别手写数字
subtitle:   Anaconda 作为Python环境
date:       2019-08-29
author:     LZY
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Tensorflow
    - Python
    - 识别手写数字
---
# 前言

>**Anaconda**是目前最好的科学计算的Python环境，集成了Tensorflow依赖库/Python依赖库/底层数值计算库。



# 安装 Tensorflow for windows


#### 在Anaconda中执行操作命令


CPU版本
```pip install tensorflow```

GPU版本
```pip install tensorflow-gpu```

安装出错则卸载重新安装
```
pip uninstall tensorflow
pip install tensorflow
```

# Softmax Regression 识别手写数字

>机器学习领域的Hello World项目 MNIST手写数字识别

- **加载MNIST数据**

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

- **查看数据集样本标注等信息**

```
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
```


- **InteractiveSession注册默认Session**


TF将forward和backward内容都自动实现，自动求导和梯度下降，不断使得loss减小，直到达到局部最优解或全局最优解，从而完成参数的自动学习


```
import tensorflow as tf
sess = tf.InteractiveSession()
```

- **输入数据 None代表不限输入条数 784代表每条输入向量维度**

```
x = tf.placeholder(tf.float32, [None, 784])
```

- **初始化weights和biases为0 10代表0-9的独热编码维度**

```
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```


- **Softmax Regression**

```
y = tf.nn.softmax(tf.matmul(x, W) + b)
```
tf.nn 包含大量神经网络组件
tf.matmul 是矩阵乘法函数


- **定义Loss Function 描述模型对问题的分类精度**

Cross-entropy 用来判断模型对真实概率分布估计的准确程度
y  是预测的概率粉补
y_ 是真实的概率分布


```
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```
tf.reduce_mean 求均值
tf.reduce_sum 求和


- **优化算法 使用常见的随机梯度下降算法SGD (Stochastic Gradient Descent)**

```
GradientDescentOptimizer #函数名可根据算法更换
```

- **训练操作train_step 设置学习速率0.5 优化目标为loss**

```
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```


- **全局变量初始化操作**

```
tf.global_variables_initializer().run()
```


- **迭代执行训练操作 每次随机抽取100个样本构成mini-batch feed给placeholder**

使用一小部分样本进行训练称之为随机梯度下降

```
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
```

- **验证模型准确率**

判断预测的概率最大的数字和真实的概率最大的数字是否一样

```
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
```

将bool值转换成float32后取平均值就得到准确率

```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

```

- **将测试数据输入评测流程accuracy计算模型在测试集上的准确率**

```
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
```

>Softmax Regression的准确率在92%，但是还达不到实用的程度，后面学习多层感知机和卷积网络。事实上在现有基础上加入隐含层变成一个正统的神经网络后，再结合Dropout,Adagrad,ReLU等技术准确率就可以到达98%。引入卷积层，池化层后也可以达到99%的正确率，而目前基于卷积神经网络的state-of-the-art的方法已经可以达到99.8%的正确率。