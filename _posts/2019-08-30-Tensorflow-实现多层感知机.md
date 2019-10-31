---
layout:     post
title:      Tensorflow 实现多层感知机
subtitle:   多层感知机的简介和实现
date:       2019-08-30
author:     LZY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Tensorflow
    - 自编码器
    - 多层感知机
---

# 多层感知机简介

>**Multi-Layer Perception (MLP)** 也叫人工神经网络（ANN，Artificial Neural Network），除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构

多层感知机层与层之间是全连接的（全连接的意思就是：上一层的任何一个神经元与下一层的所有神经元都有连接）。多层感知机最底层是输入层，中间是隐藏层，最后是输出层。


- 隐藏层的输出就是f(W1X+b1)，W1是权重（也叫连接系数），b1是偏置系数，激活函数f可以是常用的sigmoid函数或者tanh函数
- 隐藏层到输出层可以看成是一个多类别的逻辑回归，即softmax回归，所以输出层的输出就是softmax(W2X1+b2)，X1表示隐藏层的输出f(W1X+b1)

MLP所有的参数就是各个层之间的连接权重以及偏置系数，包括W1、b1、W2、b2。对于一个具体的问题，怎么确定这些参数？

求解最佳的参数是一个最优化问题，解决最优化问题，最简单的就是随机梯度下降法（SGD）：首先随机初始化所有参数，然后迭代地训练，不断地计算梯度和更新参数，直到满足某个条件为止（比如误差足够小、迭代次数足够多时）。

这个过程涉及到代价函数(cost function)、规则化(Regularization)、学习速率(learning rate)、梯度计算等。

理论上只要隐含节点足够多即使只有一个隐含层的神经网络也可以拟合任意函数，同时隐含层越多，越容易拟合复杂函数。

有理论研究表明，为了拟合复杂函数需要的隐含节点数目随着隐含层的数目增加呈指数下降的趋势，即层数越多，概念越抽象，节点越少。

### 层数越多，神经网络会遇到许多困难：

- 容易过拟合 泛化性不好，只是记忆了当前数据的特征，不具备推广能力，可能会产生参数比数据还多的情况

- 解决方法：Dropout 在训练时将某一层的输出节点数据随机丢弃一部分，创造出更多的随机样本，通过增大样本量，减少特征数量来防止过拟合


- 参数难以调试 尤其是SGD的参数(学习速率，Momentum,Nesterov..)，设置不同的学习速率会导致得到的结果差异巨大。开始的时候希望学习速率大一点，可以加速收敛，后期又希望学习速率小一点可以比较稳定地落入一个局部最优解。

- 解决方法：不同的机器学习问题所需要的学习速率也不太好设置，需要反复调试，因此就有像Adagrad,Adam,Adadelta等自适应的方法可以减轻调试参数的负担

- 梯度弥散 激活函数Sigmoid函数在反向传播中梯度值会越来越小，经过多层传递后会呈指数级急剧减小，因此梯度值在传递前几层时以及非常小，这种情况下根据训练数据的反馈来更新训练参数将会非常缓慢，基本起不到训练的作用

- 解决方法：激活函数ReLU 线性整流函数（Rectified Linear Unit, ReLU）y=max(0,x) 非常类似于人脑的阈值响应机制，可以很好地传递梯度，经过多层的反向传播，梯度依旧不会大幅减小。目前，ReLU及其变种EIU、PReLU、RReLU已经成为最主流的激活函数。实践中大部分情况下（包括MLP、CNN、RNN）将隐含层的激活函数从Sigmoid替换为ReLU都可以带来训练速度和模型准确率的提升。当然神经网络的输出层一般都是Sigmoid函数，因为它最接近概率输出分布。


# Tensorflow实现多层感知机
>增加隐含层，并使用减轻过拟合的Dropout，自适应速率的Adagrad以及可以解决梯度弥散的激活函数ReLU

- **创建模型**
```
# Create the model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
```

- **参数初始化 W1权重初始化为截断的正态分布给参数增加一点噪声来打破完全对称并且避免0梯度 在其它一些模型中 还需要给偏置系数赋值小的非零值来避免dead neuron**
```
in_units = 784#输入节点数
h1_units = 300#隐含层输出节点数
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
```

- **Sigmoid函数在在0附近敏感，梯度最大**
```
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
```

- **Dropout中keep_prob是保留节点的比率（保留节点的概率）是不一样的，通常在训练的时候小于1，预测时等于1**
```
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
```

- **keep_prob在训练时小于1用来制造随机性，防止过拟合，预测时等于1用全部特征来预测样本的类别**
```
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
```

- **定义损失函数 选择优化器优化loss**

```
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

```

- **训练 因为加入了隐含层所以需要更多的数据训练3000个batch 每个batch包含100个样本 相当于对全数据进行5轮（epoch）迭代**

```
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
```

- **对模型进行准确率预测**

```
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```


>增加隐含层对准确率的提升有显著的效果，虽使用了Dropout,Adagrad,ReLU等trick，但是起决定性作用的还是隐含层本身，它能对特征进行抽象和转化。没有隐含层的Softmax Regression只能从像素点推断是哪个数字，而多层神经网络依靠隐含层可以组合成高阶特征，比如横线，竖线，圆圈等，之后再将这些高阶特征或者说组件再组合成数字，就能实现精准的匹配和分类

>使用全连接神经网络也是有局限的，再深的网络，再多的epoch迭代，再多的隐藏节点也很难达到99%以上的准确率，那么将使用卷积神经网络进行优化，从而达到银行支票识别这种高精度系统的 