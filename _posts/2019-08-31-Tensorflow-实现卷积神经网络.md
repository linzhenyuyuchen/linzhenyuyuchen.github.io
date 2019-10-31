---
layout:     post
title:      Tensorflow 实现卷积神经网络
subtitle:   CNN简介 基础和进阶实现
date:       2019-08-31
author:     LZY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Tensorflow
    - 卷积神经网络
    - CNN
---

# 卷积神经网络简介

>**CNN** 应用于图像识别，音频信号，文本数据。最大特点在于卷积的权值共享结构，可以大幅减少神经网络的参数量，防止过拟合的同时降低模型的复杂度。

### 卷积层中的操作

- 图像通过多个不同的卷积核的滤波，并加偏置，提取出局部特征
- 将前面的滤波输出结果，进行非线性的激活函数处理(Sigmoid / ReLU)
- 对激活函数的结果再进行池化操作（降采样），最大池化和平均池化
- 其它trick：LRN（Local Response Normalization, 局部响应归一化层），Batch Normalization

### 减少需要训练的权重数量

- 降低计算的复杂度
- 过多的连接会导致严重的过拟合，减少连接数可以提升模型的泛化性


### 相关

- 需要训练的权值数量只与卷积核大小和卷积核数量有关
- 隐含节点数量只与卷积的步长有关 步长为1隐含节点的数量与输入图像的像素点数目一致 步长为5每5x5的像素才需要一个隐含节点，则隐含节点的数量为输入图像的像素点数目的1/25

# Tensorflow 实现卷积神经网络（简单）

### 初始化函数
给权重制造噪声避免打破完全对称，比如截断的正态分布，标准差设为0.1

```
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
```


给偏置增加一些小的正值0.1来避免死亡节点(dead neurons)

```
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

2维卷积函数 输入x 卷积参数W 步长strides 边界处理方式padding（SAME代表给边界加上padding让卷积的输入输出保持同样的尺寸）
```
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```

最大池化函数 保留最显著的特征 

```
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
```

输入特征x和标签y_ 将一维784转为28x28的2D图片
-1代表样本数量不定 1代表通道数量为1

```
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
```

参数W和b初始化 卷积核尺寸5x5 1个颜色通道 32个不同的卷积核即提取32个特征 使用conv2d函数进行卷积操作并加上偏置系数

```
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

使用ReLU函数进行非线性激活处理 最大池化操作

```
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

第二个卷积层 卷积核数量变为64即提取64个特征

```
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

至此已经经历了两次2x2池化操作，所以边长只有1/4图片尺寸从28x28变成7x7

使用tf.reshape函数将二维转换为一维向量，然后连接一个全连接层，隐含节点为1024，并使用ReLU激活函数

```
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

Dropout层减轻过拟合

```
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

连接Softmax层得到概率输出

```
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

定义损失函数cross entropy / 优化器Adam 学习率

```
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

定义评测准确率的操作

```
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```


训练 每100次进行准确率评测 keep_prob为1用以实时监测模型的性能
```
tf.global_variables_initializer().run()
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
```

测试集预测 得到整体的分类准确率
```
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```


>CNN模型对MNIST手写数字识别的准确率在99.2%，优于MLP的2%错误率，主要得益于卷积网络对图像特征的提取和抽象能力。依靠卷积核的权值共享，减少了参数量，降低计算量的同时也减轻了过拟合。

# Q & A

### 常用的损失函数

Reference: [towardsdatascience](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)

根据回归和分类，损失函数可分为两类。

回归

- Mean Square Error 均方差 / Quadratic Loss 平方损失 / L2 Loss
- Mean Absolute Error 平均绝对误差 / L1 Loss
- Mean Bias Error 平均误差

分类

- Hinge Loss 合页损失 / Multi class SVM Loss
- Cross Entropy Loss 交叉熵损失 / Negative Log Likelihood **(Most Common Setting)**
