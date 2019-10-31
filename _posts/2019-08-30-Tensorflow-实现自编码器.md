---
layout:     post
title:      Tensorflow 实现自编码器
subtitle:   自编码器的简介，分类和实现
date:       2019-08-30
author:     LZY
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - Tensorflow
    - 自编码器
    - 多层感知机
---

# 自编码器简介
>自编码器作为一种无监督学习的方法，与其它无监督学习的主要不同之处在于它不是对数据进行聚类，而是提取其中最有用最频繁出现的高阶特征，根据这些高阶特征重构数据。

### **使用少量稀疏的高阶特征来重构输入而不是单纯的逐个复制输入，加入几种限制：**
- 限制中间隐含层的节点数目，相当于降维的过程，只能学习数据中最重要的特征复原
- 给中间隐含层的权重加一个L1的正则，根据惩罚系数控制隐含节点的稀疏程度，惩罚系数越大，学到的特征组合越稀疏，实际使用的特征数量越少
- 给数据加入噪声（常用加性高斯噪声），从噪声中学习出数据的特征，即去噪自编码器


### 自编码器分类
- 无噪声
- 高斯噪声
- 随机遮挡噪声 Making Noise
- Variational AutoEncoder (VAE) 对中间节点的分布具有强假设，拥有额外的损失项，且会使用特殊的Stochastic Gradient Variational Bayes算法进行训练

# Tensorflow实现自编码器


- **导入常用库numpy和数据预处理模块sklearn.preprocessing**
```
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```

- **xavier initialization 定义均匀分布或者高斯分布的初始化器**
```
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)
```
- **定义去噪自编码的类class**
```
class AdditiveGaussianNoiseAutoencoder(object):
```
1. 构建函数
```
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']),
                self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
```

2. 参数初始化函数，w1初始化为激活函数的权重初始分布，对于self.reconstruction没有使用激活函数，所以w2，b2全部初始化为0

```
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
```

3. 定义计算cost和训练步骤的函数，用一个batch数据进行训练并返回当前的损失cost

```
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,
                                                                            self.scale: self.training_scale
                                                                            })
        return cost
```

4. 自编码器自训练完毕后，在测试集上对模型性能评测的cost函数

```
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
                                                     self.scale: self.training_scale
                                                     })
```

5. 获取抽象后的特征，返回自编码器隐含层的输出结果

```
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
                                                       self.scale: self.training_scale
                                                       })
```

6. 将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据

```
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
```

7. 整体运行一遍复原过程，包括提取高阶特征transform和通过高阶特征复原数据generate

```
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
                                                               self.scale: self.training_scale
                                                               })
```

8. 获取隐藏层的权重w1

```
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
```

9. 获取隐藏层的偏置系数b1

```
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
```

- **载入MNIST数据集**

```
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
```

- **定义一个对训练，测试数据进行标准化处理的函数，使用sklearn.preprossing的StandardScaler类在训练集上进行fit后用到训练数据和测试数据，共用一个Scaler保证处理数据的一致性**

```
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
```

- **定义一个随机获取batch size大小的block数据函数，不放回抽样提高数据的利用效率**

```
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
```

- **对训练集和测试集进行标准化变换**

```
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
```

- **定义常用参数**

```
n_samples = int(mnist.train.num_examples)#总训练样本数
training_epochs = 20#最大训练的轮数
batch_size = 128#每次抽样大小
display_step = 1#每隔一个epoch显示一次cost

```

- **创建AGN自编码器实例**

```
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                               scale = 0.01)
```

- **开始训练过程，每一轮epoch使用不放回抽样训练并显示本轮迭代的平均cost，每一个batch循环得到训练的cost并整合到avg_cost中**

```
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
```

- **对训练完的模型进行性能测试**

```
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

```

>实现一个自编码器与实现一个单隐含层的神经网络差不多，只不过是在数据输入时做了标准化，并加上了一个高斯噪声。同时，我们的输出结果不是数字分类结果，而是复原数据，因此不需要用标注过的数据进行监督训练。