---
layout:     post
title:      Tabular data learning
subtitle:   Tabnet, NODE
date:       2021-07-18
author:     LZY at Tencent AI Lab
header-img: img/bg-20210718.jpg
catalog: true
tags:
    - tabular data
typora-root-url: ../
---


# Tabular data learning

[reference](https://towardsdatascience.com/the-unreasonable-ineffectiveness-of-deep-learning-on-tabular-data-fd784ea29c33)

- Kaggle等数据挖掘竞赛上，对于表格数据，基本上用的是决策树模型，XGBoost和LightGBM这类提升（Boosting）树模型已经成为了现在数据挖掘比赛中的标配；优点：可解释性好，模型的决策流形（decision manifolds）是可以看成是超平面边界的，对于表格数据的效果很好。

- DNN对表格数据进行encode的表征学习，可以减少对特征工程的依赖；缺点：模型过参数化。

### Tabnet, AAAI

[tensorflow](https://github.com/ptuls/tabnet-modified), [torch](https://github.com/dreamquark-ai/tabnet)

> 保留DNN的end-to-end和representation learning特点的基础上，还拥有了树模型的可解释性和稀疏特征选择的优点



![img](https://linzhenyuyuchen.github.io/img/v2-22d876a63f96fe1db4e8472f11f35380_1440w.jpg)

##### 流程

- 输入是特征向量[x1, x2] 

- 首先分别通过两个Mask层来将 x1 和 x2 单独筛选出来
- 然后通过一个weight和bias都被专门设定过的全连接层
- 并将两个FC层的输出通过ReLU激活函数后相加起来
- 最后经过一个Softmax激活函数作为最终输出。



与决策树的流程进行对比，这个神经网络的每一层都对应着决策树的相应步骤：

- Mask层对应的是决策树中的特征选择
- FC层+Relu对应阈值判断，以 x1为例，通过一个特定的FC层+ReLU之后，可以保证输出的向量里面只有一个是正值，其余全为0，而这就对应着决策树的条件判断
- 最后将所有条件判断的结果加起来，再通过一个Softmax层得到最终的输出
- 这个输出向量可以理解成一个权重向量，它的每一维代表某个条件判断的对最终决策影响的权重



##### 模型结构

![img](https://linzhenyuyuchen.github.io//img/v2-24d90e6099f976b535f6fac2072fd5a6_1440w.jpg)



- BN层：即batch normalization层
- Feature transformer层：其作用与之前的FC层类似，都是做**特征计算**



##### Feature transformer

![img](https://linzhenyuyuchen.github.io//img/v2-79e32895cdd49a0672e8e5b1bc0268f2_1440w.jpg)

> 特征计算: 对于给定的一些特征，一棵决策树构造的是**单个特征的大小关系的组合**

- 前半部分层的参数是共享的，也就是说它们是在所有step上共同训练的，先用同样的层来做特征计算的**共性部分**；
- 而后半部分则没有共享，在每一个step上是分开训练的，做每一个step的**特性部分**。



##### Attentive transformer

> 特征选择: 根据上一个step的结果得到当前step的Mask矩阵，并尽量使得Mask矩阵是**稀疏且不重复**的

- 根据上一个step的结果，计算出当前step的Mask层



#### 自监督学习

> 同一样本的不同特征之间是有关联的，自监督学习就是先人为mask掉一些feature，然后通过encoder-decoder模型来对mask掉的feature进行预测。通过这样的方式训练出来的encoder模型，可以有效地将表征样本的feature，再将encoder模型由于回归或分类任务。

![img](https://linzhenyuyuchen.github.io//img/v2-b8973da542872f102d427c5752ca0cc3_1440w.jpg)



#### 可解释性

> Feature attribute: 可以计算出每个step对最后结果的贡献，从而计算全局重要性。



### Neural Oblivious Decision Ensembles (NODE), ICLR2020



