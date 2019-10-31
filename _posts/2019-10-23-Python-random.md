---
layout:     post
title:      Python random
subtitle:   from multiprocessing import Pool
date:       2019-10-23
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Python
    - random
---

# random

```python
import random

print( random.randint(1,10) )        # 产生 1 到 10 的一个整数型随机数  
print( random.random() )             # 产生 0 到 1 之间的随机浮点数
print( random.uniform(1.1,5.4) )     # 产生  1.1 到 5.4 之间的随机浮点数，区间可以不是整数

print( random.choice('abcdefghijklmnopqrstuvwxyz!@#$%^&*()') )   # 从序列中随机选取一个元素
print( random.randrange(1,100,2) )   # 生成从1到100的间隔为2的随机整数

rand_str = string.ascii_letters + string.digits
print( ''.join( random.sample(rand_str,5) ) )   # 多个字符中生成指定数量的随机字符

a=[1,3,5,6,7]                # 将序列a中的元素顺序打乱
random.shuffle(a)
print(a)
```

# random.seed()

>seed() 方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数

```python
import random

random.seed ( [x] )
```

我们调用 random.random() 生成随机数时，每一次生成的数都是随机的。但是，当我们预先使用 random.seed(x) 设定好种子之后，其中的 x 可以是任意数字，如10，这个时候，先调用它的情况下，使用 random() 生成的随机数将会是同一个。

注意：seed()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。

```python
import random

print random.random()
print random.random()

print "------- 设置种子 seed -------"
random.seed( 10 )
print "Random number with seed 10 : ", random.random()

# 生成同一个随机数
random.seed( 10 )
print "Random number with seed 10 : ", random.random()

# 生成同一个随机数
random.seed( 10 )
print "Random number with seed 10 : ", random.random()
```