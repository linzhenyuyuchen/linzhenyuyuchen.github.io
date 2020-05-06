---
layout:     post
title:      Python matplotlib
subtitle:   matplotlib.pyplot 数据可视化
date:       2019-09-14
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Python
---

# matplotlib.pyplot

[Reference](https://matplotlib.org/api/pyplot_api.html)

## 实例

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1,11)
plt.plot(x,x,label='linear')
plt.plot(x,x*2,label='quadratic')
plt.plot(x,x*3,label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('simple plot')
plt.legend()
plt.show()
```

![](/img/matplot_simple.png)

## subplot()

>subplot() 在同一图中绘制不同的东西

```python
import numpy as np 
import matplotlib.pyplot as plt 
x = np.arange(0,  3  * np.pi,  0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
# 建立 subplot 网格，高为 2，宽为 1  
# 激活第一个 subplot
plt.subplot(2,  1,  1)  
# 绘制第一个图像 
plt.plot(x, y_sin) 
plt.title('Sine')  
# 将第二个 subplot 激活，并绘制第二个图像
plt.subplot(2,  1,  2) 
plt.plot(x, y_cos) 
plt.title('Cosine')
plt.show()
```

## bar()

>pyplot 子模块提供 bar() 函数来生成条形图

```python
from matplotlib import pyplot as plt
x =  [5,8,10]
y =  [12,16,6]
x2 =  [6,9,11]
y2 =  [6,15,7]
plt.bar(x, y, align =  'center')
plt.bar(x2, y2, color =  'g', align =  'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()
```

## 折线图

[DEMO](https://www.matplotlib.org.cn/tutorials/introductory/usage.html#matplotlib-pyplot-and-pylab-how-are-they-related)

```python
plt.plot(list_data, c='orange', label='GA') # 设置颜色 数据标签
plt.xticks(np.arange(0,160,10)) # 刻度间距
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('AUC',fontsize=14) # 标签
plt.xlabel('Iteration',fontsize=14)

plt.legend()
plt.show()
plt.savefig("ga-auc.png", dpi=2) # 保存图片
```